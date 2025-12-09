import os
import argparse
import json
import random
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import torch
import torch.nn as nn
import gc
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
# Two options to run:
# 1) python -m src.train_qformer ...
# 2) PYTHONPATH=src python src/train_qformer.py ...

try:
    from models.dit import DiT_Llama
    from diffusion.ddpm import ddpm_schedules
    from image_caption_data import (
        get_text_captions,
        cache_all_text_states,
        CIFARTextDataset,
        build_text_states,
        get_cifar_names,
        get_synonyms,
    )
except ImportError:  # package 模式時使用相對 import
    from .models.dit import DiT_Llama
    from .diffusion.ddpm import ddpm_schedules
    from .image_caption_data import (
        get_text_captions,
        cache_all_text_states,
        CIFARTextDataset,
        build_text_states,
        get_cifar_names,
        get_synonyms,
    )

def setup_optimizer_and_scheduler(model: DiT_Llama, args: argparse.Namespace):
    """
    Setup the optimizer and scheduler for the model given the arguments in args.
    
    The parameters of QFormer should be trainable, and all of the DiT parameters should be frozen.
    Set up the AdamW optimizer with the given arguments, as well as a scheduler with linear warmup if the warmup_steps argument is set.
    set the criterion to be MSE loss.
    Return the optimizer, scheduler, and criterion.
    """
    # TODO: Set the correct parameters to be trainable, set up the optimizer and scheduler, and the criterion
    # ====== BEGIN STUDENT SOLUTION ===========================================
    # 1) Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    # 2) Unfreeze only the Q-Former (query_embedder) parameters
    if model.query_embedder is None:
        raise ValueError("QueryEmbedder is None – make sure use_text_conditioning=True when building the model.")
    for p in model.query_embedder.parameters():
        p.requires_grad = True

    # Collect trainable parameters (only Q-Former at this point)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 3) Create AdamW optimizer
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
        betas=(0.9, 0.95),
    )

    # 4) Learning-rate scheduler with optional linear warmup
    if getattr(args, "warmup_steps", 0) and args.warmup_steps > 0:
        def lr_lambda(step: int):
            # Linearly ramp from 0 -> 1 over warmup_steps, then keep 1.0
            return min(1.0, float(step + 1) / float(args.warmup_steps))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # 5) Criterion: MSE loss
    criterion = nn.MSELoss()
    # ====== END STUDENT SOLUTION =============================================
    return optimizer, scheduler, criterion

def maybe_load_resume_states(model: DiT_Llama, args: argparse.Namespace, device: torch.device, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LambdaLR, steps_per_epoch: int):
    # If provided, load model/optimizer states for resuming
    if args.resume_model_path is not None and os.path.isfile(args.resume_model_path):
        resume_state = torch.load(args.resume_model_path, map_location=device)
        model.load_state_dict(resume_state, strict=True)
    if args.resume_optimizer_path is not None and os.path.isfile(args.resume_optimizer_path):
        opt_state = torch.load(args.resume_optimizer_path, map_location=device)
        optimizer.load_state_dict(opt_state)
    if scheduler is not None and args.start_epoch > 0:
            for _ in range(args.start_epoch * steps_per_epoch):
                scheduler.step()
    return model, optimizer, scheduler

def run_qformer_inference(
    model: DiT_Llama,
    sched: Dict[str, torch.Tensor],
    classes: Optional[torch.Tensor],
    text_states: Optional[torch.Tensor],
    text_masks: Optional[torch.Tensor],
    image_size: int,
    channels: int,
    n_T: int,
    cfg_scale: float,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if classes is not None:
            B = classes.size(0)
        else:
            B = text_states.size(0)
        x_i = torch.randn(B, channels, image_size, image_size, device=device)
        for i in range(n_T, 0, -1):
            if hasattr(model, "_attn_capture_ctx") and isinstance(model._attn_capture_ctx, dict):  # type: ignore[attr-defined]
                model._attn_capture_ctx["timestep"] = int(i)  # type: ignore[attr-defined]
            t_scalar = torch.full((B,), float(i) / n_T, device=x_i.device)
            if cfg_scale > 0.0:
                # Conditional path: use QFormer text states
                eps_cond = model(x_i, t_scalar, classes, text_hidden_states=text_states, text_attention_mask=text_masks)
                # Unconditional path: use unconditional embedding from pretrained DiT model (before qformer training)
                cfg_index = getattr(model.y_embedder, "num_classes", 10)
                y_uncond = torch.full((B,), int(cfg_index), device=device, dtype=torch.long)
                eps_uncond = model(
                    x_i,
                    t_scalar,
                    y_uncond,
                    text_hidden_states=None,
                )
                eps_hat = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
            else:
                eps_hat = model(x_i, t_scalar, classes, text_hidden_states=text_states, text_attention_mask=text_masks)

            oneover_sqrta = sched["oneover_sqrta"][i]
            mab_over_sqrtmab = sched["mab_over_sqrtmab"][i]
            sqrt_beta_t = sched["sqrt_beta_t"][i]
            z = torch.randn_like(x_i) if i > 1 else 0
            x_i = oneover_sqrta * (x_i - eps_hat * mab_over_sqrtmab) + sqrt_beta_t * z
        return x_i


def train_cifar10_qformer_dense_captions(args):
    device = torch.device(f"cuda:{args.device}")
    
    channels = 3
    num_classes = 10
    image_size = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # index to captions maps an index to a list with the original CIFAR-10 caption (i.e. "a photo of a airplane") and the dense caption for that index
    # captions per class maps a class index to a list of all dense captions for that class along with the original CIFAR-10 caption
    index_to_captions, captions_per_class = get_text_captions(args.dense_captions_path)

    os.makedirs(args.data_dir, exist_ok=True)
    base_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    cifar10_names = get_cifar_names()
    synonyms = get_synonyms()

    wandb.init(project=f"qformer_training", name=f"initial_path_{os.path.basename(args.pretrained_model_path).split('.')[0]}_{args.epochs}_epochs_{args.batch_size}_batch_size")

    model = DiT_Llama(
        in_channels=channels,
        input_size=image_size,
        dim=256,
        n_layers=10,
        n_heads=8,
        num_classes=num_classes,
        use_text_conditioning=True, # this turns on the qformer
        transformer_hidden_size=args.transformer_hidden_size,
        num_query_tokens=args.num_query_tokens,
    ).to(device)

    state = torch.load(args.pretrained_model_path, map_location=device)
    model.load_state_dict(state, strict=False)

    optimizer, scheduler, criterion = setup_optimizer_and_scheduler(model, args)

    sched = ddpm_schedules(args.beta1, args.beta2, args.n_T)
    for k in list(sched.keys()):
        sched[k] = sched[k].to(device)

    os.makedirs(args.gpt2_cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir=args.gpt2_cache_dir)
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=args.gpt2_cache_dir).to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Wrap CIFAR into text-conditioned dataset using only aggregated cached embeddings
    # Precompute aggregated text states if missing
    cache_file = args.cache_text_embeddings
    if not os.path.isfile(cache_file):
        os.makedirs(os.path.dirname(cache_file) or '.', exist_ok=True)
        cache_all_text_states(
            base_dataset,
            tokenizer,
            gpt2,
            args.gpt2_layer_index,
            device,
            cifar10_names,
            synonyms,
            args.synonym_replace_prob,
            index_to_captions=index_to_captions,
            captions_per_class=captions_per_class,
            output_file=cache_file,
            batch_size=max(64, args.batch_size),
        )

    dataset = CIFARTextDataset(
        base_dataset,
        device,
        args.caption_use_prob,
        args.synonym_replace_prob,
        cache_file=cache_file,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, collate_fn=dataset.collate_fn)

    # load from optimizer and model checkpoints if provided, i.e. if your training was interrupted, you can resume from the checkpoint
    model, optimizer, scheduler = maybe_load_resume_states(model, args, device, optimizer, scheduler, len(dataloader))


    # Setup resume and checkpointing utilities
    steps_per_epoch = len(dataloader)
    start_epoch = args.start_epoch
    global_step = start_epoch * steps_per_epoch
    ckpt_dir = args.optimizer_ckpt_dir if args.optimizer_ckpt_dir else os.path.join(args.save_dir, "optimizer_ckpts")

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    for epoch in range(args.epochs):
        # Skip full epochs if resuming beyond them
        if epoch < start_epoch:
            continue
        pbar = tqdm(dataloader)
        loss_ema = None
        for step_idx, (xy, y) in enumerate(pbar):
            x_img, text_states, text_masks = xy
            x = x_img.to(device)
            y = y.to(device)
            text_masks = text_masks.to(device)

            b = x.size(0)
            optimizer.zero_grad()

            t_int = torch.randint(1, args.n_T + 1, (b,), device=x.device)
            t = t_int.float() / args.n_T
            eps = torch.randn_like(x)

            # add noise to the image
            x_t = sched["sqrtab"][t_int, None, None, None] * x + sched["sqrtmab"][t_int, None, None, None] * eps

            # model forward pass
            eps_pred = model(x_t, t, y, text_hidden_states=text_states, text_attention_mask=text_masks)

            loss = criterion(eps_pred, eps)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            wandb.log({"lr": optimizer.param_groups[0]["lr"]})

            loss_item = loss.item()
            wandb.log({"loss": loss_item})

            if loss_ema is None:
                loss_ema = loss_item
            else:
                loss_ema = 0.9 * loss_ema + 0.1 * loss_item
                wandb.log({"loss_ema": loss_ema})
            pbar.set_description(f"qformer dense loss: {loss_ema:.4f}")
            global_step += 1

        model.eval()
        with torch.no_grad():
            cond_classes = torch.arange(0, 16, device=device) % num_classes
            text_states, text_masks, used_prompts = build_text_states(
                cond_classes, tokenizer, gpt2, args.gpt2_layer_index, device,
                cifar10_names, synonyms,
                args.caption_use_prob, args.synonym_replace_prob,
                captions_per_class=captions_per_class,
            )

            x_i = run_qformer_inference(
                model, sched, cond_classes, text_states, text_masks, image_size, channels, args.n_T, args.cfg, device
            )
            x_vis = x_i * 0.5 + 0.5
            x_vis = x_vis.clamp(0, 1)
            grid = make_grid(x_vis.float(), nrow=4)
            img = grid.permute(1, 2, 0).detach().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(args.save_dir, f"sample_qformer_dense_{epoch}_last.png"))

            try:
                wandb.log({
                    "samples/grid": wandb.Image(grid, caption="epoch %d dense grid" % epoch)
                })
                per_sample_images = []
                for idx in range(x_vis.size(0)):
                    single = x_vis[idx]
                    caption = used_prompts[idx] if idx < len(used_prompts) else ""
                    per_sample_images.append(wandb.Image(single, caption=caption))
                wandb.log({"samples/per_sample": per_sample_images})
            except Exception:
                pass
            os.makedirs(os.path.dirname(args.save_model_path), exist_ok=True)
            torch.save(model.state_dict(), args.save_model_path)

            # Save optimizer (and paired model weights) every N epochs
            if (epoch + 1) % int(args.optimizer_ckpt_interval) == 0:
                try:
                    os.makedirs(ckpt_dir, exist_ok=True)
                    opt_path = os.path.join(ckpt_dir, f"optimizer_epoch_{epoch+1}.pt")
                    mdl_path = os.path.join(ckpt_dir, f"model_epoch_{epoch+1}.pth")
                    torch.save(optimizer.state_dict(), opt_path)
                    torch.save(model.state_dict(), mdl_path)
                except Exception:
                    pass

            try:
                del x_i, text_states
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            try:
                gc.collect()
            except Exception:
                pass
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPM training with DiT + QFormer using dense captions (CIFAR10)")
    parser.add_argument("--pretrained_model_path", type=str, default="./ddpm_dit_cifar.pth", help="Path to the pretrained DiT model")
    parser.add_argument("--dense_captions_path", type=str, default="./data/cifar10_dense_captions.jsonl", help="Path to CIFAR-10 dense captions jsonl")
    parser.add_argument("--data_dir", type=str, default="./data/cifar10", help="Directory to save the data")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for the optimizer")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for the AdamW optimizer")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for the training")
    parser.add_argument("--n_T", type=int, default=1000, help="Number of timesteps for the DDPM scheduler")
    parser.add_argument("--beta1", type=float, default=1e-4, help="Beta1 for the DDPM scheduler")
    parser.add_argument("--beta2", type=float, default=2e-2, help="Beta2 for the DDPM scheduler")
    parser.add_argument("--device", type=int, default=0, help="Device to train on")
    parser.add_argument("--save_dir", type=str, default="./contents_qformer_dense_training", help="Directory to save the model and optimizer checkpoints")
    parser.add_argument("--save_model_path", type=str, default="./models/ddpm_dit_cifar_qformer_dense.pth", help="Path to save the model")
    parser.add_argument("--num_query_tokens", type=int, default=4, help="Number of query tokens for the QFormer")
    parser.add_argument("--transformer_hidden_size", type=int, default=768, help="hidden size of GPT-2 base")
    parser.add_argument("--cfg", type=float, default=0.0, help="classifier-free guidance scale at epoch-end sampling")
    parser.add_argument("--warmup_steps", type=int, default=50, help="linear LR warmup steps for query embedder optimizer")
    parser.add_argument("--gpt2_layer_index", type=int, default=-1, help=("If >= 0, use only this 0-based GPT-2 hidden layer for conditioning (keep all tokens). If < 0, use all layers concatenated (default)."))
    parser.add_argument("--gpt2_cache_dir", type=str, default="./data/gpt2", help="HuggingFace cache directory for GPT-2")
    parser.add_argument("--synonym_replace_prob", type=float, default=0.5, help="Probability to replace the class name with a synonym in the caption")
    parser.add_argument("--caption_use_prob", type=float, default=0.8, help="Probability to use the dense caption; else base prompt")
    parser.add_argument("--optimizer_ckpt_interval", type=int, default=5, help="Save optimizer state every N epochs (and paired model weights)")
    parser.add_argument("--optimizer_ckpt_dir", type=str, default=None, help="Directory to save optimizer state checkpoints; defaults to save_dir/optimizer_ckpts")
    parser.add_argument("--resume_optimizer_path", type=str, default=None, help="Path to an optimizer state dict to resume from")
    parser.add_argument("--resume_model_path", type=str, default=None, help="Path to model weights matching the optimizer state")
    parser.add_argument("--start_epoch", type=int, default=0, help="Epoch to start from when resuming")
    parser.add_argument("--cache_text_embeddings", type=str, default="./data/text_embeddings.pt", help="Directory path to cache per-image base and caption text embeddings")
    args = parser.parse_args()
    
    train_cifar10_qformer_dense_captions(args)