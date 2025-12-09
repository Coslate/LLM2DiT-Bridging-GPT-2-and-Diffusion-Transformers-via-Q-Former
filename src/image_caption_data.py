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
import inspect

def get_cifar_names():
    return [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

def get_synonyms():
    return {
        "airplane": ["vehicle that flies"],
        "automobile": ["small vehicle that drives"],
        "bird": ["animal that flies"],
        "cat": ["four legged animal that purrs"],
        "deer": ["four legged animal with antlers"],
        "dog": ["four legged animal that barks"],
        "frog": ["four legged animal that hops"],
        "horse": ["four legged animal with a mane and sometimes a saddle"],
        "ship": ["vehicle that floats"],
        "truck": ["big vehicle that drives"],
    }
    
def _maybe_apply_synonym(text: str, class_index: int, cifar10_names: List[str], synonyms: Dict[str, List[str]], prob: float) -> str:
    class_name = cifar10_names[class_index]
    if random.random() >= prob:
        return text
    candidates = synonyms.get(class_name, [])
    if not candidates:
        return text
    synonym = random.choice(candidates)

    pattern = re.compile(rf"\b{re.escape(class_name)}\b", flags=re.IGNORECASE)
    return pattern.sub(synonym, text)

def get_text_captions(path: str):
    cifar10_names = get_cifar_names()
    num_classes = len(get_cifar_names())
    index_to_captions: Dict[int, List[str]] = {}
    captions_per_class: Dict[int, List[str]] = {i: [] for i in range(num_classes)}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dense captions file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            item = json.loads(line)
            idx_val = int(item["index"])  # original CIFAR-10 train index
            index_to_captions.setdefault(idx_val, []).append(item["caption"])
            if "label_index" in item:
                lbl = int(item["label_index"])
                if 0 <= lbl < num_classes:
                    captions_per_class[lbl].append(item["caption"])

    for idx in range(num_classes):
        if len(captions_per_class[idx]) == 0:
            captions_per_class[idx].append(f"a photo of a {cifar10_names[idx]}")
    return index_to_captions, captions_per_class

# -------------------------------
# Dataset wrapping CIFAR with text states
# -------------------------------
class CIFARTextDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_dataset,
        device: torch.device,
        caption_use_prob: float,
        synonym_replace_prob: float,
        cache_file: Optional[str] = None,
    ):
        self.base_dataset = base_dataset
        self.device = device
        self.caption_use_prob = float(caption_use_prob)
        self.synonym_replace_prob = float(synonym_replace_prob)
        # Aggregated cache path must be provided
        self._agg_cache_path = cache_file
        self._agg_cache = {}
        if self._agg_cache_path is None:
            raise ValueError("cache_file must be provided to CIFARTextDataset (expects aggregated text embedding cache)")
        if os.path.isfile(self._agg_cache_path):
            try:
                loaded = torch.load(self._agg_cache_path, map_location="cpu")
                if isinstance(loaded, dict):
                    self._agg_cache = loaded
                self.valid_indices = list(self._agg_cache.keys())
            except Exception:
                # If loading fails, keep empty cache and let collate_fn raise
                self._agg_cache = {}

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int):
        idx_val = self.valid_indices[idx]
        img, label = self.base_dataset[idx_val]
        return img, int(label), int(idx_val)

    def collate_fn(self, batch: List[Tuple[torch.Tensor, int, int]]):
        imgs, labels, idxs = zip(*batch) # imgs is a list of tensors, labels is a list of integers, idxs is a list of integers
        imgs_t = torch.stack(list(imgs), dim=0) # imgs_t is a tensor of shape (batch_size, 3, 32, 32)
        labels_t = torch.tensor(list(labels), dtype=torch.long) # labels_t is a tensor of shape (batch_size,)
        idxs_t = torch.tensor(list(idxs), dtype=torch.long) # idxs_t is a tensor of shape (batch_size,)

        # Assemble text states from aggregated cache only
        if not self._agg_cache:
            raise RuntimeError("Aggregated text embedding cache is empty. Please precompute using cache_all_text_states.")

        per_prompt_states: List[torch.Tensor] = []
        per_prompt_masks: List[torch.Tensor] = []
        max_len = 0
        for idx_val, lbl in zip(idxs_t.tolist(), labels_t.tolist()):
            rec = self._agg_cache.get(int(idx_val), None)
            if rec is None:
                raise KeyError(f"Index {int(idx_val)} missing in aggregated cache at {self._agg_cache_path}")
            # Choose among base/base_syn vs caption/caption_syn using probabilities
            r = random.random()
            if r >= self.caption_use_prob:
                # base branch: optionally use synonymized base if available
                want_syn = (random.random() < self.synonym_replace_prob) and ("base_syn" in rec)
                key = "base_syn" if want_syn else "base"
                if key not in rec and "base" in rec:
                    key = "base"
            else:
                # Within caption branch, flip for synonymized variant if available
                want_syn = (random.random() < self.synonym_replace_prob) and ("caption_syn" in rec)
                key = "caption_syn" if want_syn else "caption"
                if key not in rec and "caption" in rec:
                    key = "caption"
                if key not in rec:
                    key = "base"
            hs = rec[key]
            # Determine matching mask key if available
            mask_key = None
            if key == "base":
                mask_key = "base_mask"
            elif key == "base_syn":
                mask_key = "base_syn_mask"
            elif key == "caption":
                mask_key = "caption_mask"
            elif key == "caption_syn":
                mask_key = "caption_syn_mask"
            mask = rec.get(mask_key, None)
            if not torch.is_tensor(hs):
                hs = torch.as_tensor(hs)
            if mask is None:
                # Fallback: infer mask by checking for non-zero rows
                if hs.dim() != 2:
                    raise ValueError("Cached hidden state must be 2D (S, C)")
                mask = (hs.abs().sum(dim=1) > 0).to(torch.bool)
            elif not torch.is_tensor(mask):
                mask = torch.as_tensor(mask, dtype=torch.bool)
            per_prompt_states.append(hs)
            per_prompt_masks.append(mask)
            if hs.size(0) > max_len:
                max_len = hs.size(0)

        padded: List[torch.Tensor] = []
        padded_masks: List[torch.Tensor] = []
        for hs, mask in zip(per_prompt_states, per_prompt_masks):
            pad_len = max_len - hs.size(0)
            if pad_len > 0:
                pad = torch.zeros(pad_len, hs.size(1), device=hs.device, dtype=hs.dtype)
                hs = torch.cat([hs, pad], dim=0)
                mask = torch.cat([mask.to(torch.bool), torch.zeros(pad_len, dtype=torch.bool, device=mask.device)], dim=0)
            padded.append(hs)
            padded_masks.append(mask.to(torch.bool))
        states = torch.stack(padded, dim=0).to(self.device)
        masks = torch.stack(padded_masks, dim=0).to(self.device)

        return (imgs_t, states, masks), labels_t

def prompts_to_padded_hidden_states(
    prompts: List[str],
    gpt2,
    tokenizer,
    gpt2_layer_index: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a list of text prompts to a single padded tensor of GPT-2 hidden states.

    Returns the hidden states from that specific layer along with masks as a tuple (hidden_states, masks)
    Hidden states have shape [B, max_seq, C]. max_seq is the length of the longest prompt.
    Padded masks have shape [B, max_seq], with True for valid tokens and False for padding. (dtype bool)
    """
    # TODO: Construct the padded hidden states
    # ====== BEGIN STUDENT SOLUTION ===========================================
    if len(prompts) == 0:
        raise ValueError("prompts must be a non-empty list")

    # Check whether this gpt2 implementation expects a 'prompt_index' kwarg
    sig = inspect.signature(gpt2.__call__)
    accepts_prompt_index = "prompt_index" in sig.parameters

    per_prompt_states: List[torch.Tensor] = []
    seq_lengths: List[int] = []
    num_layers: Optional[int] = None

    # --- process each prompt independently ---
    for i, text in enumerate(prompts):
        # 1) tokenize this prompt
        batch = tokenizer(text, return_tensors="pt")
        input_ids = batch["input_ids"].to(device)          # (1, S)

        # 2) run GPT-2 and request hidden states
        if accepts_prompt_index:
            outputs = gpt2(
                input_ids=input_ids,
                output_hidden_states=True,
                prompt_index=i,
            )
        else:
            outputs = gpt2(
                input_ids=input_ids,
                output_hidden_states=True,
            )

        all_hidden = outputs.hidden_states                 # tuple of length L, each with shape (1, S, C), S=sequence length, C=feature dimension
        if num_layers is None:
            num_layers = len(all_hidden)
            # validate index: 0 <= gpt2_layer_index < num_layers
            if gpt2_layer_index < 0 or gpt2_layer_index >= num_layers:
                raise ValueError(
                    f"gpt2_layer_index={gpt2_layer_index} out of range "
                    f"for model with {num_layers} hidden layers"
                )

        # 3) select the requested layer: shape (1, S, C) -> (S, C)
        layer_h = all_hidden[gpt2_layer_index]
        if layer_h.dim() == 3 and layer_h.size(0) == 1:
            layer_h = layer_h[0]
        layer_h = layer_h.to(device)                       # (S, C)

        per_prompt_states.append(layer_h)
        seq_lengths.append(layer_h.size(0))

    # --- pad to a single [B, max_seq, C] tensor and build masks ---
    B = len(prompts)
    max_seq = max(seq_lengths)
    hidden_dim = per_prompt_states[0].size(1)
    dtype = per_prompt_states[0].dtype

    hidden_states = torch.zeros(B, max_seq, hidden_dim, device=device, dtype=dtype)
    masks = torch.zeros(B, max_seq, device=device, dtype=torch.bool)

    for i, (h, L) in enumerate(zip(per_prompt_states, seq_lengths)):
        hidden_states[i, :L, :] = h
        masks[i, :L] = True      # True = real token, False = padding

    return hidden_states, masks
    # ====== END STUDENT SOLUTION =============================================


def build_text_states(
    labels: torch.Tensor,
    tokenizer,
    gpt2,
    gpt2_layer_index: int,
    device: torch.device,
    cifar10_names: List[str],
    synonyms: Dict[str, List[str]],
    caption_use_prob: float,
    synonym_replace_prob: float,
    idxs: torch.Tensor = None,
    index_to_captions: Dict[int, List[str]] = None,
    captions_per_class: Dict[int, List[str]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Build prompts and convert to padded GPT-2 hidden states.
    - If idxs is provided: use per-index dense captions (requires index_to_captions).
    - Otherwise: use per-class captions (requires captions_per_class).
    """
    if idxs is not None:
        assert index_to_captions is not None, "index_to_captions must be provided when idxs is used"
    else:
        assert captions_per_class is not None, "captions_per_class must be provided when idxs is not used"

    
    prompts: List[str] = []
    if idxs is not None:
        assert index_to_captions is not None, "index_to_captions must be provided when idxs is used"
        for idx_val, lbl in zip(idxs.tolist(), labels.tolist()):
            use_caption = (random.random() < caption_use_prob) and (idx_val in index_to_captions)
            if use_caption:
                text = random.choice(index_to_captions[idx_val])
                text = _maybe_apply_synonym(text, int(lbl), cifar10_names, synonyms, synonym_replace_prob)
            else:
                base = f"a photo of a {cifar10_names[int(lbl)]}"
                text = _maybe_apply_synonym(base, int(lbl), cifar10_names, synonyms, synonym_replace_prob)
            prompts.append(text)
    else:
        assert captions_per_class is not None, "captions_per_class must be provided when idxs is not used"
        for lbl in labels.tolist():
            lbl_int = int(lbl)
            class_pool = captions_per_class.get(lbl_int, [])
            use_caption = (random.random() < caption_use_prob) and (len(class_pool) > 0)
            if use_caption:
                text = random.choice(class_pool)
                text = _maybe_apply_synonym(text, lbl_int, cifar10_names, synonyms, synonym_replace_prob)
            else:
                base = f"a photo of a {cifar10_names[lbl_int]}"
                text = _maybe_apply_synonym(base, lbl_int, cifar10_names, synonyms, synonym_replace_prob)
            prompts.append(text)
    states, masks = prompts_to_padded_hidden_states(prompts, gpt2, tokenizer, gpt2_layer_index, device)
    return states, masks, prompts

# NOTE: You won't have to use this function, but this is how we generated the aggregated cache
def cache_all_text_states(
    base_dataset,
    tokenizer,
    gpt2,
    gpt2_layer_index: int,
    device: torch.device,
    cifar10_names: List[str],
    synonyms: Dict[str, List[str]],
    synonym_replace_prob: float,
    index_to_captions: Dict[int, List[str]] = None,
    captions_per_class: Dict[int, List[str]] = None,
    output_file: str = None,
    batch_size: int = 512,
):
    """
    Precompute and save a single-file aggregated cache mapping dataset index ->
    {"base": Tensor[S,C], "base_mask": Tensor[S], "caption": Tensor[S,C], "caption_mask": Tensor[S]}
    - "base" uses the class-name prompt (caption_use_prob=0.0)
    - "caption" uses dense captions when available (caption_use_prob=1.0)
    """
    if output_file is None:
        raise ValueError("output_file must be provided for cache_all_text_states")

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Try fast path for labels
    labels_list = None
    try:
        if hasattr(base_dataset, "targets") and isinstance(base_dataset.targets, (list, tuple)):
            labels_list = base_dataset.targets
        elif hasattr(base_dataset, "labels") and isinstance(base_dataset.labels, (list, tuple)):
            labels_list = base_dataset.labels
    except Exception:
        labels_list = None
    if labels_list is None:
        labels_list = []
        for i in range(len(base_dataset)):
            _, lbl = base_dataset[i]
            labels_list.append(int(lbl))

    N = len(labels_list)
    agg: Dict[int, Dict[str, torch.Tensor]] = {}

    gpt2.eval()
    with torch.no_grad():
        for start in tqdm(range(0, N, batch_size), desc="Caching text embeddings"):
            end = min(N, start + batch_size)
            idxs = torch.arange(start, end, dtype=torch.long, device=device)
            labels = torch.tensor(labels_list[start:end], dtype=torch.long, device=device)

            # Base states (no dense captions, no synonyms) i.e. "a photo of a airplane"
            base_states, base_masks, _ = build_text_states(
                labels,
                tokenizer,
                gpt2,
                gpt2_layer_index,
                device,
                cifar10_names,
                synonyms,
                0.0,
                0.0,
                idxs=idxs,
                index_to_captions=index_to_captions,
                captions_per_class=captions_per_class,
            )
            # Base states (no dense captions, with synonyms) i.e. "a photo of a small vehicle that drives"
            base_states_syn, base_masks_syn, _ = build_text_states(
                labels,
                tokenizer,
                gpt2,
                gpt2_layer_index,
                device,
                cifar10_names,
                synonyms,
                0.0,
                1.0,
                idxs=idxs,
                index_to_captions=index_to_captions,
                captions_per_class=captions_per_class,
            )
            # Caption states (dense captions, no synonyms) i.e. "a photo of a gray airplane in the blue sky"
            cap_states_plain, cap_masks_plain, _ = build_text_states(
                labels,
                tokenizer,
                gpt2,
                gpt2_layer_index,
                device,
                cifar10_names,
                synonyms,
                1.0,
                0.0,
                idxs=idxs,
                index_to_captions=index_to_captions,
                captions_per_class=captions_per_class,
            )
            # Caption states (dense captions, with synonyms) i.e. "a photo of a gray vehicle that flies in the blue sky"
            cap_states_syn, cap_masks_syn, _ = build_text_states(
                labels,
                tokenizer,
                gpt2,
                gpt2_layer_index,
                device,
                cifar10_names,
                synonyms,
                1.0,
                1.0,
                idxs=idxs,
                index_to_captions=index_to_captions,
                captions_per_class=captions_per_class,
            )

            for j, idx_val in enumerate(range(start, end)):
                agg[int(idx_val)] = {
                    "base": base_states[j].detach().cpu(),
                    "base_mask": base_masks[j].detach().cpu(),
                    "base_syn": base_states_syn[j].detach().cpu(),
                    "base_syn_mask": base_masks_syn[j].detach().cpu(),
                    "caption": cap_states_plain[j].detach().cpu(),
                    "caption_mask": cap_masks_plain[j].detach().cpu(),
                    "caption_syn": cap_states_syn[j].detach().cpu(),
                    "caption_syn_mask": cap_masks_syn[j].detach().cpu(),
                }

    # Atomic save
    tmp_path = output_file + ".tmp"
    torch.save(agg, tmp_path)
    os.replace(tmp_path, output_file)
