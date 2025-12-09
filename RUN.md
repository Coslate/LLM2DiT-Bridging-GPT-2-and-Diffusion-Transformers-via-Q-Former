
# CUDA_VISIBLE_DEVICES=1 python train_qformer.py \
   # --pretrained_model_path ./data/ddpm_dit_cifar_100_epochs.pth \
   # --dense_captions_path "./data/cifar10_dense_captions.jsonl" \
   # --epochs 50 \
   # --batch_size 128 --lr 1e-4 \
   # --save_model_path /data/patrick/10623-hw4/models/trained_qformer.pth \
   # --save_dir /data/patrick/10623-hw4/contents_qformer_dense_training \
   # --gpt2_layer_index 12 --num_query_tokens 4 --device 0 --cfg 3.0 --data_dir ./data \
   # --gpt2_cache_dir ./data --optimizer_ckpt_interval 5 \
   # --cache_text_embeddings ./data/text_embeddings.pt

CUDA_VISIBLE_DEVICES=6 python -m src.train_qformer \
  --pretrained_model_path ./experiments/base_dit/ddpm_dit_cifar_100_epochs.pth \
  --dense_captions_path ./data/cifar10_dense_captions.jsonl \
  --data_dir ./data \
  --epochs 50 \
  --batch_size 128 \
  --lr 1e-4 \
  --n_T 1000 \
  --device 0 \
  --save_dir /data/patrick/10623GenAI/llm2dit/experiments/qformer_dense \
  --save_model_path /data/patrick/10623GenAI/llm2dit/experiments/qformer_dense/checkpoints/ddpm_dit_cifar_qformer_dense.pth \
  --num_query_tokens 4 \
  --transformer_hidden_size 768 \
  --cfg 3.0 \
  --gpt2_layer_index 12 \
  --gpt2_cache_dir ./data \
  --cache_text_embeddings ./data/text_embeddings.pt   


CUDA_VISIBLE_DEVICES=2 python eval_qformer.py --config_yaml inference.yaml
CUDA_VISIBLE_DEVICES=2 python eval_qformer.py --config_yaml inference_65.yaml
CUDA_VISIBLE_DEVICES=2 python eval_qformer.py --config_yaml inference_65b.yaml
CUDA_VISIBLE_DEVICES=2 python eval_qformer.py --config_yaml inference_66a.yaml