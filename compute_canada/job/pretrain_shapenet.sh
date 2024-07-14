#!/bin/bash
#SBATCH --job-name=pretrain   
#SBATCH --nodes=1                                     
#SBATCH --mem=16G                       
#SBATCH --time=10:00:00                 
#SBATCH --gres=gpu:a100:1
#SBATCH --tasks-per-node=1    # Request 1 process per GPU.
#SBATCH --cpus-per-task=8 
#SBATCH --output=compute_canada/logs/pretrain_%j.out  # Standard output and error log

# Load your Python module or any other modules
module load python/3.10 cuda/11.4

# If you're using a virtual environment, activate it
source ~/cc_ssl/bin/activate

export NCCL_BLOCKING_WAIT=1

export WANDB_MODE=offline
export WANDB_API_KEY="$(cat ~/.wandb_cred)"
srun python -m pointjepa fit -c configs/pretraining/shapenet.yaml  -c  configs/wandb/pointjepa/pretraining_shapenet.yaml --seed_everything 1 \
    --model.target_sample_method contiguous \
    --model.num_targets_per_sample 4 \
    --model.target_sample_ratio "[0.15, 0.2]" \
    --model.context_sample_method contiguous \
    --model.context_sample_ratio "[0.4, 0.75]" \
    --model.ema_tau_min 0.9995
    # --model.predictor_depth 12
    # --model.tokenizer_encoder_name transformer \
    # --model.tokenizer_encoder_size_name spct \
    # --model.predictor_depth 4 \
    # --trainer.max_epochs 500 \
    # --model.ema_tau_epochs 500 
    # --trainer.log_every_n_steps 10 \
    # --trainer.precision 16 \
    # --model.tokenizer_num_groups 64 \
    # --model.tokenizer_group_size 32 \
    # --model.tokenizer_encoder_dim 32 \
    # --model.tokenizer_encoder_num_heads 4 \
    # --model.tokenizer_encoder_mlp_ratio 4.0 \
    # --model.tokenizer_encoder_dropout 0.0 \
    # --model.tokenizer_encoder_drop_path_rate 0.25 \
    # --model.tokenizer_encoder_attention_dropout 0.05 \
    # --model.encoder_dim 384 \
    # --model.encoder_depth 12 \
    # --model.encoder_heads 6 \
    # --model.encoder_dropout 0.1 \
    # --model.encoder_attention_dropout 0.05 \
    # --model.encoder_drop_path_rate 0.25 \
    # --model.encoder_add_pos_at_every_layer true \
    # --model.predictor_embed_dim 192 \
    # --model.predictor_depth 6 \
    # --model.predictor_heads 6 \
    # --model.predictor_mlp_ratio 4.0 \
    # --model.predictor_dropout 0.1 \
    # --model.predictor_add_pos_at_every_layer true \
    # --model.predictor_attention_dropout 0.05 \
    # --model.predictor_drop_path_rate 0.25 \
    # --model.predictor_add_context_pos false \
    # --model.token_seq_method iterative_nearest_min_start \
    # --model.target_sample_method contiguous \
    # --model.num_targets_per_sample 4 \
    # --model.target_sample_ratio "[0.15, 0.2]" \
    # --model.context_sample_method contiguous \
    # --model.context_sample_ratio "[0.85, 1.0]" \
    # --model.target_layers "[11]" \
    # --model.target_layer_part final \
    # --model.target_layer_norm layer \
    # --model.target_norm null \
    # --model.ema_tau_min 0.9991 \
    # --model.ema_tau_max 1.0 \
    # --model.ema_tau_epochs 650 \
    # --model.loss smooth_l1 \
    # --model.learning_rate 1e-3 \
    # --model.optimizer_adamw_weight_decay 0.05 \
    # --model.lr_scheduler_linear_warmup_epochs 45 \
    # --model.lr_scheduler_linear_warmup_start_lr 1e-5 \
    # --model.lr_scheduler_cosine_eta_min 1e-6 

