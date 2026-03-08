export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

export NUM_GPUS=8
LLM_VERSION="Qwen/Qwen2.5-1.5B-Instruct"
LLM_VERSION_CLEAN=$(echo "${LLM_VERSION#*/}" | cut -d- -f1-2); 
VISION_MODEL_VERSION="google/siglip2-so400m-patch16-384"
VISION_MODEL_VERSION_CLEAN=$(echo "${VISION_MODEL_VERSION#*/}" | cut -d- -f1-2)

PROMPT_VERSION="qwen_1_5"

MID_RUN_NAME="stage1-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-llava"
FIN_RUN_NAME="stage2-sft-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-llava"

# or --attn_implementation flash_attention_2 \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --master_port 12346 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --fp16 True \
    --attn_implementation sdpa \
    --data_path ./datasets/data-processor/MMPR-v1.2/sft_mmpr_20000_777.json \
    --image_folder ./datasets \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --model_name_or_path "./checkpoints/stage1-full-train/${MID_RUN_NAME}" \
    --version ${PROMPT_VERSION} \
    --group_by_modality_length True \
    --run_name $FIN_RUN_NAME \
    --output_dir "./checkpoints/stage2/${FIN_RUN_NAME}" \
    --num_train_epochs 3 \
    --per_device_eval_batch_size 4 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 5e-06 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --dataloader_drop_last True


    # --mm_vision_tower_lr=2e-6 \
# You can delete the sdpa attn_implementation if you want to use flash attn
