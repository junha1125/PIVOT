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

PROMPT_VERSION=plain

# Source MLLM with PIVOT-enhanced vision encoder (from Stage 2)
MID_RUN_NAME="stage2-dpo-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-llava"
VISION_SOURCE_MLLM="./checkpoints/stage2/${MID_RUN_NAME}"

BASE_RUN_NAME="pivot-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "VISION_SOURCE_MLLM: ${VISION_SOURCE_MLLM}"

# or --attn_implementation flash_attention_2 \
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}"\
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --fp16 True \
    --attn_implementation sdpa \
    --data_path ./datasets/BLIP558K/blip_laion_cc_sbu_558k.json \
    --image_folder ./datasets/BLIP558K/images \
    --vision_tower ${VISION_MODEL_VERSION} \
    --vision_source_mllm_path ${VISION_SOURCE_MLLM} \
    --mm_projector_source_load_layers 0 \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --output_dir ./checkpoints/pivot-proj-train/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --save_strategy "no" \
    --save_steps 500 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME
