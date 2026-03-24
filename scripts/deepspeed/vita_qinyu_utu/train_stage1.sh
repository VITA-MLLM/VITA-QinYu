#!/bin/bash
set -e
set -x

SEQ_LENGTH=10000
DATA_PATH=$1
DATASET_PATH=$2
MODEL_PATH=$3

ln -sf $DATASET_PATH /vita-qinyu_dataset
ln -sf $MODEL_PATH /vita-qinyu_models


timestamp=`basename $DATA_PATH .yaml`

######################################################################
export NCCL_NVLS_ENABLE=0

export ROOT_PATH=exp
export AUDIO_CACHE_DIR=$ROOT_PATH/.AUDIO_CACHE
export CODE_PATH=.
export LOCAL_CODE_PATH=/vita-qinyu

# apt update
# apt install -y rsync
mkdir -p ${LOCAL_CODE_PATH}
rsync -avP --exclude ".git" --exclude ".gitee" ${CODE_PATH}/ ${LOCAL_CODE_PATH}/
cd ${LOCAL_CODE_PATH}

######################################################################
# SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# apt-get remove python3-blinker -y
source scripts/set_env_ds_gpu.sh
# pip install --no-index --find-links=/data/software/ liger-kernel
# pip install liger-kernel -i https://mirrors.cloud.tencent.com/pypi/simple

######################################################################
OUTPUT_DIR=${ROOT_PATH}/output/LM/"$0"/${timestamp}/
mkdir -p ${OUTPUT_DIR}
cp $0 ${OUTPUT_DIR}

export HF_HOME="$ROOT/.CACHE/HF_HOME_node${INDEX}/"
mkdir -p ${HF_HOME}

export TRITON_CACHE_DIR=${LOCAL_CODE_PATH}
export PYTHONPATH=$PYTHONPATH:${LOCAL_CODE_PATH}/third_party/MOSS-TTSD-v0.7/XY_Tokenizer:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice:${LOCAL_CODE_PATH}/third_party/GLM-4-Voice/third_party/Matcha-TTS/:${LOCAL_CODE_PATH}/third_party/tme
echo $PYTHONPATH

######################################################################
LOG=${OUTPUT_DIR}/log_node${INDEX}.txt
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
echo ${@}

######################################################################
MODEL_NAME_OR_PATH=/vita-qinyu-models/VITA-QinYu-4B
CONFIG_NAME=${LOCAL_CODE_PATH}/vita_qinyu/models/vita_utu_mtp_sensevoice_xy_v4_51_3/config.json
AUDIO_MODEL_NAME_OR_PATH=/vita-qinyu-models/FunAudioLLM/SenseVoiceSmall
AUDIO_TOKENIZER_PATH="/vita-qinyu-models/VITA-QinYu-Models/xy_tokenizer.ckpt /vita-qinyu-models/VITA-QinYu-Models/campplus_cn_common.bin $AUDIO_MODEL_NAME_OR_PATH"
SPK2EMB=/vita-qinyu-models/VITA-QinYu-Models/spk2embeds_roleplay.pt


######################################################################
DISTRIBUTED_ARGS="
    --nproc_per_node $NPROC_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

SIMULATE_TOTAL_GPU=128
GRAD_ACCUM=$((SIMULATE_TOTAL_GPU/NNODES/NPROC_PER_NODE))

torchrun $DISTRIBUTED_ARGS tools/finetune.py \
    --log_level "info" \
    --do_train \
    --overwrite_output_dir \
    --config_name $CONFIG_NAME \
    --tokenizer_name $MODEL_NAME_OR_PATH \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --audio_tokenizer_path $AUDIO_TOKENIZER_PATH \
    --audio_tokenizer_type "sensevoice_xytokenizer_speaker" \
    --dataset_name $DATA_PATH \
    --bf16 True \
    --torch_dtype bfloat16 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --max_steps 8000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 48 \
    --learning_rate 9.00e-5 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --report_to "tensorboard" \
    --model_max_length ${SEQ_LENGTH} \
    --gradient_checkpointing True \
    --deepspeed ${LOCAL_CODE_PATH}/scripts/deepspeed/ds_config_zero2.json \
    --trust_remote_code False \
    --ddp_timeout 7200 \
    --ddp_backend ${DISTRIBUTED_BACKEND} \
    --attn_implementation flash_attention_2 \
    --seed 44 \
    --data_seed 44 \
    --reset_attention_mask \
    --reset_position_ids \
    --create_attention_mask false \
    --create_attention_mask_2d false \
    --dataloader_num_workers 5 \
    --dataloader_prefetch_factor 25 \
    --audio-model-freeze \
    --text-audio-interval-ratio 1 8 4 8 \
    --only_last_conv_label True \
    --use_audio_special_token False \
    --dataset_class qwen3_jsonl_xyspk \
    --extra_sensevoice_token False \
    --audio_model_name_or_path $AUDIO_MODEL_NAME_OR_PATH \
    --speaker_embedding_prob 1 \
    --spk2emb $SPK2EMB \

set +x
