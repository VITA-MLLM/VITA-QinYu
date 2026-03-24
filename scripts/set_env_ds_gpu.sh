#set -e
#set -x

######################################################################
#export NCCL_NET=IB

#export NCCL_SOCKET_IFNAME="bond1"
#export GLOO_SOCKET_IFNAME="bond1"
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export GLOO_SOCKET_IFNAME=eth0
#export NCCL_DEBUG=INFO
#export NCCL_IB_QPS_PER_CONNECTION=2

#export NCCL_IB_DISABLE=1

export DISTRIBUTED_BACKEND="nccl"
export CUDA_DEVICE_MAX_CONNECTIONS=1

#export OPENBLAS_NUM_THREADS=1
#export TOKENIZERS_PARALLELISM=false

######################################################################
if [ -e /etc/pip/constraint.txt ]
then
    echo "" > /etc/pip/constraint.txt
else
    echo ""
fi

# pip uninstall -y nvidia-modelopt
#pip3 install --no-index --find-links=/data/software/ -r requirements_ds_gpu.txt
# pip3 install -e `pwd`

######################################################################
#export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

#apt update
#apt install -y openssh-server rsync tmux htop
#apt install -y ffmpeg

######################################################################

export NNODES=${WORLD_SIZE}
export NODE_RANK=${RANK}
export MASTER_PORT=45678

if [ -z "$NPROC_PER_NODE" ]
then
    export NPROC_PER_NODE=8
    export NNODES=1
    export NODE_RANK=0
    export MASTER_ADDR=127.0.0.1
fi

######################################################################
