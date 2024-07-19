module load cuda

PARTITION=agchadbourne
JOB_NAME=xplane_train
NODES=1
GPUS=2
GPUS_PER_NODE=2
CPUS_PER_NODE=5

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    python /home/agchadbourne/XPlaneAutolandScenario/src/xplane_autoland/vision/train.py --data-dir="home/agchadbourne/model"