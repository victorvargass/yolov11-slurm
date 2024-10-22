#!/bin/bash

# IMPORTANT PARAMS
#SBATCH -p gpu                       # Particion GPU
#SBATCH --gres=gpu:A100:2           # Una GPU por favor

# OTHER PARAMS
#SBATCH -J fusa-roads-yolo       # Nombre de la tarea

# COMMANDS ON THE COMPUTE NODE
pwd                         #
date                        #

# Parámetros generales
CONTAINER_WORKDIR=${PWD}
CONTAINER_NAME="ultralytics"
PYTHON_CMD="python -m torch.distributed.run --nproc_per_node 2 --master_port 12345 train.py"
WEIGHTS="yolo11n.pt"
DATA="data.yaml"
EPOCHS=2000
IMGSZ=640
BATCH=256
OPTIMIZER="AdamW" #SGD
NAME="FusaRoads"
PATIENCE=150
WORKERS=16

# Hiperparámetros
LR0=0.001 #default 0.01
LRF=0.01
MOMENTUM=0.937
WEIGHT_DECAY=0.0005
WARMUP_EPOCHS=10.0  #default 3.0
WARMUP_MOMENTUM=0.8
WARMUP_BIAS_LR=0.1
BOX=0.05 #default 7.5
CLS=0.5
HSV_H=0.015
HSV_S=0.7
HSV_V=0.4
DEGREES=0
TRANSLATE=0.1
SCALE=0.5
SHEAR=0
PERSPECTIVE=0.0
FLIPUD=0
FLIPLR=0.5
MOSAIC=1.0
MIXUP=0
COPY_PASTE=0

# Ejecutar el comando con las variables
srun --container-workdir=${CONTAINER_WORKDIR} \
     --container-name=${CONTAINER_NAME} \
     ${PYTHON_CMD} \
     --imgsz ${IMGSZ} \
     --data ${DATA} \
     --weights ${WEIGHTS} \
     --name ${NAME} \
     --batch ${BATCH} \
     --epochs ${EPOCHS} \
     --patience ${PATIENCE} \
     --optimizer ${OPTIMIZER} \
     --workers ${WORKERS} \
     --lr0 ${LR0} \
     --lrf ${LRF} \
     --momentum ${MOMENTUM} \
     --weight_decay ${WEIGHT_DECAY} \
     --warmup_epochs ${WARMUP_EPOCHS} \
     --warmup_momentum ${WARMUP_MOMENTUM} \
     --warmup_bias_lr ${WARMUP_BIAS_LR} \
     --box ${BOX} \
     --cls ${CLS} \
     --hsv_h ${HSV_H} \
     --hsv_s ${HSV_S} \
     --hsv_v ${HSV_V} \
     --degrees ${DEGREES} \
     --translate ${TRANSLATE} \
     --scale ${SCALE} \
     --shear ${SHEAR} \
     --perspective ${PERSPECTIVE} \
     --flipud ${FLIPUD} \
     --fliplr ${FLIPLR} \
     --mosaic ${MOSAIC} \
     --mixup ${MIXUP} \
     --copy_paste ${COPY_PASTE}