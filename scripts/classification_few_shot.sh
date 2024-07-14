#!/bin/bash
gpu_id=$1
weights_path=$2
way=$3
shot=$4

# Check the both arguments are provided
if [ -z "$gpu_id" ] || [ -z "$weights_path" ] || [ -z "$way" ] || [ -z "$shot" ]
then
    echo "Please provide the gpu id, the model weights path, way and shot"
    echo "Example usage: bash scripts/classification_few_shot.sh 0 pretrained0/model.pth <way> <shot>"
    exit 1
fi

for i in `seq 0 9`
do
    python -m pointjepa.tasks.classification fit \
    --config configs/classification/modelnet_fewshot.yaml \
    --config configs/wandb/pointjepa/classification_modelnet_fewshot.yaml  \
    --model.pretrained_ckpt_path $weights_path    \
    --data.way $way --data.shot $shot --data.fold $i
done
