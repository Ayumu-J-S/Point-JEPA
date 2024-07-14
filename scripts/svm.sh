#!/bin/bash
# This script will result in error. Need to debug a bit.
gpu_id=$1
weights_path=$2


# Check the both arguments are provided
if [ -z "$gpu_id" ] || [ -z "$weights_path" ]
then
    echo "Please provide the gpu id and the model weights path"
    echo "Example usage: bash scripts/classification_modelnet.sh 0 modelnet40_0/model.pth"
    exit 1
fi

#  python -m pointjepa validate -c configs/pretraining/shapenet.yaml --trainer.devices="[5]" -c configs/wandb/pointjepa/pretraining_shapenet.yaml --seed_everything 2 --model.pretrained_ckpt_path best/epoch=499-step=40500.ckpt

for i in `seq 1 10`
do
    python -m pointjepa validate -c configs/pretraining/shapenet.yaml --trainer.devices="[$gpu_id]" -c configs/wandb/pointjepa/pretraining_shapenet.yaml  --seed_everything $i --model.pretrained_ckpt_path $weights_path
done
