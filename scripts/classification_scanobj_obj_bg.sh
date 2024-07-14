#!/bin/bash
gpu_id=$1
weights_path=$2

# Check the both arguments are provided
if [ -z "$gpu_id" ] || [ -z "$weights_path" ]
then
    echo "Please provide the gpu id and the model weights path"
    echo "Example usage: bash scripts/classification_scanobj_obj_bg.sh 0 modelnet40_0/model.pth"
    exit 1
fi

for i in `seq 1 10`
do
    python -m pointjepa.tasks.classification fit \
    --config configs/Point-JEPA/classification/scanobjectnn.yaml \
    --config configs/Point-JEPA/wandb/classification_scanobjectnn_obj_bg.yaml  \
    --model.pretrained_ckpt_path $weights_path    \
    --trainer.devices="[$gpu_id]" --seed_everything $i --config configs/Point-JEPA/classification/_scanobjectnn_obj_bg.yaml
done
