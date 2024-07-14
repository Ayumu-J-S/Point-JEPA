#!/bin/bash
weights_path=$1

# Check the both arguments are provided
if [ -z "$weights_path" ]
then
    echo "Please provide the model weights path, way and shot"
    echo "Example usage: bash scripts/classification_modelnet_vote.sh fine_tuned/model.pth"
    exit 1
fi

for i in `seq 1 10`
do
    python -m pointjepa.eval.voting \
    --config "configs/Point-JEPA/classification/modelnet40.yaml" \
    --config "configs/Point-JEPA/wandb/voting_modelnet40.yaml" \
    --finetuned_ckpt_path $weights_path \
    --seed_everything $i 
done
