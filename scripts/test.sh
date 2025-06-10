#!bin/bash
export CUDA_VISIBLE_DEVICES=$4
TASK=$1
MODEL=$2
DATASET=$3

python open_biomed/scripts/train.py \
--task $TASK \
--additional_config_file configs/model/$MODEL.yaml \
--dataset_name $DATASET \
--dataset_path ./datasets/$TASK/$DATASET \
--ckpt_path ./checkpoints/server/pharmolix_fm.ckpt \
--batch_size_eval 32 \
--test_only