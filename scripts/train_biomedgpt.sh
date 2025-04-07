#!bin/bash
export CUDA_VISIBLE_DEVICES=7
TASK="multimodal_question_answering"
MODEL="biomedgpt"
DATASET="biomedical_qa"

python open_biomed/scripts/train.py \
--task $TASK \
--additional_config_file configs/model/$MODEL.yaml \
--dataset_name $DATASET \
--dataset_path ./datasets/$TASK/$DATASET \
--batch_size_train 1 \
--batch_size_eval 1 \
--max_epochs 1 \
--ckpt_freq 1 \
--empty_folder 