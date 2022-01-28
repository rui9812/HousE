#!/bin/sh

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models
LOG_PATH=logs

#The first four parameters must be provided
#MODE=$1
MODEL=$1
DATASET=$2
GPU_DEVICE=$3
SAVE_ID=$4
LOG_ID=$5

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"
LOG=$LOG_PATH/"$MODEL"_"$DATASET"_"$LOG_ID"

#Only used in training
BATCH_SIZE=$6
NEGATIVE_SAMPLE_SIZE=$7
HIDDEN_DIM=$8
ROT_DIM=$9
PRO_NUM=${10}
THRED=${11}
GAMMA=${12}
ALPHA=${13}
LEARNING_RATE=${14}
MAX_STEPS=${15}
WARM_STEPS=${16}
TEST_BATCH_SIZE=${17}
REG=${18}

echo "Start Training......"

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $HIDDEN_DIM \
    -hd $ROT_DIM -dn $PRO_NUM -th $THRED \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS --warm_up_steps $WARM_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    -r $REG > ./$LOG.log 2>&1 &