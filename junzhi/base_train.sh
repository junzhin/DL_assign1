#!/bin/bash

# Command line arguments for train.py
study_subject="lr" the focus of this experiment
layer_neurons=( 16 32 64 )
activation_funcs=( relu relu sigmoid )
learning_rate=0.001
epochs=100
dropout_prob=0.5
batch_size=32
weight_decay=0.001
beta=( 0.9 0.99 )
size=50000
batch_norm=True
loss=CE
optimizer=adam
# 修改成对应要探索的维度的名字
save_path='./results/$study_subject($learning_rate)/'

# Run train.py with the above arguments
python train.py --layer_neurons "${layer_neurons[@]}" --activation_funcs "${activation_funcs[@]}" \
--learning_rate $learning_rate --epochs $epochs --dropout_prob $dropout_prob --batch_size $batch_size \
--weight_decay $weight_decay --beta "${beta[@]}" --size $size --batch_norm $batch_norm --loss $loss \
--optimizer $optimizer --save_path "$save_path"
