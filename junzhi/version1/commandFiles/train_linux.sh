#!/bin/bash

# Command line arguments for train.py
study_subject="lr" the focus of this experiment

layer_neurons=( 128, 100, 110, 100, 10)
activation_funcs=('None' 'leakyrelu' 'leakyrelu' 'leakyrelu' 'softmax')
# learning_rate=0.001
learning_rates= (0.1 0.01 0.001)
epochs=50
dropout_prob=1
batch_size=512
weight_decay=0.001
beta=( 0.9 0.99 )
size=50000
batch_norm=True
loss=CE
optimizer=adam
# 修改成对应要探索的维度的名字
save_path='./results/$study_subject($learning_rate)/'
for learning_rate in "${learning_ratess[@]}"
do 
# Run train.py with the above arguments
python train.py --layer_neurons "${layer_neurons[@]}" --activation_funcs "${activation_funcs[@]}" \
--learning_rate $learning_rate --epochs $epochs --dropout_prob $dropout_prob --batch_size $batch_size \
--weight_decay $weight_decay --beta "${beta[@]}" --size $size --batch_norm $batch_norm --loss $loss \
--optimizer $optimizer --save_path "$save_path" > "$save_path/log.txt"
done