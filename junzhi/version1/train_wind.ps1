# This is a PowerShell script, so you need to execute it in PowerShell
# Command line arguments for train.py
$study_subject = "lr" # the focus of this experiment 需要更改的参数
$layer_neurons = @(128, 100, 110, 100, 10)
$activation_funcs = @('None','leakyrelu', 'leakyrelu', 'leakyrelu', 'softmax')
# $learning_rate = 0.001
$learning_rates = @(0.1, 0.01, 0.001) # 需要更改的参数
$epochs = 50
$dropout_prob = 1
$batch_size = 512
$weight_decay = 0.001
$beta = @(0.9, 0.99)
$size = 5000
$batch_norm = $true
$loss = "CE"
$optimizer = "adam"

$run = 0
# 修改成对应要探索的维度的名字
foreach ($learning_rate in $learning_rates) {
   
    $sub_specifer = $learning_rate  # 需要更改的参数
    $run++  
    $save_path = "./results/${study_subject}/${study_subject}_${sub_specifer}/"
    
    New-Item -ItemType Directory -Path $save_path
    # Run train.py with the above arguments
    Write-Output "$run run is starting!"
    Write-Output "$save_path"
    python train.py --layer_neurons $layer_neurons --activation_funcs $activation_funcs `
    --learning_rate $learning_rate --epochs $epochs --dropout_prob $dropout_prob --batch_size $batch_size `
    --weight_decay $weight_decay --beta $beta --size $size --batch_norm $batch_norm --loss $loss `
    --optimizer $optimizer --save_path $save_path > "${save_path}log.txt"
    Write-Output "$run run is finished!"
}