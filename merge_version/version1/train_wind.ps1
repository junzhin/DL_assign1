# This is a PowerShell script, so you need to execute it in PowerShell
# Command line arguments for train.py
$vers  = @('1.0','2.0', '3.0')
$study_subject = "lr" # the focus of this experiment 需要更改的参数
$layer_neurons = @(128, 150, 100, 10)
$activation_funcs = @('None','relu', 'relu', 'softmax')
$learning_rates = @(0.5,0.05,0.005) # 需要更改的参数
$epochs = 150
$dropout_prob = 1
$batch_size = 512
$weight_decay = 0
$beta = @(0.9, 0.99)
$size = 50000
$batch_norm = "False"
$loss = "CE"
$optimizer = "sgd_momentum"

$run = 0
# 修改成对应要探索的维度的名字
foreach ($ver in $vers) {
 
    foreach ($learning_rate in $learning_rates) {
        $sub_specifer = $learning_rate  # 需要更改的参数
        $run++  
        $save_path = "./results/${study_subject}${ver}/${study_subject}_${sub_specifer}/"
        
        New-Item -ItemType Directory -Path $save_path
        # Run train.py with the above arguments
        Write-Output "$run run is starting!"
        Write-Output "$save_path"
        python ../train.py --layer_neurons $layer_neurons --activation_funcs $activation_funcs `
        --learning_rate $learning_rate --epochs $epochs --dropout_prob $dropout_prob --batch_size $batch_size `
        --weight_decay $weight_decay --beta $beta --size $size --batch_norm $batch_norm --file_location "../../../raw_data/" --loss $loss --optimizer $optimizer --save_path $save_path > "${save_path}log.txt"
        Write-Output "$run run is finished!"
    }
}