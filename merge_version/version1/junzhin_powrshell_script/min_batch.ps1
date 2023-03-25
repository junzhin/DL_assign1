# This is a PowerShell script, so you need to execute it in PowerShell
# Command line arguments for train.py
$vers  = @('1.0','2.0', '3.0')
$study_subject = "mini_batch" # the focus of this experiment 需要更改的参数

$activation_func = @('None','relu', 'relu', 'softmax')
$layer_neuron =  @(128, 150, 100, 10) 
# $learning_rate = 0.005
$learning_rate = 0.0005 # for mini_batch of size =  1
# $learning_rates = @(0.5,0.05,0.005) # 需要更改的参数
$epochs = 150
$dropout_prob = 1 # probability that perserve the neuron
# $batch_sizes =  @(1,256, 512,1024)
$batch_sizes =  @(1)
$weight_decay = 0
$beta = @(0.9, 0.99)
$size = 50000
$batch_norm = "False"
$loss = "CE"
$optimizer = "sgd_momentum"

$run = 0
# 修改成对应要探索的维度的名字
foreach ($ver in $vers) {
 
    foreach ($i in 0..($batch_sizes.Length-1)) {
        $batch_size = $batch_sizes[$i]
        $sub_specifer = $batch_size
        $run++  
        $save_path = "./results/${study_subject}${ver}/${study_subject}_${sub_specifer}/"
        
        New-Item -ItemType Directory -Path $save_path
        # Run train.py with the above arguments
        Write-Output "$run run is starting!"
        Write-Output "$save_path"
        python ../train.py --layer_neurons $layer_neuron --activation_funcs $activation_func   --learning_rate $learning_rate --epochs $epochs --dropout_prob $dropout_prob --batch_size $batch_size --weight_decay $weight_decay --beta $beta --size $size --batch_norm $batch_norm --file_location "../../../raw_data/" --loss $loss --optimizer $optimizer --save_path $save_path > "${save_path}log.txt"
        Write-Output "$run run is finished!"
    }
}