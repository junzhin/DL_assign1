import os
import csv
import re
import chardet

# Variables
#lr
# subject = "lr"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["0.5", "0.05", "0.005", "0.0005"]
# root_folder = "./results"

#batch size
# subject = "mini_batch"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["1","256","512","1024"]
# root_folder = "./results"

# hiddenLayers
# subject = "hiddenLayers"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["12
# 8_256_150_100_10", "128_150_100_10", "128_150_10"]
# root_folder = "./results"

 
# # ActivationFunctions
# subject = "ActivationFunctions"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["None_tanh_tanh_softmax",
#                "None_relu_relu_softmax", "None_leakyrelu_leakyrelu_softmax"]
# root_folder = "./results"


# optimiser
# subject = "optimizer"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["sgd_momentum", "rmsprop", "adam"]
# root_folder = "./results"

# Dropout
# subject = "dropout_prob"
# times = ["1.0", "2.0", "3.0"]
# subelements = ["0.5","0.7","0.9","1"]
# root_folder = "./results"

# Weight decay
subject = "batch_norm"
times = ["1.0", "2.0", "3.0"]
subelements = ["False", "True"]
root_folder = "./results"


# Regular expression patterns
pattern_epoch_149 = r"Epoch: 149 \|"
pattern_train_loss = r"train_loss_per_epochs : (\d+\.\d+)"
pattern_train_acc = r"train_acc_per_epochs : (\d+\.\d+)"
pattern_val_loss = r"val_loss_per_epochs : (\d+\.\d+)"
pattern_val_acc = r"val_acc_per_epochs : (\d+\.\d+)"
pattern_train_f1 = r"train_f1_per_epochs : (\d+\.\d+)"
pattern_val_f1 = r"val_f1_per_epochs : (\d+\.\d+)"


# CSV file setup
csv_header = ["subject", "times", "subelements", "train_loss_per_epochs",
              "train_acc_per_epochs", "val_loss_per_epochs", "val_acc_per_epochs", "train_f1_per_epochs", "val_f1_per_epochs"]

print("starting .... ")
with open(f'./{subject}_output.csv', "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(csv_header)

    for time in times:
        for subelement in subelements:
            folder_name = f"{subject}{time}"
            subfolder_name = f"{subject}_{subelement}"
            log_file_path = os.path.join(
                root_folder, folder_name, subfolder_name, "log.txt")
            print('log_file_path: ', log_file_path)


            if os.path.exists(log_file_path):
                with open(log_file_path, "rb") as log_file:
                    raw_data = log_file.read()
                    encoding = chardet.detect(raw_data)["encoding"]

                with open(log_file_path, "r", encoding=encoding) as log_file:
                    log_content = log_file.read()

                    epoch_149_line = None

                    for line in log_content.splitlines():
                        if re.search(pattern_epoch_149, line):
                            epoch_149_line = line
                            break

                    if epoch_149_line:
                        val_loss = re.search(
                            pattern_val_loss, epoch_149_line).group(1)
                        train_acc = re.search(
                            pattern_train_acc, epoch_149_line).group(1)
                        train_loss = re.search(
                            pattern_train_loss, epoch_149_line).group(1)
                        val_acc = re.search(
                            pattern_val_acc, epoch_149_line).group(1)
                        train_f1 = re.search(
                            pattern_train_f1, epoch_149_line).group(1)
                        val_f1 = re.search(
                            pattern_val_f1, epoch_149_line).group(1)
                        val_f1 = re.search(
                            pattern_val_f1, epoch_149_line).group(1)
                      

                        csv_writer.writerow(
                            [subject, time, subelement, train_loss, train_acc, val_loss, val_acc, train_f1, val_f1])
