import csv
import os

# 读取csv
def load_csv(data_path):
    data = []

    with open(data_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            for i in range(len(row)):
                if row[i] == 'True':
                    row[i] = True
                elif row[i] == 'False':
                    row[i] = False
                # else:
                # row[i] = int(row[i])
            data.append(row)
    return data

# 保存csv
def save_csv(data_path, data):
    if os.path.exists(data_path):
        os.remove(data_path)

    with open(data_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(data)):
            if data[i]:
                writer.writerow(data[i])
    print("save data successfully to: {}".format(data_path))



# 记录到日志文件并打印出来
def log(logfile, str):
    print(str)
    # 训练时间
    with open(logfile, 'a') as f:
        f.write(str + "\n")

