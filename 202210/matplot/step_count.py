""""
    Vision Assembly 软件的几何搜索功能太难用，改用Python脚本
    统计金属阶差规的断差

    created by shike 2022-11-01
"""
import os
import csv
from math import sqrt

THRESHOLD = 0.85
base_step = [15.756 , 13.755 ,11.755 ,9.755 ,7.755 ,5.755 ,3.755] # 平面拟合的大致数值
pcd_cloud_dir = "C:/Users/s05559/source/repos/PCL_Learning_Project/PCB点云文件" # 阶差规的点云文件目录

def mean_calc(values):
    assert len(values) != 0
    return sum(values) / len(values)

def strip_calc(values):
    assert len(values) != 0
    values_1 = values[:-1]
    values_2 = values[1:]
    sum_1 = 0
    sum_2 = 0
    for v in values_1:
        sum_1 += v
    for v in values_2:
        sum_2 += v
    return abs(sum_1 - sum_2) / len(values_1)

def normal_distribution_calc(values):
    assert len(values) != 0
    mu_ = mean_calc(values)
    N = len(values)
    sigma_ = 0
    for v in values:
        n = abs((v-mu_)**2)
        sigma_ += n
    sigma_ /= N
    sigma_ = sqrt(sigma_)
    return mu_ ,sigma_

def local_pragam_once_():
    #----------------------------------------#
    #  打开点云文件
    # ----------------------------------------#
    with open(os.path.join(pcd_cloud_dir , "LineScan_range_new.pcd") ,encoding="utf-8") as f:
        text = f.readlines()
    lines = [c.strip() for c in text]

    # ----------------------------------------#
    #  跳过 PCD v.5 的头信息
    # ----------------------------------------#
    header_len = 11
    pointcloud_total_num = len(lines) - header_len
    pointcloud_invalid_num = 0
    pointcloud_outthrehold_num = 0
    array_normal = []
    array_result = []  # 均值
    array_worked = []  # 分布
    
    # ----------------------------------------#
    #  读取点云信息，格式FIELDS x y z
    # ----------------------------------------#
    for i in range(len(base_step)): #各个平面数值
        pointcloud_invalid_num = 0
        pointcloud_outthrehold_num = 0
        pointcloud_threhold = base_step[i]
        pointcloud_values = []

        for i ,line in enumerate(lines): #点云数据
            if header_len > i:
                continue

            # 越界判断
            content = line.split(" ")
            if len(content) != 3:
                continue

            # 无效点云
            if content[2].upper() == "NAN":
                pointcloud_invalid_num += 1
                continue

            # 浮点数转换
            try:
                value = float(content[2])
            except Exception as e:
                print("Exception: float values format methods. \n", e.__class__.__name__, e)
                continue

            # 判断阈值
            if abs(value - pointcloud_threhold) < THRESHOLD:
                pointcloud_values.append(value)
            else:
                pointcloud_outthrehold_num += 1

        mean,sap = normal_distribution_calc(pointcloud_values)
        sap = float(int(sap * 100)/100)
        array_normal.append([mean,sap])
        array_worked.append(len(pointcloud_values))
        array_result.append(mean)
    # ----------------------------------------#
    #  控制台输出统计信息
    # ----------------------------------------#
    print("//////////////////////////////////////////////////")
    print("\t点云总数量\t： {}".format(pointcloud_total_num))
    print("\t无效点云数量\t： {} ".format(pointcloud_invalid_num) + "， 占比{0:.2f}%".format((pointcloud_invalid_num / pointcloud_total_num) * 100))
    print("\t飞点点云数量\t： {} ".format(pointcloud_outthrehold_num) + "， 占比{0:.2f}%".format((pointcloud_outthrehold_num / pointcloud_total_num) * 100))
    print("\t平面数\t\t： {} ".format(len(base_step)))
    print("\t平面数值\t\t： {} ".format(base_step))
    print("\t各平面数值的样本数\t： {} ".format(array_worked))
    print("\t各平面数值的正态分布\t： {} ".format(array_normal))
    print("\t断差的均值\t： {} ".format(strip_calc(array_result)))
    print("//////////////////////////////////////.///////////")
    
    # ----------------------------------------#
    #  输出到结果到本地csv文件
    # ----------------------------------------#
    csvfile = open(os.path.join(pcd_cloud_dir ,"2022-11-02-BDRF" ,"1-py-断差-ROI.csv") , 'a',
        encoding='utf-8', newline='')
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(array_result)

    # ----------------------------------------#
    #  结束
    # ----------------------------------------#
    csvfile.close()
    f.close()

    return None

if __name__ == "__main__":
    print("Count metal steps...")
    local_pragam_once_()
    print("Hello world!")