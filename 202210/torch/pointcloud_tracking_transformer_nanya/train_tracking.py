import argparse
from Dataset import SiameseTrain

parser = argparse.ArgumentParser()
parser.add_argument("--batchsize" ,type=int ,default= 64 ,help="input batch size")
parser.add_argument("--workers" ,type=int ,default=4 ,help="number of data loading workers")
parser.add_argument("--nepoch" ,type=int ,default=160 ,help = "numbers of epoch to train for")
parser.add_argument("--learning_rate" ,type=float ,default=0.001 ,help="learning rate at t=0")
parser.add_argument("--data_dir" ,type=str ,default="./data/kitti" ,help="dataset path")
parser.add_argument("--category_name" ,type=str ,default="Car" ,help="Object to Track(Car/Pedestrian/Van/Cyclist)")
parser.add_argument("--save_root_dir" ,type=str ,default="results" ,help="output folder")
parser.add_argument("--model" ,type=str ,default="" ,help="model name for training resume")
parser.add_argument("--optimizer" ,type=str ,default="" ,help="model name for training resume")
parser.add_argument("--tiny" ,type=bool ,default=False)
parser.add_argument("--input_size" ,type=int ,default=1024)
parser.add_argument("--save_interval" ,type=int ,default=1)
opt = parser.parse_args()

if opt.category_name == 'Ped':
    opt.category_name = 'Pedestrian'
if opt.category_name == 'Cyc':
    opt.category_name = 'Cyclist'
print(opt)

#==================================================================#
#
#   加载数据集
#
#==================================================================#
train_data = SimaeseTrain()