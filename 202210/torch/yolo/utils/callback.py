import os
import torch
from matplotlib import pyplot as plt
import scipy.signal
from torch.utils.tensorboard import SummaryWriter

class LossHistory():
    def __init__(self ,log_dir ,model ,input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        print("log histoory dir save weightsL " ,self.log_dir)
        os.makedirs(self.log_dir)
        self.writer = SummaryWriter(self.log_dir)
        try:
            dummy_input = torch.randn(2 ,3 ,input_shape[0] ,input_shape[1])
            self.writer.add_graph(model ,dummy_input)
        except:
            print("Exception: loss history methods")
            pass

    def append_loss(self , epoch ,loss ,val_loss):
        if not os.path.exists(self.log_dir):
            print("make loss values conserve floder : " ,self.log_dir)
            os.makedirs(self.log_dir)
        self.losses.append(loss)
        self.val_loss.append(val_loss)

        with open(os.path.join(self.log_dir ,"epoch_loss.txt" ),'a') as f:
            f.write((str(loss)))
            f.write("\n")
            f.close()
        with open(os.path.join(self.log_dir ,"epoch_val_loss.txt") ,'a') as f:
            f.write(str(val_loss))
            f.write("\n")
            f.close()
        self.writer.add_scalar('loss' ,loss ,epoch)
        self.writer.add_scalar('val_loss' ,val_loss ,epoch)
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters ,self.losses ,'red' ,linewidth = 2 , label = 'train loss' )
        plt.plot(iters ,self.val_loss ,'coral' ,linewidth = 2 ,label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            plt.plot(iters,scipy.signal.savgol_filter(self.losses ,num ,3))
            plt.plot(iters,scipy.signal.savgol_filter(self.val_loss ,num ,3))
        except:
            print("Exception: loss plot methods")
            pass
        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc = "upper right")
        plt.savefig(os.path.join(self.log_dir ,"epoch_loss.png"))
        plt.cla()
        plt.close("all")

class EvalCallback():

    def __init__(self ,net ,input_shape ,anchors ,anchors_mask ,class_names ,num_classes ,val_lines ,log_dir ,cuda ,\
                 map_out_path=".temp_map_out" ,max_boxes = 100 ,confidence = 0.05 ,nms_iou=0.5 ,letterbox_image=True ,MINOVERLAP=0.5 ,eval_flag=True ,period=1):
        super(EvalCallback ,self).__init__()

        self.net = net
        self.input_shape = input_shape
        self.anchors = anchors
        self.anchors_mask = anchors_mask
        self.class_names = class_names
        self.num_classes = num_classes
        self.val_lines = val_lines
        self.log_dir = log_dir
        self.cuda = cuda
        self.map_out_path = map_out_path
        self.max_boxes = max_boxes
        self.confidence = confidence
        self.nms_iou = nms_iou
        self.letterbox_image = letterbox_image
        self.MINOVERLAP = MINOVERLAP
        self.eval_flag = eval_flag
        self.period = period

        # self.bbox_util = DecodeBox(...)

