import os

class LossHistory():
    def __init__(self ,log_dir ,model ,input_shape):
        self.log_dir = log_dir
        self.losses = []
        self.val_loss = []
        print("log dir" ,self.log_dir)
        os.makedirs(self.log_dir)

        return None


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

