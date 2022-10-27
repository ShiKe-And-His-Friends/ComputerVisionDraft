import colorsys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw ,ImageFont

from net.yolo import YoloBody
from utils.utils import (cvtColor ,get_anchors ,get_classes ,preprocess_input ,resize_image ,show_config)
from utils.utils_bbox import DecodeBox

""""
    训练自己的数据集需要按注释操作
"""
class YOLO(object):
    _default = {
        #-------------------------------------------------------------------------#
        #  自己的模型训练要修改model_path 和 classes_path
        #  model_path指向log下的权值文件，classes_path指向model_data下的txt
        #
        #  训练好的log文件夹下有多个权值文件夹，选择验证机损失较低的即可
        #  验证集损失较低不代表mAP较高，仅代表权值在验证集上泛化较好
        #  如果出现shape不匹配，同时要注意训练时的moel_path 和 classes_path参数的修改
        # -------------------------------------------------------------------------#
        "model_path"    : 'E:/Torch/yolov4-pytorch-master/model_data/yolo4_weights.pth',
        "classes_path"  : 'model_data/coco_classes.txt',
        # -------------------------------------------------------------------------#
        #  anchors_path代表先验框对应的txt文件，一般不修改
        #  anchors_mask用于帮助代码找到对应的先验框，一般不修改
        # -------------------------------------------------------------------------#
        "anchors_path"  :'model_data/yolo_anchors.txt',
        "anchors_mask"  :[[6,7,8],[3,4,5],[0,1,2]],
        # -------------------------------------------------------------------------#
        #  输入图片大小，必须是32倍数
        # -------------------------------------------------------------------------#
        "input_shape"   : [416 ,416],
        # -------------------------------------------------------------------------#
        #  只有得分大于置信度的预测框会被保留下来
        # -------------------------------------------------------------------------#
        "confidence"    :0.5,
        # -------------------------------------------------------------------------#
        #  非极大抑制用到的nms_iou大小
        # -------------------------------------------------------------------------#
        "nms_iou"       :0.3,
        # -------------------------------------------------------------------------#
        #  该变量用于控制是否使用letterbox_image对输入图像进行不失真resize
        #  在多次测试后，发现关闭letterbox_image直接resize效果更好
        # -------------------------------------------------------------------------#
        "letterbox_image":False,

        "cuda":False,
    }

    @classmethod
    def get_defaults(cls ,n):
        if n in cls._default:
            return cls._default[n]
        else:
            return "Unrecognized attribute name '"+ n + "'"

    #---------------------------------------------#
    #  初始化YOLO
    # ---------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._default)
        for name ,value in kwargs.items():
            setattr(self ,name ,value)
            self._default[name] = value

            # -----------------------------------#
            #  获得先验框种类和数量
            # -----------------------------------#
            self.class_names ,self.num_classes  = get_classes(self.classes_path)
            self.anchors ,self.num_anchors      = get_anchors(self.anchors_path)
            self.bbox_util                      = DecodeBox(self.anchors ,self.num_classes ,(self.input_shape[0] ,self.input_shape[1]) ,self.anchors_mask)

            # -----------------------------------#
            #  画框设置不同的颜色
            # -----------------------------------#
            hsv_tuples = [(x / self.num_classes ,1. ,1 )for x in range(self.num_classes)]
            self.colors = list(map(lambda x : colorsys.hsv_to_rgb(*x) ,hsv_tuples))
            self.colors = list(map(lambda x : (int(x[0] * 255) ,int(x[1] * 255) ,int(x[2] * 255)) ,self.colors))
            self.generate()

            show_config(**self._default)

    # ---------------------------------------------#
    #  生成模型
    # ---------------------------------------------#
    def generate(self ,onnx=False):
        # -----------------------------------#
        #  建立yolo模型，载入yolo模型的权重
        # -----------------------------------#
        self.net = YoloBody(self.anchors_mask ,self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path ,map_location=device))
        self.net = self.net.eval()
        print('{} model ,anchors ,and classes loaded.'.format(self.model_path))

        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    # ---------------------------------------------#
    #  检测图片
    # ---------------------------------------------#
    def detect_image(self ,image ,crop = False ,count=False):
        # -----------------------------------#
        #  计算输入图片的高和宽
        # -----------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # -----------------------------------#
        #  图像转换RBG到灰度
        # -----------------------------------#
        image  = cvtColor(image)
        # -----------------------------------#
        #  给图像加灰条或者直接resize
        # -----------------------------------#
        image_data = resize_image(image ,(self.input_shape[1] ,self.input_shape[0]) ,self.letterbox_image )
        # -----------------------------------#
        #  添加上batch_size维度
        # -----------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data ,dtype='float32')) ,(2,0,1)) ,0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()

            # -----------------------------------#
            #  将图片输入网络中预测
            # -----------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # -----------------------------------#
            #  将预测框进行堆叠，然后进行非极大抑制
            # -----------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs ,1) ,self.num_classes ,self.input_shape,
                    image_shape ,self.letterbox_image ,conf_thres=self.confidence ,nms_thres=self.nms_iou)
            if results[0] is None:
                return image
            top_label = np.array(results[0][:,6] ,dtype = 'int32')
            top_conf = results[0][:,4] * results[0][:,5]
            top_boxes = results[0][:,:4]
        # -----------------------------------#
        #  设置字体和边框厚度
        # -----------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf' ,size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape) ,1))
        # -----------------------------------#
        #  计数
        # -----------------------------------#
        if count:
            print("top_label:" ,top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i] ,":" ,num)
                classes_nums[i] = num
            print("classes_nums:" ,classes_nums)
        # -----------------------------------#
        #  是否进行目标的裁剪
        # -----------------------------------#
        if crop:
            for i , c in list(enumerate(top_label)):
                top ,left ,bottom ,right = top_boxes[i]
                top = max(0 ,np.floor(top).astype('int32'))
                left = max(0 ,np.floor(left).astype('int32'))
                bottom = min(image.size[1] ,np.floor(bottom).astype('int32'))
                right = min(image.size[0] ,np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = images.crop([left ,top ,right ,bottom])
                crop_image.save(os.path.join(dir_save_path ,"crop_"+str(i)) + ".png")
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # -----------------------------------#
        #  图像绘制
        # -----------------------------------#
        for i ,c in list (enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = top_conf[i]

            top,left,bottom,right = box
            top = max(0 ,np.floor(top).astype('int32'))
            left = max(0 ,np.floor(left).astype('int32'))
            bottom = min(image.size[1] ,np.floor(bottom).astype('int32'))
            right = min(image.size[0] ,np.floor(right).astype('int32'))

            label = '{}{:.2f}'.format(predicted_class ,score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label ,font)
            label = label.encode('utf-8')
            print(label ,top ,left ,bottom ,right)

            if top-label_size[1] >= 0 :
                text_origin = np.array([left ,top-label_size[1]])
            else:
                text_origin = np.array([left ,top+1])
            for i in range(thickness):
                draw.rectangle([left+i ,top+i ,right-i ,bottom-i] ,outline=self.colors[c])
            draw.rectangle([tuple(text_origin) ,tuple(text_origin+label_size)] ,fill = self.colors[c])
            draw.text(text_origin ,str(label ,'UTF-8') ,fill=(0,0,0) ,font=font)
            del draw
        return image

    def get_FPS(self ,image ,test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image ,(self.input_shape[1] ,self.input_shape[0]) ,self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data ,dtype='float32')) ,(2,0,1)) ,0)

        with torch.no_grad():
            images =torch.from_numpy(image_data)
            if self.cuda:
                images = images.__cuda_array_interface__
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                results = self.bbox_util.non_max_suppression(torch.cat(outputs ,1) ,self.num_classes ,self.input_shape,
                        image_shape ,self.letterbox_image ,conf_thres=self.confidence ,nms_thres =self.nms_iou)
            t1 = time.time()
            for _ in range(test_interval):
                with torch.no_grad():
                    outputs = self.net(images)
                    outputs = self.bbox_util.decode_box(outputs)
                    results = self.bbox_util.non_max_suppression(torch.cat(outputs ,1) ,self.num_classes ,self.input_shape,
                        image_shape ,self.letterbox_image ,conf_thres = self.confidence ,nms_thres=self.nms_iou)
            t2 = time.time()
            tact_time = (t2 - t1) /test_interval
            return tact_time

    def detect_heatmap(self ,image ,heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        image = cvtColor(image)
        image_data = resize_image(image ,(self.input_shape[1] ,self.input_shape[0]) ,self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data ,dtype='float32')) ,(2 ,0 ,1)) ,0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        plt.imshow(images,alpha=1)
        plt.axis('off')
        mask = np.zeros((image.size[1] ,image.size[0]))

        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b ,c ,h ,w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output ,[b ,3 ,-1 ,h ,w]) ,[0 ,3 ,4 ,1 ,2])[0]
            score = np.max(sigmoid(sub_output[... ,4]) ,-1)
            score = cv2.resize(score ,(image.size[0] ,image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximun(mask ,normed_score)

        plt.imshow(mask ,alpha = 0.5 ,interpolation='nearest' ,cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1 ,bottom=0 ,right=1 ,left=0 ,hspace =0 ,wspace=0)
        plt.margins(0,0)
        plt.savefig(heatmap_save_path ,dpi=200 ,bbox_inches='tight' ,pad_inches=-0.1)
        print("Save to the "+ heatmap_save_path)
        plt.show()

    def covert_to_onnx(self ,simplify ,model_path):
        import onnx
        self.generate(onnx = True)
        im = torch.zeros(1 ,3 ,*self.input_shape).to('cpu')
        input_layer_names = ["images"]
        output_layer_names = ["outputs"]

        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(
            self.net,
            im,
            f = model_path,
            verbose=False,
            opset_version=12,
            training = torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names= input_layer_names,
            output_layer_names = output_layer_names,
            dynamic_axes=None
        )
        model_onnx = onnx.load(model_path) # load onnx model
        onnx.checker.check_model(model_onnx) # check onnx model

        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx ,check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_data=None
            )
            assert check,'assert check failed.'
            onnx.save(model_onnx ,model_path)
        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self ,image_id ,image, class_names ,map_out_path):
        f = open(os.path.join(map_out_path ,"detection-results/" + image_id + ".txt") ,'w')
        image_shape = np.array(np.shape(image)[0:2])
        image = cvtColor(image)
        image_data = resize_image(image ,(self.input_shape[1] ,self.input_shape[0]) ,self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data ,dtype = 'float32')) ,(2 ,0 ,1)) ,0)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data ,dtype='float32')) ,(2,0,1)) ,0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs ,1) ,self.num_classes ,self.input_shape,
                image_shape ,self.letterbox_image ,conf_thres = self.confidence ,mns_thres = self.nms_iou)

            if results[0] in None:
                return
            top_label = np.array(results[0][:,6] ,dtype = 'int32')
            top_conf = results[0][:, 4] * results[0][:,5]
            top_boxes = results[0][:,:4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box = top_boxes[i]
            score = str(top_conf[i])
            top ,left ,bottom ,right = box
            if predicted_class not in class_names:
                continue
            f.write("%s %s %s %s %s %s %s\n" % (predicted_class ,score[:6] ,str(int(left)) ,str(int(top)) ,str(int(right)) ,str(int(bottom))))
        f.close()
        return