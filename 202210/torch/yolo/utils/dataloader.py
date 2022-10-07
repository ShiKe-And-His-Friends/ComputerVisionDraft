import cv2
from random import sample , shuffle
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
from utils.utils import cvtColor ,preprocess_input

class YoloDataset():
    def __init__(self ,annotation_lines ,input_shape ,num_classes ,train = True):
        self.annotation_lines = annotation_lines
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.train = train
        self.length = len(self.annotation_lines)
        image ,box = self.get_random_data(self.annotation_lines[3], self.input_shape, random=self.train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length
        image, box =  self.get_random_data(self.annotation_lines[index] ,self.input_shape ,random = self.train)
        image = np.transpose(preprocess_input(np.array(image,dtype=np.float32)) ,(2,0,1))
        box = np.array(box ,dtype=np.float32)
        if (len(box) !=0 ):
            box[:,[0,2]] = box[:,[0,2]] / self.input_shape[1]
            box[:,[1,3]] = box[:,[1,3]] / self.input_shape[0]
            box[:,2:4] = box[:,2:4] - box[:,0:2]
            box[:,0:2] = box[:,0:2] + box[:,2:4] / 2
        return image,box

    def get_random_data(self ,annotation_line ,input_shape ,random = True):
        # --------------------------------#
        #  Configure
        # --------------------------------#
        jitter = .3
        hue = .1
        sat = .7
        val = .4

        line = annotation_line.split()
        # 读取图片并转化为RGB图像
        image = Image.open(line[0])
        image = cvtColor(image)
        # 获取图像宽高
        iw ,ih = image.size
        h , w = input_shape
        # 获得预测框
        box =  np.array([np.array(list(map(int ,box.split(',')))) for box in line[1:]])
        if not random:
            scale = min(w/iw ,h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            #--------------------------------#
            #  图像多余部分加上灰条
            # --------------------------------#
            image = image.resize((nw,nh) ,Image.BICUBIC)
            new_image = Image.new('RGB' ,(w,h) ,(128 ,128 ,128))
            new_image.paste(image ,(dx ,dy))
            image_data = np.array(new_image ,np.float32)
            # --------------------------------#
            #  对真实框进行调整
            # --------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[: ,[0,2]] *nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]] * nh / ih + dy
                box[:,0:2][box[:,0:2]<0] = 0
                box[:,2][box[:,2]>w] = w
                box[:,3][box[:,3]>h] = h
                box_w = box[:,2] - box[:,0]
                box_h = box[:,3] - box[:,1]
                box = box[np.logical_and(box_w > 1, box_h > 1)] #discard invalid box
            return image_data ,box
        # --------------------------------#
        #  对图像进行缩放并进行长和宽的扭曲
        # --------------------------------#
        new_ar = iw/ih *self.rand(1-jitter ,1+jitter) / self.rand(1-jitter ,1+jitter)
        scale = self.rand(.25 ,2)
        if (new_ar < 1):
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw,nh) ,Image.BICUBIC)
        # --------------------------------#
        #  图像多余部分加上灰条
        # --------------------------------#
        dx = int (self.rand(0 ,w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image
        # --------------------------------#
        #  翻转图像
        # --------------------------------#
        flip = self.rand() < 0.5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image_data = np.array(image,np.uint8)
        # --------------------------------#
        #  对图像进行色域变换
        #  计算色域变换参数
        # --------------------------------#
        r = np.random.uniform(-1 ,1 ,3) * [hue ,sat ,val] + 1
        # --------------------------------#
        #  图像转到HSV
        # --------------------------------#
        hue ,sat ,val = cv2.split(cv2.cvtColor(image_data ,cv2.COLOR_RGB2HSV))
        dtype = image_data.dtype
        # --------------------------------#
        #  应用变换
        # --------------------------------#
        x = np.arange(0 ,256 ,dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1] ,0 ,255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        image_data = cv2.merge((cv2.LUT(hue,lut_hue) ,cv2.LUT(sat ,lut_sat) ,cv2.LUT(val ,lut_val)))
        image_data = cv2.cvtColor(image_data ,cv2.COLOR_HSV2RGB)
        # --------------------------------#
        #  对真实框进行调整
        # --------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            if flip:
                box[:,[0,2]] = w - box[:,[2,0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]
        return image_data ,box

    def rand(self ,a=0.0 ,b=1.0):
        return np.random.rand()*(b-a)+a

# print("shikeDebug",box)
'''
plt.figure('dog')
plt.imshow(image)
plt.show()
'''