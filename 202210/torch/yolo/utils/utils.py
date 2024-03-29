import numpy as np
from PIL import Image

def get_debug_switch_state():
    debug_now = False
    return debug_now

def resize_image(image ,size ,letterbox_image):
    iw ,ih = image.size
    w , h = size
    if letterbox_image:
        scale = min(w/iw ,h/ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw,nh) ,Image.BICUBIC)
        new_image = Image.new('RGB' ,size ,(128 ,128 ,128))
        new_image.paste(image ,((w-nw)//2 ,(h-nh)//2))
    else:
        new_image = image.resize((w,h) ,Image.BICUBIC)
    return new_image

def preprocess_input(image):
    image /= 255.0
    return image

def get_classes(classes_path):
    with open(classes_path ,encoding='utf-8') as f:
        classes_names = f.readlines()
    classes_names = [c.strip() for c in classes_names]
    f.close()
    return classes_names ,len(classes_names)

def get_anchors(anchors_path):
    with open(anchors_path ,encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1 ,2)
    f.close()
    return anchors ,len(anchors)

def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else :
        image = image.convert('RGB')
        return  image

def preprocess_input(image):
    image /= 255.0
    return image

# 获得学习率
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def show_config(**kwargs):
    print("Configurations:")
    print('-'*100)
    print('|%25s | %70s ' % ('keys' ,'values'))
    print('-'*100)
    for key,value in kwargs.items() :
        print('|%25s | %70s |' % (str(key) ,str(value)))
    print('-'*100)
