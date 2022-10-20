import numpy as np

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