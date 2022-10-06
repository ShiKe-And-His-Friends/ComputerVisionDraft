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