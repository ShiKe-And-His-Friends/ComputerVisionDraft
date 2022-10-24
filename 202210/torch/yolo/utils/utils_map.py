import os

""""
    convert the lines of file to a list
"""
def file_lines_to_list(path):
    with open(path ) as f:
        content = f.readlines()
    # remove withespace characters like '\n' at the end of each line

def preprocess_dr(gt_path ,classes_names):
    images_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i , images_id in enumerate(images_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path ,images_id))

def get_coco_map(class_names ,path):
    GT_PATH = os.path.join(path ,"ground-truth")
    DR_PATH = os.path.join(path ,"detection-results")
    COCO_PATH = os.path.join(path ,"coco_eval")

    if not os.path.exists(COCO_PATH):
        os.makedirs(COCO_PATH)
    GT_JOIN_PATH = os.path.join(COCO_PATH ,"instances_gt.json")
    DR_JOIN_PATH = os.path.join(COCO_PATH ,"instances_dr.json")

    with open(GT_JOIN_PATH ,"w") as f:
        result_gt = preprocess_dr(GT_PATH ,class_names)