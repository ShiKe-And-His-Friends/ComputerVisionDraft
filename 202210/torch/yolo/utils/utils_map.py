import glob
import os
import json
import shutil
import sys
import cv2
import math

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except:
    pass


""""
    throw error and exit
"""
def error(msg):
    print(msg)
    sys.exit(0)

"""
    log-average miss rate:
        calculated by averaging miss rate at 9 evenly spaced FPPI points
        between 10e-2 an 10e0 ,in log-space
        
    output:
        lamr | log-average miss rate
        mr | miss rate
        fppi | false positive per image
    
    references:
        [1] Dollar ,pointer ,et al "Pedestrian Detection: An Evaluation of the 
            State of the Art." (2012)
"""
def log_average_miss_rate(precision ,fp_cumsum ,num_images):
    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr ,mr ,fppi
    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi ,0 ,-1.0)
    mr_tmp = np.insert(mr,0,0)

    ref = np.logspace(-2.0 ,0.0 ,num=9)
    for i ,ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximun(1e-10 ,ref))))
    return lamr ,mr ,fppi

"""
    Calculate the AP given the recall and precision array
    1st> we compute a version of the measured precision/recall curve with
        precision montonically decreassing
    2nd> we compute the AP as the area under this curve the numerical integration
"""
def voc_ap(rec ,prec):
    '''
        --- official matlab code VOC2012 ---
        mrec = [0 ; rec ; 1];
        mpre = [0 ; prec ; 0];
        for i = numel(mpre)-1:-1:1
            mpre(i) = max(mpre(i) ,mpre(i+1));
        end
        i = find((mrec(2:end) != mrec(1:end-1) + 1));
        ap = sum((mrec(i) - mrec(i-1)).*mprec(i))
    '''
    rec.insert(0 ,0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0 ,0.0)
    prec.append(0.0)
    mpre = prec[:]
    '''
        Thie part makes the precision monotonically decreasing
        for i = numel(mpre)-1 : -1 : 1
            mpre(i) = max(mpre(i) ,mpre(i+1))
    '''
    for i in range(len(mpre)-2 ,-1 ,-1):
        mpre[i] = max(mpre[i],mpre[i+1])

    '''
        This part creates a list of indexes where the recall changes
        i = find(mrec(2:end)!= mrec(1:end-1)+1);
    '''
    i_list = []
    for i in range(1 ,len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    '''
        The Average Precision(AP) is the area under the curve(numerical integration)
        ap = sum((mrec(i) - mrec(i-1)).*mpre(i))
    '''
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1]) * mpre[i])
    return ap ,mrec ,mpre

""""
    convert the lines of file to a list
"""
def file_lines_to_list(path):
    with open(path ) as f:
        content = f.readlines()
    # remove withespace characters like '\n' at the end of each line
    content = [x.strip() for x in content]
    return content

"""
    Draws text in image
"""
def draw_text_in_image(image ,text ,pos ,color ,line_width):
    font = cv2.FONT_HERSHEY_PLAIN
    fontScale = 1
    lineType = 1
    bottomLeftCOrnerOfText = pos
    cv2.putText(image ,text,
                bottomLeftCOrnerOfText,
                font,
                fontScale,
                color,
                lineType)
    text_width , _ = cv2.getTextSize(text, font ,fontScale ,lineType)[0]
    return image ,(line_width + text_width)

def preprocess_dr(gt_path ,class_names):
    images_ids = os.listdir(gt_path)
    results = {}

    images = []
    bboxes = []
    for i , images_id in enumerate(images_ids):
        lines_list = file_lines_to_list(os.path.join(gt_path ,images_id))
        images_id = os.path.splitext(images_id)[0]
        for line in lines_list:
            line_split = line.split()
            confidence , left ,top ,right ,bottom = line_split[-5:]
            class_name = ""
            for name in line_split[:-5]:
                class_name += name +  " "
            class_name = class_name[:-1]
            left ,top ,right ,bottom = float(left),float(top),float(right),float(bottom)
            result = {}
            result["image_id"] = str(images_id)
            if class_name not in class_names:
                continue
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left ,top ,right - left ,bottom - top]
            result["score"] = float(confidence)
            results.append(result)
        return results

def get_map(MINOVERLAP , draw_plot ,score_threhold = 0.5 ,path = './map_out'):
    GT_PATH = os.path.join(path ,"ground-truth")
    DR_PATH = os.path.join(path ,"detection-results")
    IMG_PATH = os.path.join(path ,"images-optional")
    TEMP_FILES_PATH = os.path.join(path ,".temp_files")
    RESULTS_FILES_PATH = os.path.join(path ,"results")

    show_animation = True
    if os.path.exists(IMG_PATH):
        for dirpath ,dirnames ,files in os.walk(IMG_PATH):
            if not files:
                show_animation = False
    else:
        show_animation = False

    if not os.path.exists(TEMP_FILES_PATH):
        os.makedirs(TEMP_FILES_PATH)
    if os.path.exists(RESULTS_FILES_PATH)
        shutil.rmtree((RESULTS_FILES_PATH))
    else:
        os.makedirs(RESULTS_FILES_PATH)

    if draw_plot :
        try:
            matplotlib.use("TKAgg")
        except:
            pass
        os.makedirs(os.path.join(RESULTS_FILES_PATH ,"AP"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "F1"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Recall"))
        os.makedirs(os.path.join(RESULTS_FILES_PATH, "Precision"))
    if show_animation:
        os.makedirs(os.path.join(RESULTS_FILES_PATH ,"images" ,"detection_one_by_one"))
    ground_truth_files_list = glob.glob(GT_PATH + "/*.txt")
    if len(ground_truth_files_list) == 0:
        error("Error:No ground-truth files found.")
    ground_truth_files_list.sort()
    gt_counter_per_class = {}
    counter_images_per_class = {}

    for txt_file in ground_truth_files_list:
        file_id = txt_file.split(".txt" ,1)[0]
        filr_id = os.path.basename(os.path.normpath(file_id))
        temp_path = os.path.join(DR_PATH ,(file_id + ".txt"))
        if not os.path.exists(temp_path):
            error_msg = "Error not found:{}\n".format(temp_path)
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                    class_name ,left ,top ,right ,bottom ,_difficult = line.split()
                    is_difficult = True
                else:
                    class_name ,left, top, right ,bottom = line.split()
            except:
                if "difficult" in line:
                    lines_split = line.split()
                    _difficult = lines_split[-1]
                    bottom = lines_split[-2]
                    right = lines_split[-3]
                    top = lines_split[-4]
                    left = lines_split[-5]
                    class_name = ""
                    for name in lines_split[:-5]:
                        class_name += name + " "
                    class_name = class_name[:-1]
                    is_difficult = True
                else:
                    lines_split = line.split()
                    bottom = lines_split[-1]
                    right = lines_split[-2]
                    top  = lines_split[-3]
                    left = lines_split[-4]
                    class_name = ""
                    for name in lines_split[:-4]:
                        class_name += name + " "
                    class_name = class_name[:-1]
            bbox = left + " " + top + " " + right + " " + bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name ,"bbox":bbox ,"used":False ,"difficult" ,True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name ,"bbox":bbox ,"used":False})
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    gt_counter_per_class[class_name] = 1
                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)
        with open(TEMP_FILES_PATH+ "/" + file_id + "_gorund_truth.json" ,"w") as outfile:
            json.dump(bounding_boxes ,outfile)
    gt_classes = list(gt_counter_per_class.keys())
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    dr_file_list = glob.glob(DR_PATH + "/*.txt")
    dr_file_list.sort()
    for class_index ,class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_file_list:
            file_id = txt_file.split(".txt",1)[0]
            file_id = os.path.basename(os.path.normpath(file_id))
            temp_path = os.path.join(GT_PATH ,(file_id + ".txt") )
            if class_index == 0:
                if not os.exits(temp_path):
                    error_msg = "Error. File not found:{}\n".format(temp_path)
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name ,confidence ,left ,top ,right ,bottom = line.split()
                except:
                    line_split = line.split()
                    bottom = line_split[-1]
                    right = line_split[-2]
                    top = line_split[-3]
                    left = line_split[-4]
                    confidence = line_split[-5]
                    tmp_class_name = ""
                    for name in line_split[:-5]:
                        tmp_class_name += name + " "
                    tmp_class_name = tmp_class_name[:-1]

                if tmp_class_name == class_name:
                    bbox = left + " " + top + " " + right + " " + bottom
                    bounding_boxes.append({"confidence":confidence ,"file_id":file_id ,"bbox":bbox})

        bounding_boxes.sort(key=lambda x:float(x['confidence']) ,reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json" ,"w") as outfile:
            json.dump(bounding_boxes ,outfile)
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}

    with open(RESULTS_FILES_PATH + "/results.txt" ,"w") as result_file:
        result_file.write("# AP and precision/recall per class.\n")
        count_true_positives = {}

        for class_index ,class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))

            nd = len(dr_data)
            tp = [0] * nd
            fp = [0] * nd
            score = [0] * nd
            score_threhold_idx = 0
            for idx ,detection in enumerate(dr_data):
                file_id = detection["file_id"]
                score[idx] = float(detection["confidence"])
                if score[idx] >= score_threhold:
                    score_threhold_idx = idx

                if show_animation:
                    ground_truth_img = glob.glob1(IMG_PATH ,file_id+".*")
                    if len(ground_truth_img) == 0:
                        error("Error ,Image not found with id :" + file_id)
                    elif len(ground_truth_img) > 1:
                        error("Error, Multiple image with id: " + file_id)
                    else:
                        img = cv2.imread(IMG_PATH + "/" + ground_truth_img[0])
                        img_cumulative_path = RESULTS_FILES_PATH + "/images/" + ground_truth_img[0]
                        if os.path.isfile(img_cumulative_path):
                            img_cumulative = cv2.imread(img_cumulative_path)
                        else:
                            img_cumulative = img.copy()
                        bottom_border = 60
                        BLACK = [0 ,0 ,0]
                        img = cv2.copyMakeBorder(img,0 ,bottom_border ,0 ,0 ,cv2.BORDER_CONSTANT ,value=BLACK)
                gt_file = TEMP_FILES_PATH + "/" + file_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                bb = [float(x) for x in detection["bbox"].split()]
                for obj in ground_truth_data:
                    if obj["class_name"] == class_name:
                        bbgt = [float(x) for x in obj['bbox'].split()]
                        bi = [max(bb[0] ,bbgt[0]) ,max(bb[1] ,bbgt[1]) ,min(bb[2],bbgt[2]) ,min(bb[3] ,bbgt[3])]
                        iw = bi[2] - bi[0] + 1
                        ih = bi[3] - bi[1] + 1
                        if iw > 0 and ih > 0:
                            ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0] + 1) \
                                * (bbgt[3] - bbgt[1] + 1) - iw * ih
                            ov = iw * ih /ua

                            if ov > ovmax:
                                ovmax = ov
                                gt_match = obj
                if show_animation:
                    status = "No MATCH FOUND!"
                min_overlap = MINOVERLAP
                if ovmax >= min_overlap:
                    if "difficult" not in gt_match:
                        if not bool(gt_match["used"]):
                            tp[idx] = 1
                            gt_match["used"] = True
                            count_true_positives[class_name] += 1
                            with open(gt_file ,"w") as f:
                                f.write(json.dumps(ground_truth_data))
                            if show_animation:
                                status = "MATCH!"
                        else:
                            fp[idx] = 1
                            if show_animation:
                                status = "REPEATED MATCH!"
                    else:
                        fp[idx] =1
                        if ovmax > 0:
                            status = "INSUFFICIENT OVERLAP"
                    """
                        draw image to show animation
                    """
                    if show_animation:
                        height ,width = img.shape[:2]
                        white = (255 ,255 ,255)
                        light_blue = (255,200,100)
                        green = (0,255,0)
                        light_red = (30,30,255)
                        margin = 10
                        # 1nd line
                        v_pos = int(height - margin - (bottom_border / 2.0))
                        text = "Image: "+ ground_truth_img[0] + " "
                        img,line_width = draw_text_in_image(img ,text ,(margin ,v_pos) ,white ,0)
                        text = "Class [" + str(class_index) + "/" + str(n_classes) + "]:" + class_name + " "
                        img ,line_width = draw_text_in_image(img ,text ,(margin+ line_width,v_pos) ,light_blue ,line_width)
                        if ovmax != -1:
                            color = light_red
                            if status == "UFFICIENT OVERLAP":
                                text = "IOU:{0:.2f} %".format(ovmax*100) + " < {0:.2f}% ".format(min_overlap * 100)
                            else:
                                text = "IOU:{0:.2f} %".format(ovmax*100) + ">={0:.2f}% ".format(min_overlap * 100)
                                color = green
                            img , _ = draw_text_in_image(img ,text ,(margin+line_width ,v_pos) ,color ,line_width)
                        # 2nd line
                        v_pos  += int(bottom_border / 2.0)
                        rank_pos = str(idx+1)
                        text = "Detection #rank:" + rank_pos + " confidecne:(0:.2f) %".format(float(detection["confidence"])* 100)
                        img ,line_width = draw_text_in_image(img ,text ,(margin ,v_pos) ,white ,0)
                        color = light_red
                        if status == "MATCH!":
                            color = green
                        text = "Result:" + status + " "
                        img,line_width = draw_text_in_image(img ,text ,(margin+line_width ,v_pos) ,color ,line_width)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        if ovmax > 0:
                            bbgt = [int(round(float(x))) for x in gt_match['bbox'].split()]
                            cv2.rectangle(img ,(bbgt[0] ,bbgt[1]) ,(bbgt[2] ,bbgt[3]) ,light_blue ,2)
                            cv2.rectangle(img_cumulative ,(bbgt[0] ,bbgt[1]) ,(bbgt[2] ,bbgt[3]) ,light_blue ,2)
                            cv2.putText(img_cumulative ,class_name ,(bbgt[0] ,bbgt[1] - 5) , font ,0.6 ,light_blue ,1 ,cv2.LINE_AA)
                        bb = [int(i) for i in bb]
                        cv2.rectangle(img ,(bb[0] ,bb[1]) ,(bb[2] ,bb[3]) ,color ,2)
                        cv2.rectangle(img_cumulative ,(bb[0] ,bb[1]) ,(bb[2],bb[3]) ,color ,2)
                        cv2.putText(img_cumulative ,class_name ,(bb[0],bb[1] -5) ,font ,0.6 ,color ,1 ,cv2.LINE_AA)
                        cv2.imshow("Animation",img)
                        cv2.waitKey(20)
                        output_img_path = RESULTS_FILES_PATH + "/images/detections_one_by_one/" + class_name + "_detection" + str(idx) + ".jpg"
                        cv2.imwrite(output_img_path ,img)
                        cv2.imweitr(img_cumulative_path ,img_cumulative)

                cunsum = 0
                for idx ,val in enumerate(fp):
                    fp[idx] += cunsum
                    cunsum += val
                cunsun = 0
                for idx ,val in enumerate(tp):
                    tp[idx] += cunsum
                    cunsum += val
                rec = tp[:]
                for idx ,val in enumerate(tp):
                    rec[idx] = float(tp[idx]) / np.maximum(gt_counter_per_class[class_name] ,1)
                prec = tp[:]
                for idx ,val in enumerate(tp):
                    prec[idx] = float(tp[idx]) / np.maximum((fp[idx] + tp[idx]) ,1)
                ap ,mrec ,mprec = voc_ap(rec[:] ,prec[:])

                F1 = np.array(rec) * np.array(prec)*2 / np.where((np.array(prec) + np.array(rec)) == 0 ,1,(np.array(prec) + np.array(rec)))
                sum_AP += ap
                text = "{0:.2f} %".format(ap*100) + " = " + class_name + " AP " #class_name + "AP = {0:.2f}%".format(ap*100)

                if len(prec) > 0:
                    F1_text = "{0:.2f}".format(F1[score_threhold_idx]) + " = " + class_name + " F1 "
                    Recall_text = "{0:.2f}%".format(rec[score_threhold_idx]*100) + " = " + class_name + " Recall "
                    Precision_text = "{0:,2f}%".format(prec[score_threhold_idx]*100) + " = " + class_name + " Precision "
                else :
                    F1_text = "0.00" + " = " + class_name + " F1 "
                    Recall_text = "0.00f" + " = " + class_name + " Recall "
                    Precision_text = "0.00%" + " = " + class_name + " Precision "
                rounded_prec = ['%.2f' % elem for elem in prec]
                rounded_rec = ['%.2f' % elem for elem in rec]
                result_file.write(text+"\n Precision: " + str(rounded_prec) + "\bRecall: " + str(rounded_rec) + "\n\n")

                if len(prec) > 0:
                    print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " +"F1= " + "{0:.2f}".format(F1[score_threhold_idx])\
                        + " ; Recall = " + "{0:.2f}%".format(rec[score_threhold_idx] * 100) + " ;Precision= " + "{0:.2f}%".format(prec[score_threhold_idx] * 100))
                else:
                    print(text + "\t||\tscore_threhold=" + str(score_threhold) + " : " + "F1=0.00% ;Recall= 0.00% ; Precision=0.00%")
                ap_dictionary[class_name] = ap

                n_images = counter_images_per_class[class_name]
                lamr ,mr ,fppi = log_average_miss_rate(np.array(rec) ,np.array(fp) ,n_images)
                lamr_dictionary[class_name] = lamr
                if draw_plot:
                    plt.plot(rec,prec ,"-o")
                    area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                    area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                    plt.fill_between(area_under_curve_x ,0 ,area_under_curve_y ,alpha=0.2 ,edgecolor ='r' )

                    fig = plt.gcf()
                    fig.canvas.set_window_title("AP_" + class_name)
                    plt.title('class:' + text)
                    plt.xlabel('Recall')
                    plt.ylabel("Precision")
                    axes = plt.gca()
                    axes.set_xlim([0.0 ,1.0])
                    axes.set_ylim([0.0 ,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/AP/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score ,F1 ,"-" ,color = 'orangered')
                    plt.title('class:' + F1_text + "\nscore_threhold=" + str(score_threhold))
                    plt.xlabel("Score_threhold")
                    plt.ylabel("F1")
                    axes = plt.gca()
                    axes.set_xlim([0.0 ,1.0])
                    axes.set_ylim([0.0 ,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/F1/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score ,rec ,"-H" ,color = 'gold')
                    plt.title('class: ' +Recall_text + "\nScore_threhold=" + str(score_threhold) )
                    plt.xlabel("score_Threhold")
                    plt.ylabel("Recall")
                    axes =plt.gca()
                    axes.set_xlim([0.0 ,1.0])
                    axes.set_ylim([0.0 ,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/Recall/" + class_name + ".png")
                    plt.cla()

                    plt.plot(score ,prec ,"-s" ,color ='palevioletred')
                    plt.title("class: " + Precision_text + "\nScore_threhold=" + str(score_threhold))
                    plt.xlabel("Score_Threhold")
                    plt.ylabel("Precision")
                    axes = plt.gca()
                    axes.set_xlim([0.0 ,1.0])
                    axes.set_ylim([0.0 ,1.05])
                    fig.savefig(RESULTS_FILES_PATH + "/Precision/" + class_name + '.png"')
                    plt.cla()

                # TODO show animations
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
        json.dump(result_gt ,f ,indent = 4)
        f.close()

    with open(DR_JOIN_PATH ,"w") as f:
        result_dr = preprocess_dr(DR_PATH ,class_names)
        json.dump(result_dr ,f ,indent = 4)
        if len(result_dr) == 0:
            print("未检测到任何目标")
            return [0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ]
    cocoGt = COCO(GT_JOIN_PATH)
    cocoDt = cocoGt.loadRes(DR_JOIN_PATH)
    cocoEval = COCOeval(cocoGt ,cocoDt ,"bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats
