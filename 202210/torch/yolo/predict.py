# *********************************************************#
#   预测
#   单张图片  摄像头  FPS  目录遍历
# *********************************************************#
import time
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

if __name__ == "__main__":
    yolo = YOLO()
    # -------------------------------------------------------------------------#
    #  mode 指定预测的模式
    #   'predict'     ： 单张图片预测，如果想对预测过程修改，比如保存图片、截取对象，可以看以下注释
    #   'video'       ： 视频检测，可调用摄像头或者视频进行检测
    #   'fps'         ： 测试fps，使用图片是img的street.jpg
    #   'dir_predict' ： 遍历文件夹进行检测并保存。默认遍历img文件夹，保存到img_out文件夹
    #   'heatmap'     ： 进行预测结果的热力图可视化
    #   'export_mode' ： 模型到处onnx，需要pytorch1.7.1以上
    # -------------------------------------------------------------------------#
    mode = 'video'

    # -------------------------------------------------------------------------#
    #   crop        ： 是否预测单张图片后对目标截取
    #   count       ： 是否进行目标的计数
    #
    #       crop count 仅在 mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = True
    count = True

    # -------------------------------------------------------------------------#
    #   video_path        ： 指定视频路径。设置0时表示检测摄像头。
    #                        想要检测视频，设置如"XXX.mp4"即可
    #   video_save_path   ： 视频保存路径。设置“”时表示不保存。
    #                        想要保存视频，设置如"YYY.mp4"即可
    #   video_fps         ： 用于保存视频的fps
    #
    #        video_path video_save_path video_fps 仅在 mode='video'时有效
    # -------------------------------------------------------------------------#
    video_path = "a1.mp4"
    video_save_path = "a.mp4v"
    video_fps = 10.0

    # -------------------------------------------------------------------------#
    #   test_interval     ： 指定fps,图片检测数量。
    #   fps_image_path    ： 指定测试的fps图片
    #
    #        test_interval fps_image_path 仅在 mode='fps'时有效
    # -------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "street01.jpg"

    # -------------------------------------------------------------------------#
    #   dir_origin_path     ： 指定检测图片的路径文件夹。
    #   dir_save_path       ： 指定检测图片的保存路径。
    #
    #        dir_origin_path dir_save_path 仅在 mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"

    # -------------------------------------------------------------------------#
    #   heatmap_save_path     ： 热力图保存路径，默认保存model_data下。
    #
    #        heatmap_save_path 仅在 mode='heatmap'时有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"

    # -------------------------------------------------------------------------#
    #   simplify        ： 使用Simplify_onnx
    #   onnx_save_path  ： 指定onnx的保存路径
    #
    #        heatmap_save_path 仅在 mode='heatmap'时有效
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode == "predict":
        print("\nPredict Mode. \n\n")
        """
            1 检测完图片进行保存，r_image.save("img.jpg")即可保存，直接在predict.py进行操作
            2 获取预测坐标，进入yolo.detect_image函数，绘图部分[top left bottom right]
            3 截取预测框目标，进入yolo.detect_image函数，用[top left bottom right]和矩阵方式进行截取
            4 在预测框写额外的字，例如检测数量，进入yolo_detect_image函数，对绘图部分predicted_class 进行判断
                例如predict_class == 'car' ，记录后在图片写字
        """
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except Exception as e:
                print("Error: not image. " ,e.__class__.__name__ ,e)
                continue
            else:
                r_image = yolo.detect_image(image,crop = crop ,count= count)
                r_image.show()

    elif mode == "video":
        print("\nVideo Mode. \n")
        print("input video {}\n".format(video_path))
        print("input video {}\n\n".format(video_save_path))

        capture = cv2.VideoCapture(video_path)
        if video_save_path != "":
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out = cv2.VideoWriter(video_save_path ,fourcc ,video_fps ,size)
        ref ,frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取视频（摄像头），请检查输入源")

        fps = 0.0
        while True:
            t1 = time.time()
            ref ,frame = capture.read()
            if not  ref:
                break
            # BGRtoRGB
            frame = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
            # to Image
            frame = Image.fromarray(np.uint8(frame))
            # video
            frame = np.array(yolo.detect_image(frame))

            #RGBtoBGR cv show
            frame = cv2.cvtColor(frame ,cv2.COLOR_RGB2BGR)

            fps = (fps + (1./(time.time() - t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame ,"fps= %.2f"%(fps) ,(0 ,40) ,cv2.FONT_HERSHEY_SIMPLEX ,1 ,(0,255,0) ,2)

            cv2.imshow("video" ,frame)
            c = cv2.waitKey(1) & 0xff
            if video_save_path != "":
                out.write(frame)

            if c == 27 :
                capture.release()
                break
        print("Video Detection Done!")
        capture.release()
        if video_save_path != "":
            print("Save processed video to the path : " + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        print("\nFPS Mode. \n\n")
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img ,test_interval=test_interval)
        print(str(tact_time) + " seconds, " + str(1/tact_time) + "FPS , @batch_size 1")

    elif mode == "dir_predict":
        print("\nDir Predict Mode. \n")
        print("input path {}\n".format(dir_origin_path))
        print("output path {}\n".format(dir_save_path))

        import os
        from tqdm import tqdm
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endsWith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff', )):
                image_path = os.path.join(dir_origin_path ,img_name)
                image = Image.open(image_path)
                r_image = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path ,img_name.replace(".jpg" ,".png")) ,quality=95 ,subsampling=0)

    elif mode == "heatmap" :
        print("\nHeatmap Mode. \n")
        while True:
            img = input("Input image filename:")
            try:
                image = Image.open(img)
            except Exception as e:
                print("Open error" ,e.__class__.__name__ ,e)
                continue
            else:
                yolo.detect_heatmap(image ,heatmap_save_path)

    elif mode == "export_onnx":
        print("\nExport Onnx Mode. \n")
        yolo.covert_to_onnx(simplify ,onnx_save_path)

    else:
        raise AssertionError("Please specity the correct mode : 'predict' ,'video' ,'fps' ,'heatmap' ,'export_onnx' ,'dir_predict'  \n")
