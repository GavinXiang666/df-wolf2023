
'''
@Gavin向
AimBot2023识别装甲板算法部分代码
Date:2023.6.5
'''

import cv2
import torch
import time,math,serial
import numpy as np
import traceback
'''
AimBot_init函数:
-device 推理设备模式
-yolo_path Yolo文件目录
-model_path 模型文件
'''
def AimBot_init(device, yolo_path, model_path):
    device = torch.device(device)  # 设置推理设备

    torch_model = torch.hub.load(yolo_path, 'custom',
                                 model_path, source='local',
                                 force_reload=True)  # 加载本地yolov5模型(需要修改路径和文件)
    return device, torch_model#返回

def AimBot_Yolo_Detect(frame, armor_color, size, conference, device, torch_model):
    armor_information_dict = {}
    armor_list = [];key_name_list = []
    fontStyle = cv2.FONT_HERSHEY_TRIPLEX  # 设置显示字体样式

    if armor_color == "red":
        detect_armor_class = 16
    elif armor_color == "blue":
        detect_armor_class = 15
    torch_model = torch_model.to(device)
    results = torch_model(frame, size=size)  # 推理图像
    try:  # 尝试
        xyxy = results.pandas().xyxy[0].values
        xmins, ymins, xmaxs, ymaxs, class_list, confidences = xyxy[:, 0], xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], xyxy[:,5], xyxy[:, 4]
        count = 0
        for xmin, ymin, xmax, ymax, class_l, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
            if class_l == detect_armor_class and conf >= conference:  # 如果置信度大于0.3
                count = count + 1
                key_name = "armor" + str(count)
                key_name_list.append(key_name)
                armor_list.append([int(xmin), int(ymin), int(xmax), int(ymax)])  # 将识别物体信息加入armor_list列表

        armor_information_dict = zip(key_name_list, armor_list)
        armor_information_dict = dict(armor_information_dict)
        
    except:  # 若出现错误：
        traceback.print_exc()  # 打印报错信息
        return (frame, armor_information_dict)
    return (frame, armor_information_dict)

   