
'''
Date:2022.9.17
简介:
设置
'''

#导入第三方库
import numpy as np
import cv2,time

# 设置需要识别的颜色
armor_color = "blue"
if armor_color == "blue":
    #蓝色armor
    Color_lower = np.array([100, 123, 120])
    Color_upper = np.array([160, 255, 230])
elif armor_color == "red":
    Color_lower = np.array([120, 150, 100])  
    Color_upper = np.array([180, 255, 250])

# 摄像头参数定义,设置
def camera_init(cap,num):
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, num)#设置曝光
    return cap


