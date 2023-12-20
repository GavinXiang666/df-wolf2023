
'''
Date:2022.9.17
简介:
RMYC2023的初代装甲板识别的传统Demo程序
此程序为基于Opencv-Python的传统视觉
'''

import cv2,math
import time,serial
import numpy as np
from main import *

#计算帧率的变量
fps_time_past = 0
fps_time_now = 0

armor_distance = 6.2
armor_contours=[]#装甲板轮廓列表
armor_area_list = []#装甲板面积列表

com = serial.Serial("/dev/ttyS0",115200)#设置通信串口
def send_message(x_position,y_position,distance):
    print("send",x_position,y_position)
    send_string = str(x_position)+','+str(y_position)+','+str(distance)#将坐标信息整合到send_string中
    com.write(send_string.encode('utf-8'))#将send_string发送给EP(EP和开发板的串口波特率需要保持一致

k = np.ones((13,13),np.uint8)#创建10*10的数组作为核
erode = np.ones((1,1),np.uint8)

def count_distance(P):
    return (armor_distance*464.0)/P

'''
    Image_Change()函数注释:
mask_image---------经过hsv颜色通道处理传过来的图像
'''
def Image_Change(mask_image):
    res = cv2.morphologyEx(mask_image,cv2.MORPH_GRADIENT,k)
    return res#返回处理完成的图像
'''
    recognize_armor()函数注释:
res---------------传过来的图像
frame-------------摄像头原图像
'''
def recognize_armor(res,frame):
    armor_contours = []#清空列表
    fontStyle = cv2.FONT_HERSHEY_TRIPLEX#设置字体样式
    res = cv2.Canny(res,100,1050)#Canny边缘检测
    #cv2.imshow("canny",res)#显示Canny边缘检测后的图像(测试，可注释该行代码)
    # # contours--轮廓，hierarchy--等级制度，层次体系
    contours, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#寻找轮廓(只识别外围轮廓)
	
    contours_amount = len(contours)#统计轮廓的数量
    #print("未筛选前的轮廓数量：",contours_amount)
    armor_contours_amount = 0#计算轮廓数量
    for count in range(contours_amount):#遍历所有的轮廓(用于筛选出装甲板)
        x, y, w, h = cv2.boundingRect(contours[count])#获取坐标，宽高
        if w>=20 and h>=20:#增加判断装甲板条件
            armor_contours.append(contours[count])
            armor_contours_amount = armor_contours_amount + 1
        else:
            pass#如果不符合判断条件，就跳过该轮廓
    #print("经过筛选后的轮廓数量：",armor_contours_amount)
    try:
        armor_distance_list = []#重置列表，用于筛选出最大面积装甲板
        for i in range(armor_contours_amount):
            x, y, w, h = cv2.boundingRect(armor_contours[i])
            distance = math.sqrt(abs(y-240)*abs(y-240) + abs(x-320)*abs(x-320))
            armor_distance_list.append(distance)

        min_dis = armor_distance_list[0]
        for k in range(1,len(armor_distance_list)):
            if min_dis >= armor_distance_list[k]:
                min_dis = armor_distance_list[k]
        
        min_dis_index = armor_distance_list.index(min_dis)
        finally_armor_information = armor_contours[min_dis_index]
        x, y, w, h = cv2.boundingRect(finally_armor_information)
        #draw_armor = cv2.drawContours(frame,armor_contours,-1,(0,255,0),2)
        
        #print("成功匹配图像", "\n图像宽度:", w, "图像高度:", h)#打印测试（可注释该行代码)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),5)#在原画面上绘制出方框
        real_distance = count_distance(w)
        print("distance",real_distance)
        position_string = "X:"+str(x)+"Y:"+str(y)#讲数据整合成一个字符串，用于后续Uart通信方便
        
        send_message(x,y,real_distance)#发送坐标信息
        cv2.putText(frame,position_string,(x,y-40),fontStyle,1,(0,255,255))#在识别的装甲板上显示装甲板坐标
        #print("已完成一次识别")
        return frame#返回图像
    except:
        return frame#返回原图像

'''
    Get_Frame()函数参数注释:
cap_frame----------摄像头视频流
Color_Lower-----
               |---HSV设置的颜色区域
Color_upper-----
display------------是否显示图传窗口[0为不显示,1为显示]
'''
def Get_Frame(cap_frame,Color_lower,Color_upper,display):
    fps_time_past = time.time()
    ret, frame = cap_frame.read()  # 读取摄像头,获取视频流,ret表示是否读取到视频流,frame表示获取到的图像信息
    #print(ret)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)#HSV颜色设置
    mask = cv2.inRange(hsv, Color_lower, Color_upper)#提取在HSV范围内的图像
    res = Image_Change(mask)#膨胀腐蚀
    new_frame = recognize_armor(res,frame)#提取特征圈出目标
    if display == 1:#显示视频流
        fps_time_now=time.time()
        show_fps_string = "FPS:"+str(1/(fps_time_now-fps_time_past))
        cv2.putText(frame,show_fps_string, (100, 100), cv2.FONT_HERSHEY_TRIPLEX, 2, (0, 255, 255))  # 显示文字在方框之上
        #cv2.imshow("HSV",hsv)#HSV通道画面
        #cv2.imshow("HSV_MASK",mask)#HSV提取出的颜色画面
        #cv2.imshow("RES",res)#腐蚀膨胀后的画面               
        cv2.imshow('frame', frame)
        
        #print('FPS:',1/(fps_time_now-fps_time_past))
    elif display == 0:#不显示视频流
        pass#占位


if __name__ == "__main__":
    armor_color = "blue"
    if armor_color == "blue":
        #蓝色armor
        Color_lower = np.array([100, 150, 160])
        Color_upper = np.array([140, 255, 200])
    elif armor_color == "red":
    	#红色armor
        Color_lower = np.array([120, 160, 100])  
        Color_upper = np.array([180, 255, 250])
    cap = cv2.VideoCapture(0);cap = camera_init(cap,100)
    while True:
        Get_Frame(cap,Color_lower, Color_upper,1)
        if cv2.waitKey(1) & 0xFF == 27:#按ESC退出
            print("程序结束")
            break  # 退出
