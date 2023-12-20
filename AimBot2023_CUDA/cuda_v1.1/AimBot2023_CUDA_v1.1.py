
import cv2, time, math
import torch
import traceback


def AimBot_Check_ROI_Position(x1, y1, x2, y2):
    max_x = 640
    max_y = 480
    if x1 <= 0:
        x1 = 0
    if y1 <= 0:
        y1 = 0
    if x2 >= max_x:
        x2 = max_x
    if y2 >= max_y:
        y2 = max_y
    return (x1, y1, x2, y2)


def AimBot_Yolo_Detect(frame, armor_color, size, device, torch_model):
    armor_information_dict = {}
    M1 = [0, 0];
    M2 = [0, 0]
    armor_list = [];
    key_name_list = []
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
        count = 0  # 记数
        for xmin, ymin, xmax, ymax, class_l, conf in zip(xmins, ymins, xmaxs, ymaxs, class_list, confidences):
            if class_l == detect_armor_class and conf >= 0.3:  # 如果置信度大于0.3
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
def Count_real_position(x,y,M1_x,M2_y):
    real_x_position = M1_x + x
    real_y_position = M2_y - y
    return (real_x_position,real_y_position)

def AimBot_init(device, yolo_path, model_path):
    device = torch.device(device)  # [只有N卡才可以使用cuda加速]

    torch_model = torch.hub.load(yolo_path, 'custom',
                                 model_path, source='local',
                                 force_reload=True)  # 加载本地yolov5模型(需要修改路径和文件)
    return device, torch_model


if __name__ == "__main__":
    global ROI_frame
    device, torch_model = AimBot_init("cuda", "/home/gavin/df-wolf2023/yolov5",
                                      "/home/gavin/df-wolf2023/pytorch_model/armor/weights/best.pt")
    cap = "/home/gavin/df-wolf2023/inference_test/video/test010.mp4"
    #cap = 2
    cap = cv2.VideoCapture(cap)
    state = 0
    roi_get = 0
    while True:
        armor_distance_list = []
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 480))  # 将图片输入大小设置为640*480
        new_frame = frame.copy()
        if state == 0:
            result = AimBot_Yolo_Detect(frame, "blue", 540, device, torch_model)
            print("whole detect")
            armor_dict = result[1]
            armor_information_list = list(armor_dict.values())
            #print(armor_information_list)
            #cv2.imshow("result_output_image", result[0])
            #print('Armor Count:', len(list(armor_dict.keys())), "Armor Name:", list(armor_dict.keys()))
            i = len(armor_information_list)
            if len(armor_information_list) > 0:
                for i in range(i):
                    xmin = list(armor_dict.values())[i - 1][0]
                    ymin = list(armor_dict.values())[i - 1][1]
                    xmax = list(armor_dict.values())[i - 1][2]
                    ymax = list(armor_dict.values())[i - 1][3]
                    armor_x_position = int(xmax - (xmax - xmin) / 2)  # 计算识别目标的X坐标
                    armor_y_position = int(ymax - (ymax - ymin) / 2)  # 计算识别目标的Y坐标
                    print(xmin)
                    #print('Armor XY Position:', armor_x_position, armor_y_position)

                    distance = math.sqrt(abs(320 - armor_x_position) * abs(320 - armor_x_position) + abs(240 - armor_y_position) * abs(240 - armor_y_position))
                    armor_distance_list.append(distance)
                    min_armor_distance = min(armor_distance_list)  # 找到最小的distance(即识别的最终目标)
                    k = armor_distance_list.index(min_armor_distance)
                    new_frame = cv2.rectangle(new_frame, (xmin, ymax), (xmax, ymin), (0, 255, 255), 2)  # 在原图像中画出方框框记识别目标

                state = 1
                xmin = list(armor_dict.values())[k - 1][0]
                ymin = list(armor_dict.values())[k - 1][1]
                xmax = list(armor_dict.values())[k - 1][2]
                ymax = list(armor_dict.values())[k - 1][3]
                armor_x_position = int(xmax - (xmax - xmin) / 2)  # 计算识别目标的X坐标
                armor_y_position = int(ymax - (ymax - ymin) / 2)  # 计算识别目标的Y坐标
                target_width = int(int(xmax) - int(xmin))  # 装甲板长
                target_height = int(int(ymin) - int(ymax))  # 装甲板宽(高)

                M1 = [armor_x_position - 1.5 * target_width,armor_y_position + 1.5 * target_height]
                M2 = [armor_x_position + 1.5 * target_width,armor_y_position - 1.5 * target_height]
                M1[0], M1[1], M2[0], M2[1] = AimBot_Check_ROI_Position(M1[0], M1[1], M2[0], M2[1])

                ROI_frame = frame[int(M1[1]):int(M2[1]), int(M1[0]):int(M2[0])]  # 根据选定目标在原图像中提取ROI

                #cv2.imshow("ROI",ROI_frame)
                new_frame = cv2.rectangle(new_frame, (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)  # 在原图像中画出方框框记识别目标
                new_frame = cv2.rectangle(new_frame, (int(M1[0]), int(M1[1])), (int(M2[0]), int(M2[1])), (255, 0, 204), 2)
                new_frame = cv2.putText(new_frame, "ROI", (int(M1[0]), int(M1[1]) - 10), cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 0, 204))
                new_frame = cv2.line(new_frame, (320, 240), (armor_x_position, armor_y_position), (255, 255, 0), 3)
                new_frame = cv2.line(new_frame, (0, 240), (640, 240), (255, 0, 0), 4)
                new_frame = cv2.line(new_frame, (320, 0), (320, 640), (255, 0, 0), 4)
                continue
            else:
                state = 0
                new_frame = cv2.line(new_frame, (0, 240), (640, 240), (255, 0, 0), 4)
                new_frame = cv2.line(new_frame, (320, 0), (320, 640), (255, 0, 0), 4)
            cv2.imshow("frame", new_frame)
            cv2.waitKey(1)
        elif state == 1:
            if roi_get == 0:
                result = AimBot_Yolo_Detect(ROI_frame, "blue", 300, device, torch_model)
            elif roi_get == 1:
                ret, frame = cap.read()
                frame = cv2.resize(frame, (640, 480))  # 将图片输入大小设置为640*480
                M3[1],M4[1],M3[0],M4[0] = AimBot_Check_ROI_Position(M3[1],M4[1],M3[0],M4[0])
                ROI_frame = frame[int(M3[1]):int(M4[1]), int(M3[0]):int(M4[0])]  # 根据选定目标在原图像中提取ROI
                print(int(M3[1]),int(M4[1]),int(M3[0]),int(M4[0]))
                #time.sleep(3)
                #cv2.imshow("roi_1",ROI_frame)
                #M1=[0,0];M2=[0,0]
                result = AimBot_Yolo_Detect(ROI_frame, "blue", 300, device, torch_model)
            armor_dict = result[1]
            armor_information_list = list(armor_dict.values())

            i = len(armor_information_list)
            if len(armor_information_list) > 0:
                for i in range(i):
                    xmin = list(armor_dict.values())[i - 1][0]
                    ymin = list(armor_dict.values())[i - 1][1]
                    xmax = list(armor_dict.values())[i - 1][2]
                    ymax = list(armor_dict.values())[i - 1][3]
                    armor_x_position = int(xmax - (xmax - xmin) / 2)  # 计算识别目标的X坐标
                    armor_y_position = int(ymax - (ymax - ymin) / 2)  # 计算识别目标的Y坐标

                    cv2.rectangle(ROI_frame, (xmin,ymax), (xmax, ymin),
                                          (0, 255, 0), 2)

                    armor_x_position,armor_y_position = Count_real_position(armor_x_position,armor_y_position,M1[0],M2[1])

                    target_width = int(xmax) - int(xmin) # 装甲板长
                    target_height = int(ymin) - int(ymax) # 装甲板宽(高)
                    M3 = [armor_x_position - 1.5 * target_width, armor_y_position + 1.5 * target_height]
                    M4 = [armor_x_position + 1.5 * target_width, armor_y_position - 1.5 * target_height]

                    #ROI_frame = frame[int(M1[1]):int(M2[1]), int(M1[0]):int(M2[0])]  # 根据选定目标在原图像中提取ROI
                    show_roi_new = frame.copy()
                    frame = cv2.putText(show_roi_new, "detect", (int(armor_x_position), int(armor_y_position) - 10), cv2.FONT_HERSHEY_TRIPLEX, 1,(255, 0, 204))
                    frame = cv2.rectangle(show_roi_new, (int(M3[0]), int(M3[1])), (int(M4[0]), int(M4[1])), (255, 0, 204), 2)
                    roi_get = 1
                    state = 1
                    cv2.imshow("roi_frame",ROI_frame)
                    cv2.imshow("camera",show_roi_new)
            else:
                roi_get = 0
                state = 0
            cv2.waitKey(1)


