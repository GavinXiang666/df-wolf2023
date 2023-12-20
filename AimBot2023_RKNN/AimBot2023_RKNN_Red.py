'''
此项目基于RKNN-Toolkit2仓库中的yolov5目标检测项目修改,原仓库链接:
https://github.com/rockchip-linux/rknn-toolkit2
安装好依赖以后可以使用RK3588(s)的NPU进行加速推理，使用的模型经INT8量化
'''
import os,math
import urllib
import traceback
import time
import serial
import sys
import numpy as np
import cv2
from rknnlite.api import RKNNLite as RKNN
print("start")
ONNX_MODEL = 'yolov5s.onnx'
RKNN_MODEL = 'armor.rknn'
IMG_PATH = './bus.jpg'
DATASET = './dataset.txt'

QUANTIZE_ON = True

OBJ_THRESH = 0.25
NMS_THRESH = 0.45
IMG_SIZE = 640

CLASSES = ('dog', "person", "cat", "tv", "car", "meatballs", "marinara sauce", "tomato soup", "chicken noodle soup",
"french onion soup", "chicken breast","ribs", "pulled pork", "hamburger", "cavity", "red","blue")

com = serial.Serial("/dev/ttyS0",115200)

armor_distance = 6.2

#Uart通信函数
def send_message(bool,x_position,y_position,distance):
    if bool == True:#如果bool为真才会发送信息``````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````````1
        
        print("send",x_position,y_position)
        send_string = str(x_position)+','+str(y_position)+','+str(distance)#将坐标信息整合到send_string中
        com.write(send_string.encode('utf-8'))#将send_string发送给EP(EP和开发板的串口波特率需要保持一致)
    else:
        pass

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xywh2xyxy(x):
    # Convert [x, y, w, h] to [x1, y1, x2, y2]
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def process(input, mask, anchors):

    anchors = [anchors[i] for i in mask]
    grid_h, grid_w = map(int, input.shape[0:2])

    box_confidence = sigmoid(input[..., 4])
    box_confidence = np.expand_dims(box_confidence, axis=-1)

    box_class_probs = sigmoid(input[..., 5:])

    box_xy = sigmoid(input[..., :2])*2 - 0.5

    col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
    row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)
    col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
    grid = np.concatenate((col, row), axis=-1)
    box_xy += grid
    box_xy *= int(IMG_SIZE/grid_h)

    box_wh = pow(sigmoid(input[..., 2:4])*2, 2)
    box_wh = box_wh * anchors

    box = np.concatenate((box_xy, box_wh), axis=-1)

    return box, box_confidence, box_class_probs


def filter_boxes(boxes, box_confidences, box_class_probs):
    """Filter boxes with box threshold. It's a bit different with origin yolov5 post process!

    # Arguments
        boxes: ndarray, boxes of objects.
        box_confidences: ndarray, confidences of objects.
        box_class_probs: ndarray, class_probs of objects.

    # Returns
        boxes: ndarray, filtered boxes.
        classes: ndarray, classes for boxes.
        scores: ndarray, scores for boxes.
    """
    boxes = boxes.reshape(-1, 4)
    box_confidences = box_confidences.reshape(-1)
    box_class_probs = box_class_probs.reshape(-1, box_class_probs.shape[-1])

    _box_pos = np.where(box_confidences >= OBJ_THRESH)
    boxes = boxes[_box_pos]
    box_confidences = box_confidences[_box_pos]
    box_class_probs = box_class_probs[_box_pos]

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    _class_pos = np.where(class_max_score >= OBJ_THRESH)

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]
    scores = (class_max_score* box_confidences)[_class_pos]

    return boxes, classes, scores


def nms_boxes(boxes, scores):
    """Suppress non-maximal boxes.

    # Arguments
        boxes: ndarray, boxes of objects.
        scores: ndarray, scores of objects.

    # Returns
        keep: ndarray, index of effective boxes.
    """
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep


def yolov5_post_process(input_data):
    masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
               [59, 119], [116, 90], [156, 198], [373, 326]]

    boxes, classes, scores = [], [], []
    for input, mask in zip(input_data, masks):
        b, c, s = process(input, mask, anchors)
        b, c, s = filter_boxes(b, c, s)
        boxes.append(b)
        classes.append(c)
        scores.append(s)

    boxes = np.concatenate(boxes)
    boxes = xywh2xyxy(boxes)
    classes = np.concatenate(classes)
    scores = np.concatenate(scores)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        keep = nms_boxes(b, s)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores
        

def letterbox(im, new_shape=(640, 640), color=(0, 0, 0)):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def rknn_init(number):
    QUANTIZE_ON = True

    OBJ_THRESH = 0.25
    NMS_THRESH = 0.45
    IMG_SIZE = 640

    CLASSES = ('dog', "person", "cat", "tv", "car", "meatballs", "marinara sauce", "tomato soup", "chicken noodle soup",
"french onion soup", "chicken breast","ribs", "pulled pork", "hamburger", "cavity", "red","blue")
    # Create RKNN object
    rknn = RKNN()
    number = int(number)
   
    rknn.load_rknn("/home/gavin/df-wolf2023/rknn_model/RK3588/armor.rknn")
    # Init runtime environment
    print('--> Init runtime environment')
    if number == 1:
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0)
    elif number == 2:
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1)
    elif number == 3:
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0_1_2)
    elif number == -1:
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0)
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_1)
        ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_1)
    else:
        print('Init runtime environment failed!')
        exit(ret)
    print('done')
    return rknn

def camera_init(cap):
    #cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
    #cap.set(cv2.CAP_PROP_BRIGHTNESS, 10)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, 40)#设置曝光
    return cap

def count_distance(P):
    return (armor_distance*464.0)/P

def AimBot_Yolo_Detect(frame, boxes, scores, classes,color):
    armor_information_dict = {}
    armor_list = [];key_name_list = []
    try:
        count = 0
        for box, score, cl in zip(boxes, scores, classes):
            top, left, right, bottom = box
            top = int(top);left = int(left);right = int(right);bottom = int(bottom)
            
            if score >= 0.3 and classes[0] == color:
                count = count + 1
                key_name = "armor" + str(count)
                key_name_list.append(key_name)
                armor_list.append([top, left, right, bottom])  # 将识别物体信息加入armor_list列表
                P = abs(bottom-left)
                #print("P",P)
                distance = count_distance(P)
                #print("distance",distance)
                #print("distance:",abs(bottom-left))
                #cv2.rectangle(frame, (top, left), (right, bottom), (0, 255, 0), 2)
            else:
                distance = 0

        armor_information_dict = zip(key_name_list, armor_list)
        armor_information_dict = dict(armor_information_dict)
    except:
        traceback.print_exc()  # 打印报错信息
        return (frame, armor_information_dict,0)

    #print(armor_information_dict)
    return (frame, armor_information_dict,distance)



if __name__ == '__main__':
    armor_color = "red"

    rknn = rknn_init(3)

    cap = cv2.VideoCapture(0);cap = camera_init(cap)

    armor_information_dict = {}
    armor_list = [];key_name_list = []

    while True:
        ret,frame = cap.read()
        armor_distance_list = []
        if armor_color == "blue":
            # blue armor
            color = 16
        elif armor_color == "red":
            # red armor
            color = 15
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        t1 = time.time()
        outputs = rknn.inference(inputs=[frame])
        #print(time.time()-t1)

        input0_data = outputs[0];input1_data = outputs[1];input2_data = outputs[2]
        input0_data = input0_data.reshape([3, -1]+list(input0_data.shape[-2:]))
        input1_data = input1_data.reshape([3, -1]+list(input1_data.shape[-2:]))
        input2_data = input2_data.reshape([3, -1]+list(input2_data.shape[-2:]))
                
        input_data = list()
        input_data.append(np.transpose(input0_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input1_data, (2, 3, 0, 1)))
        input_data.append(np.transpose(input2_data, (2, 3, 0, 1)))
        boxes, classes, scores = yolov5_post_process(input_data)

        new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        if boxes is not None:
            #print(boxes,scores,classes)
            inference_result = AimBot_Yolo_Detect(new_frame, boxes, scores, classes,color)
            cv2.line(new_frame,(0,320),(640,320),(255,0,0),3)
            cv2.line(new_frame,(320,640),(320,0),(255,0,0),3)
            armor_dict = inference_result[1]
            armor_information_list = list(armor_dict.values())
            i = len(armor_information_list)
            if len(armor_information_list) > 0:
                for i in range(i):
                    xmin = list(armor_dict.values())[i - 1][0]
                    ymin = list(armor_dict.values())[i - 1][1]
                    xmax = list(armor_dict.values())[i - 1][2]
                    ymax = list(armor_dict.values())[i - 1][3]
                    frame = cv2.rectangle(inference_result[0], (xmin, ymax), (xmax, ymin), (0, 255, 0), 2)  # 在原图像中画出方框框记识别目标
                    distance = inference_result[2]
                    #print('d',distance)
                    armor_x_pos = int(xmax-(xmax-xmin)/2);armor_y_pos = int(ymin-(ymin-ymax)/2)
                    distance = math.sqrt(abs(armor_y_pos-320)*abs(armor_y_pos-320) + abs(armor_x_pos-320)*abs(armor_x_pos-320))
                    armor_distance_list.append(distance)

                    cv2.line(new_frame,(320,320),(armor_x_pos,armor_y_pos),(255,0,255),3)
            try:
                #print(armor_distance_list)
                min_armor_distance = min(armor_distance_list)
                #print(armor_distance_list.index(min_armor_distance))
                xmin_test = list(armor_dict.values())[armor_distance_list.index(min_armor_distance)-1][0]
                ymin_test = list(armor_dict.values())[armor_distance_list.index(min_armor_distance)-1][1]
                xmax_test = list(armor_dict.values())[armor_distance_list.index(min_armor_distance)-1][2]
                ymax_test = list(armor_dict.values())[armor_distance_list.index(min_armor_distance)-1][3]
                cv2.rectangle(inference_result[0], (xmin_test, ymax_test), (xmax_test, ymin_test), (255, 0, 255), 3)
                armor_x_pos_test = int(xmax_test-(xmax_test-xmin_test)/2);armor_y_pos_test = int(ymin_test-(ymin_test-ymax_test)/2)
                #print("detect")
                #print(armor_x_pos_test,armor_y_pos_test)
                send_message(True,armor_x_pos_test,armor_y_pos_test,distance)
            except:
                pass
                #print("ERROR")
            #print(inference_result[1])
            cv2.imshow("inference result",frame)
            cv2.waitKey(1)

        cv2.imshow("inference result",new_frame)
        cv2.waitKey(1)

   # cv2.destroyAllWindows()
    #cv2.imwrite("hello.jpg",img_1)
    rknn.release()
