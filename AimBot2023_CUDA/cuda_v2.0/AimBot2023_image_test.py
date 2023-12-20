
from AimBot2023_Main import *


image = cv2.imread("/home/gavin/df-wolf2023/inference_test/image/2.png")#读取测试图片
device, torch_model = AimBot_init("cuda", "/home/gavin/df-wolf2023/yolov5",
"/home/gavin/df-wolf2023/pytorch_model/armor/weights/best.pt")#AimBot2023初始化
inference_result = AimBot_Yolo_Detect(image, "red", 640, 0.3, device, torch_model)
armor_dict = inference_result[1]
armor_information_list = list(armor_dict.values())
i = len(armor_information_list)
if len(armor_information_list) > 0:
    for i in range(i):
        xmin = list(armor_dict.values())[i - 1][0]
        ymin = list(armor_dict.values())[i - 1][1]
        xmax = list(armor_dict.values())[i - 1][2]
        ymax = list(armor_dict.values())[i - 1][3]
        image = cv2.rectangle(inference_result[0], (xmin, ymax), (xmax, ymin), (0, 255, 255), 2)  # 在原图像中画出方框框记识别目标

print(inference_result[1])
cv2.imshow("inference result",image)
cv2.waitKey(0)