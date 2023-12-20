from AimBot2023_Main import *


if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/gavin/df-wolf2023/inference_test/video/test014.mp4")
    device, torch_model = AimBot_init("cuda", "/home/gavin/df-wolf2023/yolov5",
"/home/gavin/df-wolf2023/pytorch_model/armor/weights/best.pt")#AimBot2023初始化
    while True:
        ret,frame = cap.read()
        inference_result = AimBot_Yolo_Detect(frame, "blue",640, 0.3, device, torch_model)
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
        cv2.imshow("inference result",frame)
        cv2.waitKey(1)