from rknnlite.api import RKNNLite as RKNN#导入推理库
if __name__ == '__main__':

    # Create RKNN object
    rknn = RKNN()

   
    rknn.load_rknn("/home/gavin/df-wolf2023/rknn_model/RK3588/armor.rknn")
    # Init runtime environment
    print('--> Init runtime environment')
    ret = rknn.init_runtime(core_mask=RKNN.NPU_CORE_0)#设置为单核NPU推理模式
    if ret != 0:#判断是否成功设置
        print('Init runtime environment failed!')
    else:
        print('done')
