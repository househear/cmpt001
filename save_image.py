#!/usr/bin/env python3

"""
本示例显示了在修剪模式下使用灰度相机并可以移动修剪的可能性。
使用“ WASD”控件移动裁剪窗口，使用“ T”触发自动对焦“ IOKL”。 手动曝光/对焦:
Control:      key[dec/inc]  min..max
exposure time:     I   O      1..33000 [us]
sensitivity iso:   K   L    100..1600
要返回自动控制，请执行以下操作:
'E' - autoexposure
"""


import cv2
import depthai as dai
import os
# 步长 ('W','A','S','D' 控制)
stepSize = 0.02
# 手动曝光/对焦设置步骤
expStep = 500  # us
isoStep = 50

# 开始定义管道
pipeline = dai.Pipeline()

# 定义两个单（灰度）相机
camRight = pipeline.createMonoCamera()
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
camRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
camLeft = pipeline.createMonoCamera()
camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)


depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True) # 校准后的流默认情况下是水平镜像的
depth.setRectifyEdgeFillColor(0)  # 黑色，以更好地看到来自校正的切口（边缘有黑色条纹）
camLeft.out.link(depth.left)
camRight.out.link(depth.right)

# 作用范围
topLeft = dai.Point2f(0.2, 0.2)
bottomRight = dai.Point2f(0.8, 0.8)

manipRight = pipeline.createImageManip()
manipRight.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
manipLeft = pipeline.createImageManip()
manipLeft.initialConfig.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
print(camRight.getResolutionHeight())
print(camRight.getResolutionWidth())
#manipRight.setMaxOutputFrameSize(camRight.getResolutionHeight()*camRight.getResolutionWidth()*3)

# 相机移动配置 (wasd)
configIn = pipeline.createXLinkIn()
configIn.setStreamName('config')
configIn.out.link(manipRight.inputConfig)
configIn.out.link(manipLeft.inputConfig)

# 相机控制 (exp, iso, focus)
controlIn = pipeline.createXLinkIn()
controlIn.setStreamName('control')
controlIn.out.link(camRight.inputControl)
controlIn.out.link(camLeft.inputControl)

# 与USB连结
camRight.out.link(manipRight.inputImage)
camLeft.out.link(manipLeft.inputImage)

# 创建输出流
manipOutRight = pipeline.createXLinkOut()
manipOutRight.setStreamName("right")
camRight.out.link(manipOutRight.input)

manipOutLeft = pipeline.createXLinkOut()
manipOutLeft.setStreamName("left")
camLeft.out.link(manipOutLeft.input)

xoutRectifLeft = pipeline.createXLinkOut()
xoutRectifRight = pipeline.createXLinkOut()
xoutRectifLeft.setStreamName('rectified_left')
xoutRectifRight.setStreamName('rectified_right')

depth.rectifiedLeft.link(xoutRectifLeft.input)
depth.rectifiedRight.link(xoutRectifRight.input)

def clamp(num, v0, v1):
    return max(v0, min(num, v1))

# 连接并启动管道
with dai.Device(pipeline) as device:

    # 输出队列将用于获取灰度帧
    qRight = device.getOutputQueue(manipOutRight.getStreamName(), maxSize=4, blocking=False)
    qLeft = device.getOutputQueue(manipOutLeft.getStreamName(), maxSize=4, blocking=False)
    qRecRight = device.getOutputQueue(xoutRectifRight.getStreamName(), maxSize=4, blocking=False)
    qRecLeft = device.getOutputQueue(xoutRectifLeft.getStreamName(), maxSize=4, blocking=False)

    configQueue = device.getInputQueue(configIn.getStreamName())
    controlQueue = device.getInputQueue(controlIn.getStreamName())

    def displayFrame(name, frame):
        cv2.imshow(name, frame)

    sendCamConfig = False

    # 手动对焦/曝光控制的默认值和限制
    expTime = 20000
    expMin = 1
    expMax = 33000

    sensIso = 800
    sensMin = 100
    sensMax = 1600

    i = 0
    num_saved_imgs = 10
    while i < num_saved_imgs:
        inRight = qRight.get()
        inLeft = qLeft.get()
        frameRight = inRight.getCvFrame()
        frameLeft = inLeft.getCvFrame()

        inRecRight = qRecRight.get()
        inRecLeft = qRecLeft.get()
        frameRecRight = inRecRight.getCvFrame()
        frameRecLeft = inRecLeft.getCvFrame()
        displayFrame("right", frameRight)
        displayFrame("left", frameLeft)
        prefix = 'images/' + str(i)
        if not os.path.exists(prefix):
            os.makedirs(prefix)
        cv2.imwrite(prefix + '/right.png', frameRight)
        cv2.imwrite(prefix + '/left.png', frameLeft)
        cv2.imwrite(prefix + '/right_rect.png', frameRecRight)
        cv2.imwrite(prefix + '/left_rect.png', frameRecLeft)

        i += 1

        # 更新画面
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            ctrl = dai.CameraControl()
            ctrl.setCaptureStill(True)
            controlQueue.send(ctrl)
        elif key == ord('e'):
            print("Autoexposure enable")
            ctrl = dai.CameraControl()
            ctrl.setAutoExposureEnable()
            controlQueue.send(ctrl)
        elif key in [ord('i'), ord('o'), ord('k'), ord('l')]:
            if key == ord('i'): expTime -= expStep
            if key == ord('o'): expTime += expStep
            if key == ord('k'): sensIso -= isoStep
            if key == ord('l'): sensIso += isoStep
            expTime = clamp(expTime, expMin, expMax)
            sensIso = clamp(sensIso, sensMin, sensMax)
            print("Setting manual exposure, time:", expTime, "iso:", sensIso)
            ctrl = dai.CameraControl()
            ctrl.setManualExposure(expTime, sensIso)
            controlQueue.send(ctrl)
        elif key == ord('w'):
            if topLeft.y - stepSize >= 0:
                topLeft.y -= stepSize
                bottomRight.y -= stepSize
                sendCamConfig = True
        elif key == ord('a'):
            if topLeft.x - stepSize >= 0:
                topLeft.x -= stepSize
                bottomRight.x -= stepSize
                sendCamConfig = True
        elif key == ord('s'):
            if bottomRight.y + stepSize <= 1:
                topLeft.y += stepSize
                bottomRight.y += stepSize
                sendCamConfig = True
        elif key == ord('d'):
            if bottomRight.x + stepSize <= 1:
                topLeft.x += stepSize
                bottomRight.x += stepSize
                sendCamConfig = True
        if sendCamConfig:
            cfg = dai.ImageManipConfig()
            cfg.setCropRect(topLeft.x, topLeft.y, bottomRight.x, bottomRight.y)
            configQueue.send(cfg)
            sendCamConfig = False