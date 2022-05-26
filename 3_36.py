#!/usr/bin/env python3

import cv2
import numpy as np
import depthai as dai
from time import sleep
import datetime
import argparse
from pathlib import Path

datasetDefault = str((Path(__file__).parent / Path('models/dataset')).resolve().absolute())
parser = argparse.ArgumentParser()
parser.add_argument('-dataset', nargs='?', help="Path to recorded frames", default=datasetDefault)
args = parser.parse_args()

if not Path(datasetDefault).exists():
    import sys
    raise FileNotFoundError(f'Required file/s not found, please run "{sys.executable} install_requirements.py"')


# StereoDepth配置选项。
out_depth = False  # 预期情况下的差异
out_rectified = True   # 输出和显示整流后的流
lrcheck = False   # 更好地处理咬合
extended = True  # 接近最小深度，视差范围加倍
subpixel = False # 更好的精度，可实现更长的距离，分数差异为32级
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7

# 清理一些不兼容的选项
if lrcheck or extended or subpixel:
    median = dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF

print("StereoDepth config options: ")
print("Left-Right check: ", lrcheck)
print("Extended disparity: ", extended)
print("Subpixel: ", subpixel)
print("Median filtering: ", median)

right_intrinsic = [[860.0, 0.0, 640.0], [0.0, 860.0, 360.0], [0.0, 0.0, 1.0]]


def create_stereo_depth_pipeline():
    print("Creating Stereo Depth pipeline: ", end='')

    print("XLINK IN -> STEREO -> XLINK OUT")
    pipeline = dai.Pipeline()

    camLeft = pipeline.createXLinkIn()
    camRight = pipeline.createXLinkIn()
    stereo = pipeline.createStereoDepth()
    xoutLeft = pipeline.createXLinkOut()
    xoutRight = pipeline.createXLinkOut()
    xoutDepth = pipeline.createXLinkOut()
    xoutDisparity = pipeline.createXLinkOut()
    xoutRectifLeft = pipeline.createXLinkOut()
    xoutRectifRight = pipeline.createXLinkOut()

    camLeft.setStreamName('in_left')
    camRight.setStreamName('in_right')

    stereo.setConfidenceThreshold(200)
    stereo.setRectifyEdgeFillColor(0) # 黑色，以更好地看到切口
    stereo.setMedianFilter(median) # 默认KERNEL_7x7
    stereo.setLeftRightCheck(lrcheck)
    stereo.setExtendedDisparity(extended)
    stereo.setSubpixel(subpixel)

    stereo.setEmptyCalibration() # 设置输入帧是否已纠正

    stereo.setInputResolution(1280, 720)

    xoutLeft.setStreamName('left')
    xoutRight.setStreamName('right')
    xoutDepth.setStreamName('depth')
    xoutDisparity.setStreamName('disparity')
    xoutRectifLeft.setStreamName('rectified_left')
    xoutRectifRight.setStreamName('rectified_right')

    camLeft.out.link(stereo.left)
    camRight.out.link(stereo.right)
    stereo.syncedLeft.link(xoutLeft.input)
    stereo.syncedRight.link(xoutRight.input)
    if out_depth:
        stereo.depth.link(xoutDepth.input)
    stereo.disparity.link(xoutDisparity.input)
    if out_rectified:
        stereo.rectifiedLeft.link(xoutRectifLeft.input)
        stereo.rectifiedRight.link(xoutRectifRight.input)

    streams = ['left', 'right']
    if out_rectified:
        streams.extend(['rectified_left', 'rectified_right'])
    streams.extend(['disparity', 'depth'])

    return pipeline, streams


def convert_to_cv2_frame(name, image):
    baseline = 75 #mm
    focal = right_intrinsic[0][0]
    max_disp = 96
    disp_type = np.uint8
    disp_levels = 1
    if (extended):
        max_disp *= 2
    if (subpixel):
        max_disp *= 32
        disp_type = np.uint16
        disp_levels = 32

    data, w, h = image.getData(), image.getWidth(), image.getHeight()
    if name == 'depth':
        # 包含带有（lrcheck或Extended或subpixel）的FP16
        frame = np.array(data).astype(np.uint8).view(np.uint16).reshape((h, w))
    elif name == 'disparity':
        disp = np.array(data).astype(np.uint8).view(disp_type).reshape((h, w))

        # 根据差异计算深度
        with np.errstate(divide='ignore'):
            depth = (disp_levels * baseline * focal / disp).astype(np.uint16)

        if 1: # （可选）扩展视差范围以更好地可视化它
            frame = (disp * 255. / max_disp).astype(np.uint8)

        if 1: # （可选）应用颜色图
            frame = cv2.applyColorMap(frame, cv2.COLORMAP_HOT)

    else: # 单流/单通道
        frame = np.array(data).reshape((h, w)).astype(np.uint8)
        if name.startswith('rectified_'):
            frame = cv2.flip(frame, 1)
        if name == 'rectified_right':
            last_rectif_right = frame
    return frame


pipeline, streams = create_stereo_depth_pipeline()

print("Connecting and starting the pipeline")
# 连接并启动管道
with dai.Device(pipeline) as device:

    inStreams = ['in_right', 'in_left']
    inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
    in_q_list = []
    for s in inStreams:
        q = device.getInputQueue(s)
        in_q_list.append(q)

    # 为每个流创建一个接收队列
    q_list = []
    for s in streams:
        q = device.getOutputQueue(s, 8, blocking=False)
        q_list.append(q)

    # 需要为“立体声”节点中的同步阶段设置输入帧的时间戳 timestamp_ms = 0

    timestamp_ms = 0
    index = 0
    while True:
        # 处理输入流（如果有）
        if in_q_list:
            dataset_size = 2  # 图像对数
            frame_interval_ms = 500
            for i, q in enumerate(in_q_list):
                path = args.dataset + '/' + str(index) + '/' + q.getName() + '.png'
                data = cv2.imread(path, cv2.IMREAD_GRAYSCALE).reshape(720*1280)
                tstamp = datetime.timedelta(seconds = timestamp_ms // 1000,
                                            milliseconds = timestamp_ms % 1000)
                img = dai.ImgFrame()
                img.setData(data)
                img.setTimestamp(tstamp)
                img.setInstanceNum(inStreamsCameraID[i])
                img.setType(dai.ImgFrame.Type.RAW8)
                img.setWidth(1280)
                img.setHeight(720)
                q.send(img)
                if timestamp_ms == 0:  # 发送两次以进行第一次迭代
                    q.send(img)
                print("Sent frame: {:25s}".format(path), 'timestamp_ms:', timestamp_ms)
            timestamp_ms += frame_interval_ms
            index = (index + 1) % dataset_size
            sleep(frame_interval_ms / 1000)
        # 处理输出流
        for q in q_list:
            if q.getName() in ['left', 'right', 'depth']: continue
            frame = convert_to_cv2_frame(q.getName(), q.get())
            cv2.imshow(q.getName(), frame)
        if cv2.waitKey(1) == ord('q'):
            break