from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np

# 首先获取模型
mobilenet_path = str((Path(__file__).parent / Path('models/mobilenet.blob')).resolve().absolute())
if len(sys.argv) > 1:
    mobilenet_path = sys.argv[1]


# 开始定义管道
pipeline = dai.Pipeline()

# 创建左右灰度相机流
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# 创建一个将产生深度图的节点（使用视差输出，因为这样更容易可视化深度）
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True) # 校准后的流默认情况下是水平镜像的
depth.setRectifyEdgeFillColor(0)  # 黑色，以更好地看到来自校正的切口（边缘有黑色条纹）
left.out.link(depth.left)
right.out.link(depth.right)

# 创建一个节点以将灰度图像转换为nn可接受的形式
manip = pipeline.createImageManip()
manip.initialConfig.setResize(300, 300)
# NN模型需要BGR输入。默认情况下，ImageManip输出类型将与输入相同（在这种情况下为灰色）
manip.initialConfig.setFrameType(dai.RawImgFrame.Type.BGR888p)
depth.rectifiedLeft.link(manip.inputImage)

# 定义一个将基于源帧进行预测的神经网络
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(mobilenet_path)
manip.out.link(detection_nn.input)

# 创建输出
xout_depth = pipeline.createXLinkOut()
xout_depth.setStreamName("depth")
depth.disparity.link(xout_depth.input)

xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")
manip.out.link(xout_right.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

# MobilenetSSD标签文本
texts = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
         "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# 管道已定义，现在设备已连接到管道
with dai.Device(pipeline) as device:
    # 启动管道
    device.startPipeline()

    # 输出队列将用于从上面定义的输出中获取灰度深度帧和nn数据
    q_right = device.getOutputQueue("right", maxSize=4, blocking=False)
    q_depth = device.getOutputQueue("depth", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue("nn", maxSize=4, blocking=False)

    frame_right = None
    frame_depth = None
    bboxes = []
    labels = []
    confidences = []


    # nn数据（作为边界框的位置）在<0..1>范围内-需要使用图像的width/height对其进行归一化
    def frame_norm(frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)


    while True:
        # 使用tryGet方法（非阻塞）而不是get方法来获取（阻塞），这将返回可用数据，否则返回None
        in_right = q_right.tryGet()
        in_nn = q_nn.tryGet()
        in_depth = q_depth.tryGet()

        if in_right is not None:
            # 如果灰度帧数据可用，则将一维数据转换为HxWxC帧
            shape = (3, in_right.getHeight(), in_right.getWidth())
            frame_right = in_right.getData().reshape(shape).transpose(1, 2, 0).astype(np.uint8)
            frame_right = np.ascontiguousarray(frame_right)

        if in_nn is not None:
            # 检测结果有7个数，最后一次检测后跟着-1位数，以后填充0
            bboxes = np.array(in_nn.getFirstLayerFp16())
            # 仅取-1位之前的结果
            bboxes = bboxes[:np.where(bboxes == -1)[0][0]]
            # 将一维数组转换为Nx7矩阵
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            # 筛选出置信度小于定义阈值的结果
            bboxes = bboxes[bboxes[:, 2] > 0.5]
            # 剪切bbox和标签
            labels = bboxes[:, 1].astype(int)
            confidences = bboxes[:, 2]
            bboxes = bboxes[:, 3:7]

        if in_depth is not None:
            # 数据最初表示为平面1维数组，需要将其转换为HxW形式
            frame_depth = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
            frame_depth = np.ascontiguousarray(frame_depth)
            # 使用OpenCV的applyColorMap方法给图像添加伪颜色，将应用颜色图以突出显示深度信息
            frame_depth = cv2.applyColorMap(frame_depth, cv2.COLORMAP_JET)

        if frame_right is not None:
            # 如果图像不为空，请在其上绘制边框并显示图像
            for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                bbox = frame_norm(frame_right, raw_bbox)
                cv2.rectangle(frame_right, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame_right, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.putText(frame_depth, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.imshow("right", frame_right)

        if frame_depth is not None:
            for raw_bbox, label, conf in zip(bboxes, labels, confidences):
                bbox = frame_norm(frame_depth, raw_bbox)
                cv2.rectangle(frame_depth, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
                cv2.putText(frame_depth, texts[label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
                cv2.putText(frame_depth, f"{int(conf * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255))
            cv2.imshow("depth", frame_depth)

        if cv2.waitKey(1) == ord('q'):
            break