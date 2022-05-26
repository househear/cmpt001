import cv2
import depthai as dai
import numpy as np

# 定义管道
pipeline = dai.Pipeline()

# 创建左右两个灰度相机流
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# 创建一个将产生深度图的节点（使用视差输出，因为这样更容易可视化深度）
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
left.out.link(depth.left)
right.out.link(depth.right)

# 创建输出流
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)

# 管道已创建，现在将设备连接管道
with dai.Device(pipeline) as device:
    # 开启管道
    device.startPipeline()

    # 输出队列将用于从上面定义的输出中获取视差帧
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    while True:
        in_depth = q.get()  # 阻止呼叫，将等待直到新数据到达
        # 数据最初表示为平面1维数组，需要将其转换为HxW形式
        frame = in_depth.getData().reshape((in_depth.getHeight(), in_depth.getWidth())).astype(np.uint8)
        frame = np.ascontiguousarray(frame)
        # 使用applyColorMap方法给图像添加伪色彩，将应用颜色图以突出显示深度信息
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        # 使用OpenCV的imshow方法显示图像
        cv2.imshow("disparity", frame)

        if cv2.waitKey(1) == ord('q'):
            break