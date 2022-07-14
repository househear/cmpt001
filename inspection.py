import config
from tri3d import Tri3d
import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np
import mvsdk
import platform
from plot import SubplotAnimation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
class Camera:
    def __init__(self):
        self.init_camera()

    def init_camera(self):
        DevList = mvsdk.CameraEnumerateDevice()
        nDev = len(DevList)
        if nDev < 1:
            print("No camera was found!")
            return

        for i, DevInfo in enumerate(DevList):
            print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
        i = 0 if nDev == 1 else int(input("Select camera: "))
        DevInfo = DevList[i]
        print(DevInfo)

        # 打开相机
        self.hCamera = 0
        try:
            self.hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException as e:
            print("CameraInit Failed({}): {}".format(e.error_code, e.message))
            return

        # 获取相机特性描述
        cap = mvsdk.CameraGetCapability(self.hCamera)

        # 判断是黑白相机还是彩色相机
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

        # 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

        # 相机模式切换成连续采集
        mvsdk.CameraSetTriggerMode(self.hCamera, 0)

        # 手动曝光，曝光时间30ms
        mvsdk.CameraSetAeState(self.hCamera, 0)
        mvsdk.CameraSetExposureTime(self.hCamera, 60 * 1000)

        # 让SDK内部取图线程开始工作
        mvsdk.CameraPlay(self.hCamera)

        # 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

        # 分配RGB buffer，用来存放ISP输出的图像
        # 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

    def capture_img(self):
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 200)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)

            # windows下取到的图像数据是上下颠倒的，以BMP格式存放。转换成opencv则需要上下翻转成正的
            # linux下直接输出正的，不需要上下翻转
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)

            # 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
            # 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth,
                                   1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            #frame = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
            return frame

        except mvsdk.CameraException as e:
            if e.error_code != mvsdk.CAMERA_STATUS_TIME_OUT:
                print("CameraGetImageBuffer failed({}): {}".format(e.error_code, e.message))

    def close_camera(self):
        # 关闭相机
        mvsdk.CameraUnInit(self.hCamera)

        # 释放帧缓存
        mvsdk.CameraAlignFree(self.pFrameBuffer)


class Inspection:
    def __init__(self, online, offline_image_path):
        self.i = 0
        self.online = online
        self.offline_image_path = offline_image_path
        self.img = None
        self.circles = None
#        self.plot_init()
        self.result= {'thetas': [0, 0.5, 1],
                      'rs': [500, 600, 700]}
        #self.ani = SubplotAnimation(self.result)
        if online:
            self.camera = Camera()

    def plot_circle_as_circle(self):

        # x axis values
        x = [1,2,3,4,5,6]
        # corresponding y axis values
        y = [2,4,1,5,2,6]
        color_index = 0
        for circle in self.circles:
            if len(circle.rs) > 0:

                rs_arr = np.array(circle.rs)
                thetas_arr = np.array(circle.thetas)

                circle_x = np.dot(rs_arr,np.cos(thetas_arr))
                circle_y = np.dot(rs_arr,np.sin(thetas_arr))
                # plotting the points 
                plt.plot(circle_x, circle_y, 
                color=config.color_array[np.mod(color_index, len(config.color_array))], 
                linestyle='dashed', linewidth = 3,
                marker='o', markerfacecolor='blue', markersize=12)
                color_index =  color_index + 1
        
        # setting x and y axis range
        # plt.xlim(0,6.28)
        # plt.ylim(600,1200)
        plt.xlim(-2000,2000)
        plt.ylim(-2000,2000)        
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        
        # giving a title to my graph
        plt.title('Some cool customizations!')
        
        # function to show the plot
        plt.show()

    def plot_circle(self):

        # x axis values
        x = [1,2,3,4,5,6]
        # corresponding y axis values
        y = [2,4,1,5,2,6]
        color_index = 0
        for circle in self.circles:

            # plotting the points 
            plt.plot(circle.thetas, circle.rs, 
            color=config.color_array[np.mod(color_index, len(config.color_array))], 
            linestyle='dashed', linewidth = 3,
                    marker='o', markerfacecolor='blue', markersize=12)
            color_index =  color_index + 1
        
        # setting x and y axis range
        plt.xlim(0,6.28)
        plt.ylim(600,1200)
        
        # naming the x axis
        plt.xlabel('x - axis')
        # naming the y axis
        plt.ylabel('y - axis')
        
        # giving a title to my graph
        plt.title('Some cool customizations!')
        
        # function to show the plot
        plt.show()

    def fetch_image(self):
        if self.online:
            self.img = self.camera.capture_img()
        else:
            self.img = cv2.imread(self.offline_image_path, cv2.IMREAD_COLOR)

    def run(self):
        depth = Tri3d(self.img)
        depth.run(show_ray = True,
                  show_cross_point = True)

        self.circles = depth.parse_circles_v01()
        m = 0
        self.result = depth.cal_3d()

    def plot(self):
        thetas = []
        rs = []
        for theta_r_ray in self.result:
            if len(theta_r_ray['3d_list']) > 0:
                _3d_list = theta_r_ray['3d_list'][0]
                thetas.append(theta_r_ray['theta'])
                rs.append(_3d_list[3])
        #
        # # r_x = np.random.rand(10)
        # # r_y = np.random.rand(10)
        # # update data
        # self.line1.set_data(thetas, rs)
        # # redraw the canvas
        # self.fig.canvas.draw()
        #
        # # convert canvas to image
        # img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8,
        #                     sep='')
        # img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
        #
        # # img is rgb, convert to opencv's default bgr
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with opencv or any operation you like
        #cv2.imshow("plot", img)

        #frame = cv2.resize(self.img, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        cv2.imshow('img', self.img)
        small_frame = cv2.resize(self.img, (512, 512), interpolation=cv2.INTER_LINEAR)
        cv2.imshow('img_small', small_frame)