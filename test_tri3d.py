from inspection import Inspection
import cv2
import matplotlib.pyplot as pltp
inspection = Inspection(online = True,
                        offline_image_path = 'images/testing/grab.bmp')
while (cv2.waitKey(1) & 0xFF) != ord('q'):
    inspection.fetch_image()
    inspection.cal_3d()
    inspection.plot()
    #plt.show()

cv2.waitKey(0)

