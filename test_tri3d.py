from inspection import Inspection
import cv2
import matplotlib.pyplot as pltp
inspection = Inspection(online = False,
                        offline_image_path = 'images/testing/364.bmp')
while (cv2.waitKey(1) & 0xFF) != ord('q'):
    inspection.fetch_image()
    inspection.run()
    inspection.plot_circle()
    inspection.plot()
    pltp.show()

cv2.waitKey(0)

