from disparity import Disparity
import matplotlib.pyplot as plt
import cv2
depth = Disparity()
depth.run()
depth.cal_disparity()
depth.plot()
plt.show()
cv2.waitKey(0)