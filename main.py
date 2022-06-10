from inspection import Inspection
from tri3d import Tri3d
import threading
from plot import SubplotAnimation
import numpy as np
import matplotlib.pyplot as plt

class thread_img_pro(threading.Thread):
    def __init__(self, thread_name, thread_ID, ani):
        threading.Thread.__init__(self)
        self.thread_name = thread_name
        self.thread_ID = thread_ID
        self.ani = ani

        # helper function to execute the threads
    def run(self):
        depth = Tri3d()
        depth.run(show_ray=False,
                  show_cross_point=False)
        depth.cal_3d()
        while True:
            thetas = []
            rs = []
            #depth.cal_3d()

            for theta_r_ray in depth.theta_rs:
                if len(theta_r_ray['3d_list']) > 0 :
                    _3d_list = theta_r_ray['3d_list'][0]
                    thetas.append(theta_r_ray['theta'])
                    rs.append(_3d_list[3])

            r_x = np.random.rand(10)
            r_y = np.random.rand(10)
            self.ani.update_data(thetas[0:10], rs[0:10])

ani = SubplotAnimation()
thread1 = thread_img_pro("GFG", 1000, ani)
thread1.start()
plt.show()