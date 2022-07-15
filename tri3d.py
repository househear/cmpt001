
import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
import config
import bisect 
from statistics import mean
def insert(list, n):
    bisect.insort(list, n) 
    index = list.index(n)
    return list, index


def weighted_average(dataframe, value, weight):
	val = dataframe[value]
	wt = dataframe[weight]
	return (val * wt).sum() / wt.sum()

half_window = 20

class Pulse: #contain

	def __init__(self, _name):
		self.name = _name
		self.dataframe_r = []
		self.dataframe_grayscale = []
		self.central = None

	def add_point(self, r, grayscale):
		self.dataframe_r.append(r)
		self.dataframe_grayscale.append(grayscale)

	def weighted_average(self):
		val = np.array(self.dataframe_r)
		wt = np.array(self.dataframe_grayscale)
		self.central = (val * wt).sum() / wt.sum()
		return self.central

class Ray: #contain
	# Pulse_trigger: trigger a pulse when a value greater then (max_light - half_window)
	def __init__(self, _img, _centroid_x, _centroid_y, _theta):
		self.pulse_index = 0
		self.centroid = {'x': _centroid_x,
		'y': _centroid_y}
		self.theta = _theta
		self.parse_info()
		self.fetching_status = False
		self.pulses = []
		self.img = _img
		self.rs = {'theta': _theta,
				   'rs_list': [],
				   '3d_list': []}

	def parse_info(self):
		self.max_light= 100
		self.half_window = 40

	def scan(self):
		for r in np.arange(config.r_inner, config.r_outter, config.step_radius):
			cur_grayscale = self.img[round(self.centroid['y'] + r*math.sin(self.theta))][round(self.centroid['x'] + r*math.cos(self.theta))][0]
			if (self.fetching_status == False) and (cur_grayscale > (self.max_light - self.half_window)): # create a pulse
				self.fetch_pulse()

			if (self.fetching_status == True) and (cur_grayscale < (self.max_light - self.half_window)): # create a pulse
				self.rs['rs_list'].append(self.end_pulse()) # add pluse r

			if self.fetching_status:
				self.pulse.add_point(r, cur_grayscale)
			else:
				pass

	def fetch_pulse(self):
		self.pulse_index += 1
		self.pulse = Pulse(str(self.pulse_index))
		self.fetching_status = True

	def end_pulse(self):
		pulse_r = self.pulse.weighted_average()
		self.pulses.append(self.pulse)
		self.pulse = []
		self.fetching_status = False
		return pulse_r

	def plot(self):
		cv2.line(self.img , (round(self.centroid['x'] + config.r_inner * math.cos(self.theta)), round(self.centroid['y'] + config.r_inner * math.sin(self.theta))),
				 (round(self.centroid['x'] + config.r_outter * math.cos(self.theta)), round(self.centroid['y'] + config.r_outter * math.sin(self.theta))),
				 (0, 255, 0))
		cv2.circle(self.img, (self.centroid['x'], self.centroid['y']), 50, (0, 255, 0), 2)

		# plt.xlabel('x - axis')
		# # naming the y axis
		# plt.ylabel('y - axis')
		#
		# # giving a title to my graph
		# plt.title('My first graph!')
		# for pulse in self.pulses:
		#
		# 	plt.plot(pulse.dataframe_r, pulse.dataframe_grayscale, color='green', linestyle='dashed', linewidth=3,
		# 			 marker='o', markerfacecolor='blue', markersize=12)
		#
		# 	cv2.circle(self.img, (round(self.centroid['x'] + pulse.central * math.cos(self.theta)),
		# 	round(self.centroid['y'] + pulse.central * math.sin(self.theta))), 5, (255, 255, 0), 2)

		# function to show the plot


# class Scan: #contain
# 	def __init__(self, image, _centroid_x, _centroid_y):
# 		self.centroid = {'x': _centroid_x,
# 		'y': _centroid_y}
# 		self.img = image
# 		self.theta_rs = []

# 	def run(self, show_ray):
# 		for theta in np.arange(0, 360, config.step_turning):
# 			ray = Ray(self.img, self.centroid['x'], self.centroid['y'], theta * 3.14 / 180)
# 			ray.scan()
# 			self.theta_rs.append(ray.rs)
# 			if show_ray:
# 				ray.plot()


class Circle_v01:
	def __init__(self, index):
		self.index = index
		self.thetas = []
		self.rs = []
		self.pc_xs = []
		self.pc_ys = []
		self.pc_zs = []


	def add_point(self, _theta, _r):
		self.thetas.append(_theta)
		self.rs.append(_r)
	
	def plot(self,ax,_color):

		# Creating plot
		ax.scatter3D(self.pc_xs, self.pc_ys, self.pc_zs, color = _color)

class Circle:
	def __init__(self, _init_theta, _init_r):
		self.thetas = [_init_theta]
		self.rs = [_init_r]


	def add_point(self, _theta, _r):
		self.thetas.append(_theta)
		self.rs.append(_r)

class Tri3d:
	def __init__(self, img):
		self.img = img
		self.centroid = {
			'x': 1024,
			'y': 1024
		}

		self.theta_rs = []
		self.circles = []

	def plot_cross_point(self):
		for rs_along_ray in self.theta_rs:
			for r in rs_along_ray['rs_list']:
				cv2.circle(self.img, (round(self.centroid['x'] + r * math.cos(rs_along_ray['theta'])),
									  round(self.centroid['y'] + r * math.sin(rs_along_ray['theta']))), 0, (255, 0, 0), 1)
		#cv2.imshow('disparity', self.scan.img)
		#cv2.waitKey(0)

	def parse_circles_v01(self):
		active_circle_r = []
		active_circle_theta = []
		#search the min r in the first 5 rays
		rays_r_5 = []
		for rs_along_ray in self.theta_rs[0:5]:
			rays_r_5 = rays_r_5 + rs_along_ray['rs_list']
		r_min = min(rays_r_5)

		search_template_steps = range(0, config.search_template_circle_num*
		config.search_template_r_step, config.search_template_r_step)

		search_template_rs = r_min + search_template_steps

		#init circles
		for cir_index in range(0, config.search_template_circle_num):
			self.circles.append(Circle_v01(cir_index))


		search_template_rs_live = search_template_rs
		
		for rs_along_ray in self.theta_rs:
			search_template_r_offsets = []
			# if rs_along_ray['theta'] > 2.6:
			# 	aa = 0

			if rs_along_ray['rs_list']:
				index = 0
				search_template_rs_live_new = search_template_rs_live * 0
				for template_r in search_template_rs_live:
					if min(abs(np.array(rs_along_ray['rs_list']) - template_r)) < config.search_template_delta_r_error:
						r = rs_along_ray['rs_list'][np.argmin(abs(np.array(rs_along_ray['rs_list']) - template_r), axis=0)]
						self.circles[index].add_point(rs_along_ray['theta'], r)
						search_template_rs_live_new[index] = r
#						search_template_steps[index] = r - template_r + search_template_steps[index]
						#search_template_r_offset = r - template_r
						search_template_r_offsets.append(r - template_r)
					index = index + 1
				# move the whole search template rs
				search_template_rs_live += mean(search_template_r_offsets)
				# fill the empty in search_template_rs_live_new using the search_template_rs_live
				i = 0
				for ele in search_template_rs_live_new:
					if ele == 0:
						search_template_rs_live_new[i] = search_template_rs_live[i]

					#self.circles[i].add_point(rs_along_ray['theta'], search_template_rs_live_new)
					i += 1
				
				# update search_template_rs_live by search_template_rs_live_new
				search_template_rs_live = search_template_rs_live_new


				# for r in rs_along_ray['rs_list']:
				# 	if min(abs(np.array(search_template_rs_live) - r)) < config.search_template_delta_r_error:
				# 		self.circles[np.argmin(abs(np.array(search_template_rs_live) - r), axis=0)].add_point(rs_along_ray['theta'], r)
				# 		search_template_r_offset = r - search_template_rs_live[np.argmin(abs(np.array(search_template_rs_live) - r), axis=0)] #update r local template
				# 		search_template_r_offsets.append(search_template_r_offset)
				# if search_template_r_offsets:
				# 	search_template_rs_live = search_template_rs_live + mean(search_template_r_offsets)

		return self.circles
		

	def parse_circles(self):
		
		active_circle_r = []
		active_circle_theta = []
		for rs_along_ray in self.theta_rs:
			if rs_along_ray['rs_list']:
				for r in rs_along_ray['rs_list']:
					#search r in active circle r to see if it exists, if yes, 
					# add it to the corresponding circle, if not, create a circle
					if active_circle_r and min(abs(np.array(active_circle_r) - r)) < config.jump_r_threshold_in_circle and abs(active_circle_theta[np.argmin(abs(np.array(active_circle_r) - r), axis=0)] - rs_along_ray['theta']) < config.jump_theta_threshold_in_circle:  #add r to this circle
						self.circles[np.argmin(abs(np.array(active_circle_r) - r), axis=0)].add_point(rs_along_ray['theta'], r) #add r to this circle
						active_circle_r[np.argmin(abs(np.array(active_circle_r) - r), axis=0)] = r #update active r
						active_circle_theta[np.argmin(abs(np.array(active_circle_r) - r), axis=0)] = rs_along_ray['theta'] #update active theta
					else:
						active_circle_r, insert_index = insert(active_circle_r, r) # add a r to acitive r
						active_circle_theta, insert_index_0 = insert(active_circle_theta, rs_along_ray['theta']) # add a r to acitive r
						self.circles.insert(insert_index, Circle(rs_along_ray['theta'], r))#create a new circle
			
		return self.circles



	def cal_3d_v01(self):

		for circle in self.circles:
			l = config.L[circle.index]
			f = config.F[circle.index]
			rs_arr = np.array(circle.rs) * config.pix_to_physics
			thetas_arr = np.array(circle.thetas)
			circle.pc_xs = (l/f) * rs_arr * np.cos(thetas_arr)
			circle.pc_ys = l
			circle.pc_zs = (l/f) * rs_arr * np.sin(thetas_arr)

	def cal_3d(self):
		theta_i = 0

		for rs_along_ray in self.theta_rs:
			theta_j = 0
			l_index = 0
			for r in rs_along_ray['rs_list']:
				l_index += 1
				r = r * config.pix_to_physics
				fea_x = r * math.cos(rs_along_ray['theta']) * config.ls[str(l_index)] / config.f
				if l_index > 24:
					l_index = 24
				fea_y = config.ls[str(l_index)]
				fea_z = r * math.sin(rs_along_ray['theta']) * config.ls[str(l_index)] / config.f
				fea_r = r

				rs_along_ray['3d_list'].append([fea_x, fea_y, fea_z, fea_r])

				theta_j += 1
			theta_i += 1
		return self.theta_rs

	def run(self, show_ray, show_cross_point, show_theta_r_circles, show_3d_circles):
		# scan rays
		for theta in np.arange(0, 360, config.step_turning):
			ray = Ray(self.img, self.centroid['x'], self.centroid['y'], theta * 3.14 / 180)
			ray.scan()
			self.theta_rs.append(ray.rs)
			if show_ray:
				ray.plot()
		if show_cross_point:
			self.plot_cross_point()
		
		self.parse_circles_v01()
		if show_theta_r_circles:
			self.plot_theta_r_circles()

		self.cal_3d_v01()
		if show_3d_circles:
			self.plot_3d_circles()

	def plot_3d_circles(self):
				# Creating figure
		fig = plt.figure(figsize = (10, 7))
		ax = plt.axes(projection ="3d")
		color_index = 0
		for circle in self.circles:
			color=config.color_array[np.mod(color_index, len(config.color_array))]
			circle.plot(ax, color)
			color_index =  color_index + 1
		plt.title("simple 3D scatter plot")
		
		# show plot
		plt.show()


	def plot_theta_r_circles(self):
		fig = plt.figure(figsize = (10, 7))
		color_index = 0
		
		for circle in self.circles:

            # plotting the points 
			plt.plot(circle.thetas, circle.rs, 
			color=config.color_array[np.mod(color_index, len(config.color_array))], 
			linestyle='dashed', linewidth = 2,
					marker='o', markerfacecolor='blue', markersize = 5)
			color_index =  color_index + 1
        
		# setting x and y axis range
		plt.xlim(0,6.28)
		plt.ylim(600,1000)

		# naming the x axis
		plt.xlabel('x - axis')
		# naming the y axis
		plt.ylabel('y - axis')

		# giving a title to my graph
		plt.title('r vs theta')

		# function to show the plot
		plt.show()
		#cv2.imshow('mono_', self.scan.img)

# Press the green button in the gutter to run the script.
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
