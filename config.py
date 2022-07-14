testing_path = 'images/testing/'
color_array = ['b', 'g','r','c','m','y','k']
step_radius = 0.3
step_turning = 1
r_inner = 700
r_outter = 1000
pix_to_physics = 0.083 # mm/pix
f = 12 # lens focus lenght (mm)
ls = {'1': 50,
     '2': 55,
     '3': 60,
     '4': 65,
     '5': 70,
     '6': 75,
      '7': 80,
      '8': 85,
      '9': 90,
      '10': 100,
      '11': 105,
      '12': 110,
      '13': 115,
      '14': 120,
      '15': 125,
      '16': 130,
      '17': 135,
      '18': 140,
      '19':145,
      '20':150,
      '21': 155,
      '22': 160,
      '23': 165,
      '24': 170,
      '25': 175}

jump_r_threshold_in_circle = 10
jump_theta_threshold_in_circle = 0.5
search_template_r_step = 33
search_template_circle_num = 8
search_template_delta_r_error = 20