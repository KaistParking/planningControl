import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from control_toolbox_q10.q10Controller import Controller

map_w, map_h = 60, 30

map_img = cv2.imread('maps/map_parking.png', cv2.IMREAD_GRAYSCALE)
print(map_img.shape)

car = Controller(map_img, map_w, map_h)

start = [5.0, 25.0, np.deg2rad(-90.0)]
goal = [45.0, 10.0, np.deg2rad(90.0)]

car.plan_path(start, goal)
car.show_path()

plt.gcf().canvas.mpl_connect('key_release_event',
                             lambda event: [exit(0) if event.key == 'escape' else None])
while not car.check_goal():
    car.update()
    car.show()
    plt.pause(0.2)
    plt.clf()
plt.close()

