import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

import hybrid_a_star as astar

XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interpolate resolution
N_STEER = 20  # number of steer command
VR = 1.0  # robot radius

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost


class Q10Planner:
    def __init__(self):
        self.map_w = 240
        self.map_h = 120

        self.map_obs_x = []
        self.map_obs_y = []

        self.start_pose = [0, 0, 0]
        self.target_pose = [0, 0, 0]

    def load_map(self, input_img, img_w=240, img_h=120):
        map_w, map_h = self.map_w, self.map_h
        map_img = input_img.copy()

        map_img = cv2.resize(map_img, dsize=(map_w, map_h), interpolation=cv2.INTER_LINEAR)
        ret, map_img = cv2.threshold(map_img, 127, 255, 0)

        contours, hierarchy = cv2.findContours(map_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cont in contours:
            self.map_obs_x = np.append(self.map_obs_x, cont[:, 0, 0]).astype(np.float64)
            self.map_obs_y = np.append(self.map_obs_y, cont[:, 0, 1]).astype(np.float64)
        self.map_obs_x = self.map_obs_x.tolist()
        self.map_obs_y = self.map_obs_y.tolist()

    def planning(self, start_pose, target_pose):
        path = astar.hybrid_a_star_planning(
            start_pose, target_pose, self.map_obs_x, self.map_obs_y, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        return path
