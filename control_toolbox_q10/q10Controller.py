import warnings

warnings.simplefilter("ignore")

import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
import math

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/planning")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/following")

try:
    # planner
    from hybrid_a_star import *
    # Follower
    from mpc.model_predictive_speed_and_steer_control import *
except:
    # planner
    from control_toolbox_q10.planning.hybrid_a_star import *
    # Follower
    from control_toolbox_q10.following.mpc.model_predictive_speed_and_steer_control import *

'''
Properties (Planner)
'''
XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.5  # [m] path interpolate resolution
N_STEER = 20  # number of steer command
VR = 1.0  # robot radius

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost

'''
Properties (Follower)
'''
NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 5  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param
TARGET_SPEED = 10.0 / 3.6  # [m/s] target speed
# N_IND_SEARCH = 10  # Search index number
N_IND_SEARCH = 30  # Search index number
DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]
MAX_STEER = np.deg2rad(45.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(30.0)  # maximum steering speed [rad/s]
MAX_SPEED = 55.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = -20.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 1.0  # maximum accel [m/ss]

# dl = 1.0  # course tick
dl = 0.5  # course tick


def update_state(state, a, delta, dt):
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER
    state.x = state.x + state.v * math.cos(state.yaw) * dt
    state.y = state.y + state.v * math.sin(state.yaw) * dt
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * dt
    state.v = state.v + a * dt
    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED
    return state


def check_goal(state, goal, tind, nind):
    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)
    isgoal = (d <= GOAL_DIS)
    if abs(tind - nind) >= 5:
        isgoal = False
    isstop = (abs(state.v) <= STOP_SPEED)
    if isgoal and isstop:
        return True
    return False


def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="g", ec="g"):
    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)


class Controller:
    def __init__(self, map_image, map_w, map_h):
        self.map_origin = map_image
        self.map_img = None
        self.map_w = map_w
        self.map_h = map_h
        self.map_obs_x = []
        self.map_obs_y = []

        self.plan_x = []
        self.plan_y = []
        self.plan_yaw = []

        self.speed_profile = []
        self.initial_state = []
        self.goal = []

        self.x = []  # pose_x
        self.y = []  # pose_y
        self.yaw = []  # pose_yaw
        self.v = []  # velocity
        self.times = []  # time
        self.d = []  # steer
        self.a = []  # accel

        self.state = None
        self.target_ind = None
        self.odelta = None
        self.oa = None

        self.update_map(map_image, map_w, map_h)

        self.stuck = 0

    def update_map(self, map_image, map_w, map_h):
        self.map_origin = map_image
        self.map_w = map_w
        self.map_h = map_h

        # resize map
        map_img = cv2.resize(map_image, dsize=(map_w, map_h), interpolation=cv2.INTER_LINEAR)
        ret, map_img = cv2.threshold(map_img, 127, 255, 0)
        # map contour searching
        contours, hierarchy = cv2.findContours(map_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for cont in contours:
            self.map_obs_x = np.append(self.map_obs_x, cont[:, 0, 0]).astype(np.float64)
            self.map_obs_y = np.append(self.map_obs_y, cont[:, 0, 1]).astype(np.float64)
        self.map_img = map_img
        self.map_obs_x = self.map_obs_x.tolist()
        self.map_obs_y = self.map_obs_y.tolist()

    def plan_path(self, start, target):
        path = hybrid_a_star_planning(
            start, target, self.map_obs_x, self.map_obs_y, XY_GRID_RESOLUTION, YAW_GRID_RESOLUTION)
        self.plan_x = path.x_list
        self.plan_y = path.y_list
        self.plan_yaw = path.yaw_list
        self.speed_profile = calc_speed_profile(self.plan_x, self.plan_y, self.plan_yaw, TARGET_SPEED)

        self.initial_state = State(x=self.plan_x[0], y=self.plan_y[0], yaw=self.plan_yaw[0], v=0.0)
        self.goal = [self.plan_x[-1], self.plan_y[-1], self.plan_yaw[-1]]

        # initial pose
        self.state = self.initial_state
        if self.state.yaw - self.plan_yaw[0] >= math.pi:
            self.state.yaw -= math.pi * 2.0
        elif self.state.yaw - self.plan_yaw[0] <= -math.pi:
            self.state.yaw += math.pi * 2.0
        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)
        self.times.append(time.time())
        self.d.append(0.0)
        self.a.append(0.0)

        # initial update
        self.target_ind, _ = calc_nearest_index(self.state, self.plan_x, self.plan_y, self.plan_yaw, 0)
        self.odelta = None
        self.oa = None
        self.plan_yaw = smooth_yaw(self.plan_yaw)

        return path

    def update(self, pose=None, v=None):
        t = time.time()  # time

        # update present pose
        if pose is None:
            if v is None:
                self.state = update_state(self.state, self.a[-1], self.d[-1], t - self.times[-1])
            else:
                self.state.v = v
                self.state = update_state(self.state, self.a[-1], self.d[-1], t - self.times[-1])
        else:
            self.state.x = pose[0]
            self.state.y = pose[1]
            self.state.yaw = pose[2]
            if v is not None:
                self.state.v = v

        x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state

        # stuck situation...
        if sum(np.abs(self.a[-5:])) + sum(np.abs(self.v[-5:])) < 0.1:
            self.stuck = 30

        if self.stuck:
            n = 20
            if self.target_ind + n >= len(self.plan_x):
                n = len(self.plan_x) - self.target_ind - 1
            for _ in range(n):
                # calculate control output
                xref, self.target_ind, dref = calc_ref_trajectory(
                    self.state, self.plan_x, self.plan_y, self.plan_yaw, [], self.speed_profile, dl, self.target_ind + 1)
                self.oa, self.odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, self.oa,
                                                                                      self.odelta)
                if self.odelta is not None and abs(self.oa[0]) > 0.2:
                    self.stuck -= 1
                    break
        else:
            # calculate control output
            xref, self.target_ind, dref = calc_ref_trajectory(
                self.state, self.plan_x, self.plan_y, self.plan_yaw, [], self.speed_profile, dl, self.target_ind)
            x0 = [self.state.x, self.state.y, self.state.v, self.state.yaw]  # current state
            self.oa, self.odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, self.oa, self.odelta)

        # save states
        if self.odelta is not None:
            di, ai = self.odelta[0], self.oa[0]

        self.x.append(self.state.x)
        self.y.append(self.state.y)
        self.yaw.append(self.state.yaw)
        self.v.append(self.state.v)
        self.times.append(t)
        self.d.append(di)
        self.a.append(ai)

        return self.a[-1], self.d[-1]

    def check_goal(self):
        return check_goal(self.state, self.goal, self.target_ind, len(self.plan_x))

    def show_path(self):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [plt.close() if event.key == 'escape' else None])

        plt.title("Planned Driving Path")

        plt.imshow(self.map_img, cmap='gray')
        plot_arrow(self.initial_state.x, self.initial_state.y, self.initial_state.yaw,
                   length=2.0, width=2, fc="r", ec="r")
        plot_arrow(self.goal[0], self.goal[1], self.goal[2], length=2.0, width=2, fc="b", ec="b")
        plt.plot(self.plan_x, self.plan_y, "-r", label="course")
        plot_car(self.state.x, self.state.y, self.state.yaw, steer=self.d[-1])

        plt.axis("equal")
        plt.grid(False)
        plt.show()

    def show(self):
        # plt.cla()
        # for stopping simulation with the esc key.
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #                              lambda event: [exit(0) if event.key == 'escape' else None])

        # plt.title("Time[s]:" + str(round(self.times[-1] - self.times[0], 2))
        #           + ", speed[km/h]:" + str(round(self.v[-1] * 3.6, 2)))

        plt.title("Driving Simulation")

        plt.imshow(self.map_img, cmap='gray')
        # plot_arrow(self.initial_state.x, self.initial_state.y, self.initial_state.yaw,
        #            length=2.0, width=2, fc="r", ec="r")
        plot_arrow(self.goal[0], self.goal[1], self.goal[2], length=2.0, width=2, fc="b", ec="b")
        plt.plot(self.plan_x, self.plan_y, "-r", label="course")
        plt.plot(self.x, self.y, "ob", label="trajectory")
        plt.plot(self.plan_x[self.target_ind], self.plan_y[self.target_ind], "xg", label="target")
        plot_car(self.state.x, self.state.y, self.state.yaw, steer=self.d[-1])

        plt.text(2, -5,
                 "Time[s]:" + str(round(self.times[-1] - self.times[0], 2))
                 + ", speed[km/h]:" + str(round(self.v[-1] * 3.6, 2))
                 )

        plt.text(2, -2,
                 'Accel: {}m/s^2, Steer: {} degree'.format(round(self.a[-1], 2), round(np.rad2deg(self.d[-1]), 2)),
                 color='r')

        plt.axis("equal")
        plt.grid(False)
        # plt.pause(0.0001)
