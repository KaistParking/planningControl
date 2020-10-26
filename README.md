# parkingControl

# 주행 경로 플래닝 & 경로 트래킹

---

## 1. 설명

- 주차 공간이 표시 된 이미지를 입력으로 넣으면, 경로 생성 후 매 시간마다 핸들을 꺾어야 하는 각도(스티어링)와 입력해 주어야 하는 가속도 반환
- **Planner:** Hybrid A-star
- **Tracker:** Motion Predictive Control(MPC) with speed and steer

---

## 2. 코드 및 사용법

- control_toolbox_q10에서 Controller 클래스를 import해서 사용하시면 됩니다.
- 사용 방법은 아래와 같습니다.

```python
from control_toolbox_q10.q10Controller import Controller

# 컨트롤러 객체 생성 (맵 이미지 입력)
car = Controller(map_img, map_w, map_h)

# 현재 지점에서 목표 지점에 대한 경로 생성
start = [5.0, 25.0, np.deg2rad(-90.0)]
goal = [45.0, 10.0, np.deg2rad(-90.0)]
car.plan_path(start, goal)
car.show_path() # 생성된 경로 시각화

# 제어 입력 받아오기
# 위치 지정해서 업데이트
accel, steer = car.update(pose=current_pose, v=current_velocity) # 위치 지정해서 업데이트
# 따로 위치 지정하지 않고 이전 값을 기준으로 계산할 수도 있음
accel, steer = car.update()
car.show() # 현재 주행상태 시각화

car.check_goal() # 도착지점에 도착한 경우 True 반환
```

- 예시 코드는 아래와 같습니다.

```python
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
goal = [45.0, 10.0, np.deg2rad(-90.0)]

car.plan_path(start, goal)
car.show_path()

plt.gcf().canvas.mpl_connect('key_release_event',
                             lambda event: [exit(0) if event.key == 'escape' else None])
while not car.check_goal():
    car.update()
    car.show()
    plt.pause(0.2)
    plt.cla()
plt.close()
```

## 3. 실험 예시

---

### 3-1) Sample Map

- 아래와 같은 샘플 맵을 만들어 실험해 봤습니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ac82d873-b264-423e-9fbc-d9d88f7f2bcd/map.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/ac82d873-b264-423e-9fbc-d9d88f7f2bcd/map.png)

---

### 3-2) Path Planning

- 빨간색 선이 생성된 경로, 각 화살표는 시작 지점에서의 방향과 도착 지점에서의 방향입니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b838d58-4655-4dcb-9744-5d1ef69eeddb/path.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/4b838d58-4655-4dcb-9744-5d1ef69eeddb/path.png)

---

### 3-3) Path Following(Tracking)

- Linear한 모델을 가정해서 실제로 잘 따라 가는지 테스트 해봤습니다.
- 빨간색으로 표시되는 accel과 steer를 실시간으로 받아 차량이 움직입니다.
- 실제 차량에 구현할 때도 실시간으로 제어 입력을 받을 수 있습니다.

![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/966bdd3a-f9f2-4114-b0e1-d4c695a1c5d2/simulation.gif](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/966bdd3a-f9f2-4114-b0e1-d4c695a1c5d2/simulation.gif)

---

## 4. 알고리즘 설명

### 4-1) Path Planning - Hybrid A*

- A* 알고리즘은 알고리즘 시간에 많이 배우는데, 최단거리 탐색 알고리즘으로 목적지까지의 거리를 가지고 cost function을 만들어 cost가 낮은 순으로 우선해서 경로를 탐색합니다.
- 근데 차량은 non-holonomic 하므로(일정 각도 내에서 밖에 못 움직이므로) 이런 특성을 고려해서 경로를 만들어야 합니다..
- 그래서 탐색을 하기 위한 각도를 cost function에 추가해서 최대 각도 내에서, 최대한 덜 꺾는 각도로 최단경로를 탐색하도록 만든 것이 Hybrid A* 입니다.
- 아래 두 블로그 글 참고하시면 좋을 것 같아요.

    [Explaining the Hybrid A Star pathfinding algorithm for selfdriving cars](https://blog.habrador.com/2015/11/explaining-hybrid-star-pathfinding.html)

    [Application of hybrid A star](https://miracleyoo.tistory.com/21)

### 4-2) Path Tracking - MPC(Model Predictive Control)

- **MPC 알고리즘은 차량의 시스템 모델(어떻게 움직이는지)을 정해 두고, 타겟 주행 경로와의 거리, 조향각 등에 패널티를 부여해서 실제 차량의 행동 방식에 맞게 입력을 제어하는 방식입니다.**
    - 상태 벡터(State vector):
        - x: x-position, y: y-position, v: velocity, φ: yaw angle

            [https://render.githubusercontent.com/render/math?math=z%20%3D%20%5Bx%2C%20y%2C%20v%2C%5Cphi%5D&mode=display](https://render.githubusercontent.com/render/math?math=z%20%3D%20%5Bx%2C%20y%2C%20v%2C%5Cphi%5D&mode=display)

    - 입력 벡터(Input vector):
        - a: acceleration, δ: steering angle

            [https://render.githubusercontent.com/render/math?math=u%20%3D%20%5Ba%2C%20%5Cdelta%5D&mode=display](https://render.githubusercontent.com/render/math?math=u%20%3D%20%5Ba%2C%20%5Cdelta%5D&mode=display)

    - The MPC controller cost function:
        - MPC는 아래 식을 최소화 하는 방향으로 작동합니다. (z_ref는 타겟 경로 상의 점)

            [https://render.githubusercontent.com/render/math?math=min%5C%20Q_f%28z_%7BT%2Cref%7D-z_%7BT%7D%29%5E2%2BQ%5CSigma%28%7Bz_%7Bt%2Cref%7D-z_%7Bt%7D%7D%29%5E2%2BR%5CSigma%7Bu_t%7D%5E2%2BR_d%5CSigma%28%7Bu_%7Bt%2B1%7D-u_%7Bt%7D%7D%29%5E2&mode=display](https://render.githubusercontent.com/render/math?math=min%5C%20Q_f%28z_%7BT%2Cref%7D-z_%7BT%7D%29%5E2%2BQ%5CSigma%28%7Bz_%7Bt%2Cref%7D-z_%7Bt%7D%7D%29%5E2%2BR%5CSigma%7Bu_t%7D%5E2%2BR_d%5CSigma%28%7Bu_%7Bt%2B1%7D-u_%7Bt%7D%7D%29%5E2&mode=display)

        - 시스템은 다음과 같은 선형 모델(Linear vehicle model)을 사용하였습니다.

            [https://render.githubusercontent.com/render/math?math=z_%7Bt%2B1%7D%3DAz_t%2BBu%2BC&mode=display](https://render.githubusercontent.com/render/math?math=z_%7Bt%2B1%7D%3DAz_t%2BBu%2BC&mode=display)

- 좀 더 구체적인 설명은 여기 참고하시면 좋을 것 같습니다 (저도 잘 몰라요 ㅜㅜ)

    [AtsushiSakai/PythonRobotics](https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathTracking/model_predictive_speed_and_steer_control/Model_predictive_speed_and_steering_control.ipynb)
