import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def get_cartesian(s, d, mapx, mapy, maps):
    """
    Convert Frenet coordinates (s, d) to Cartesian coordinates (x, y)
    """
    # Find the segment with the closest s value
    prev_wp = np.max(np.where(maps <= s)[0])

    # Calculate the proportion along the segment
    seg_s = (s - maps[prev_wp])
    seg_x = mapx[prev_wp+1] - mapx[prev_wp]
    seg_y = mapy[prev_wp+1] - mapy[prev_wp]

    # Normalize segment vector
    seg_norm = np.sqrt(seg_x**2 + seg_y**2)
    seg_x /= seg_norm
    seg_y /= seg_norm

    # Compute the x, y coordinates
    x = mapx[prev_wp] + seg_s * seg_x + d * -seg_y
    y = mapy[prev_wp] + seg_s * seg_y + d * seg_x

    return x, y 

def get_frenet(x, y, mapx, mapy, maps):
    """
    Convert Cartesian coordinates (x, y) to Frenet coordinates (s, d)
    """
    # Calculate the closest waypoint index to the current x, y position
    closest_wp = np.argmin(np.sqrt((mapx - x)**2 + (mapy - y)**2))

    # Compute the vector from the closest waypoint to the x, y position
    dx = x - mapx[closest_wp]
    dy = y - mapy[closest_wp]

    # Compute the vector from the closest waypoint to the next waypoint
    next_wp = (closest_wp + 1) % len(mapx)
    next_dx = mapx[next_wp] - mapx[closest_wp]
    next_dy = mapy[next_wp] - mapy[closest_wp]

    # Normalize next segment vector
    seg_norm = np.sqrt(next_dx**2 + next_dy**2)
    next_dx /= seg_norm
    next_dy /= seg_norm

    # Compute the projection of the x, y vector onto the next segment vector
    proj = dx * next_dx + dy * next_dy

    # Frenet d coordinate
    d = np.sqrt(dx**2 + dy**2 - proj**2)

    # Compute the s coordinate by adding the distance along the road
    # to the distance of the projection
    s = maps[closest_wp] + proj

    return s,d
    
    
def moving_average(data, window_size):
    moving_averages = []
    window_sum = 0
    
    # 처음 window_size만큼의 요소에 대한 합을 구함
    for i in range(window_size):
        window_sum += data[i]
        moving_averages.append(window_sum / (i + 1))
    
    # 이후 요소부터는 이동평균을 계산하여 리스트에 추가
    for i in range(window_size, len(data)):
        window_sum += data[i] - data[i - window_size]
        moving_averages.append(window_sum / window_size)
    
    return moving_averages

def check_intersection(p1, p2, p3, p4):
    """ 
    Check if line segments (p1, p2) and (p3, p4) intersect. 
    Returns True if they intersect.
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

def segment_intersection(p1, p2, p3, p4):
    """ 
    Find the intersection point of line segments (p1, p2) and (p3, p4) 
    if they intersect.
    """
    det = lambda a, b: a[0] * b[1] - a[1] * b[0]
    xdiff = (p1[0] - p2[0], p3[0] - p4[0])
    ydiff = (p1[1] - p2[1], p3[1] - p4[1])
    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(p1, p2), det(p3, p4))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return (x, y)

def find_intersections(points1, points2):
    """
    Check for intersections between two lists of points that define two lines.
    Returns a list of intersection points.
    """
    intersections = []
    for i in range(len(points1) - 1):
        for j in range(len(points2) - 1):
            if check_intersection(points1[i], points1[i+1], points2[j], points2[j+1]):
                intersect = segment_intersection(points1[i], points1[i+1], points2[j], points2[j+1])
                if intersect:
                    intersections.append((intersect, (i,j)))
                    
    return intersections

def find_intersections_with_indices(points1, points2):
    """
    Check for intersections between two lists of points that define two lines.
    Returns a list of tuples with intersection points and the indices of the segments.
    """
    intersections = []
    for i in range(len(points1) - 1):
        for j in range(len(points2) - 1):
            if check_intersection(points1[i], points1[i+1], points2[j], points2[j+1]):
                intersect = segment_intersection(points1[i], points1[i+1], points2[j], points2[j+1])
                if intersect:
                    intersections.append((intersect, (i, j)))
                    
    return intersections

import numpy as np

def create_local_path(sensor_info):
    """
    Processes raw sensor data to create a structured local path.
    
    Parameters:
    - sensor_info (list of lists): Raw sensor data including object ID, relative positions, yaw, and velocities.
      Each element is in the format [obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy].
    
    Returns:
    - list of lists: Processed local data ready for transformation to global coordinates.
      Each element contains [obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy].
    """
    local_path = []
    for data in sensor_info:
        obj_id = data[0]
        rel_x = data[1]
        rel_y = data[2]
        rel_h = data[3]
        rel_vx = data[4]
        rel_vy = data[5]

        local_path.append([obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy])

    return local_path

def convert_local_to_global(local_data, pose):
    """
    Converts local coordinates and velocities to global coordinates and velocities based on the vehicle's pose.

    Parameters:
    - local_data (list of lists): Each element contains [obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy]
      where rel_x, rel_y are the local coordinates,
      rel_h is the heading relative to the vehicle's heading,
      rel_vx, rel_vy are the local velocities.
    - pose (dict): The vehicle's current pose with 'x', 'y', and 'theta' (orientation in radians).

    Returns:
    - list of lists: Each element contains [obj_id, global_x, global_y, global_h, global_vx, global_vy]
    """
    global_data = []
    cos_theta = np.cos(pose['theta'])
    sin_theta = np.sin(pose['theta'])

    for data in local_data:
        obj_id, rel_x, rel_y, rel_h, rel_vx, rel_vy = data

        # Transform position
        global_x = pose['x'] + cos_theta * rel_x - sin_theta * rel_y
        global_y = pose['y'] + sin_theta * rel_x + cos_theta * rel_y

        # Transform heading
        global_h = (pose['theta'] + rel_h) % (2 * np.pi)

        # Transform velocities
        global_vx = cos_theta * rel_vx - sin_theta * rel_vy
        global_vy = sin_theta * rel_vx + cos_theta * rel_vy

        global_data.append([obj_id, global_x, global_y, global_h, global_vx, global_vy])

    return global_data

def kalman_filter(sensor_data):
    # 초기 상태 추정치 및 공분산 행렬
    x_est = np.array([0, 0, 0, 0])  # [x, y, vx, vy] 초기 위치 및 속도
    P_est = np.eye(4)  # 초기 상태 공분산

    # 칼만 필터 파라미터
    dt = 1.0  # 시간 간격
    A = np.array([[1, 0, dt, 0],  # 상태 전이 행렬
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
    H = np.array([[1, 0, 0, 0],  # 측정 행렬
                  [0, 1, 0, 0]])
    Q = 0.01 * np.eye(4)  # 프로세스 노이즈 공분산
    R = 0.1 * np.eye(2)  # 측정 노이즈 공분산
    I = np.eye(4)  # 단위 행렬

    # 센서 데이터 필터링
    filtered_data = []
    for data in sensor_data:
        # 측정 업데이트
        z = np.array([data[1], data[2]])  # 측정된 위치

        # 예측 단계
        x_pred = A @ x_est
        P_pred = A @ P_est @ A.T + Q

        # 측정 업데이트 단계
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x_est = x_pred + K @ (z - H @ x_pred)
        P_est = (I - K @ H) @ P_pred

        filtered_data.append(x_est.tolist())

    return filtered_data

def determine_possible_paths(vehicle, filtered_global_path):
    """
    현재 차량 위치에서 충분히 가까운 다른 차량이 있는지 확인하고, 해당 차량이 있으면 'stop'을 반환합니다.

    매개변수:
    - vehicle: 현재 차량 객체
    - filtered_global_path: 필터링된 글로벌 경로 데이터 리스트 [id, x, y, yaw, vx, vy]

    반환값:
    - string: 'stop' 또는 'continue'
    """
    current_x, current_y = vehicle.x, vehicle.y
    for data in filtered_global_path:
        _, x, y, _, _, _ = data
        if np.sqrt((x - current_x) ** 2 + (y - current_y) ** 2) < 10:  # 10미터 이내에 다른 차량이 있는지 검사
            return 'stop'
    return 'continue'
