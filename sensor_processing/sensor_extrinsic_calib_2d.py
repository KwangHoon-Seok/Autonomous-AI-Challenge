import numpy as np
import math
import open3d as o3d

import random
import math

# 원 방정식 모델 계산 함수
def fit_circle(points):
    p1, p2, p3 = points
    # 세 점에서 원의 중심과 반지름 계산
    temp = p2[0]**2 + p2[1]**2
    bc = (p1[0]**2 + p1[1]**2 - temp) / 2
    cd = (temp - p3[0]**2 - p3[1]**2) / 2
    det = (p1[0] - p2[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p2[1])

    if abs(det) < 1e-10:
        return None  # 세 점이 일직선이면 원을 정의할 수 없음

    # 중심 좌표 (cx, cy)와 반지름 r 계산
    cx = (bc * (p2[1] - p3[1]) - cd * (p1[1] - p2[1])) / det
    cy = (cd * (p1[0] - p2[0]) - bc * (p2[0] - p3[0])) / det
    radius = math.sqrt((cx - p1[0])**2 + (cy - p1[1])**2)

    return (cx, cy, radius)

# 잔차 계산 함수
def residuals(circle, points, threshold):
    cx, cy, r = circle
    inliers = []
    for point in points:
        distance = math.sqrt((point[0] - cx)**2 + (point[1] - cy)**2)
        if abs(distance - r) < threshold:
            inliers.append(point)
    return inliers

# RANSAC 알고리즘
def ransac_circle(points_, iterations=1000, threshold=0.1):
    best_circle = None
    best_inliers = []
    points = list(points_)
    for _ in range(iterations):
        # 임의의 세 점 선택
        sample_points = random.sample(points, 3)
        circle = fit_circle(sample_points)
        if circle is None:
            continue

        # 잔차 계산하여 인라이어 점수 확인
        inliers = residuals(circle, points, threshold)
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_circle = circle

    return best_circle, best_inliers

def find_radius(point, centerpoint=np.array([0, 0, 0])):
    return math.sqrt((point[0] - centerpoint[0])**2 + (point[1] - centerpoint[1])**2)

def find_optimal_circle(points, radius=5.0):
    centerpoint = [0,0,0]
    optimal_circle = None
    largest_num_of_inliers = 0

    for idx in range(50):
        radius_from_vehicle_coord = np.array([find_radius(point, centerpoint) for point in points])
        circle_candidates = points[(radius_from_vehicle_coord <= radius)]
        best_circle, inliers = ransac_circle(circle_candidates, iterations=1000, threshold=0.05)
        print("--------------------------------------------------------------------------------")
        print(f"iteration : {idx}")
        if(len(inliers) > largest_num_of_inliers):
            print(f"number of inliers : {len(inliers)}")
            print(best_circle)
            optimal_circle = best_circle
            largest_num_of_inliers = len(inliers)
            centerpoint = [best_circle[0], best_circle[1], 0]
    return optimal_circle

def main():
    #(0.13545413508348617, -0.059511052833606276, 4.855168795387466)
    file_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100208.npy"
    pointcloud = np.load(file_path)
    pointcloud = pointcloud.reshape(-1,4)
    pointcloud = pointcloud[:,:3]
    roi_points_ = pointcloud[(pointcloud[:,0] >= -5) & (pointcloud[:,0] <= 5)  & (pointcloud[:,1] >= -5) & (pointcloud[:,1] <= 5) & (pointcloud[:,2] < 0.25)]

    #optimal_circle = find_optimal_circle(roi_points_)
    optimal_circle_64 = [1.0558905478534133, -0.09557040868690889, 4.8643649928382695]
    optimal_circle = [0.133, -0.070, 4.8643649928382695]

    key = np.array([find_radius(point, [optimal_circle[0], optimal_circle[1], 0]) for point in pointcloud])

    roi_points = pointcloud[(key<=4.9) & (pointcloud[:,2] < 0.5)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(roi_points)
    pcd.paint_uniform_color(np.array([0,0,0], dtype=np.float64))
    centerpoint = pcd.get_center()
    print(f"true, new : {[optimal_circle[0], optimal_circle[1], 0]} , {centerpoint}")
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)

    sphere.paint_uniform_color([1, 0, 0])

    sphere_true = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    sphere_true.translate((optimal_circle[0], optimal_circle[1], 0))
    sphere_true.paint_uniform_color([0, 1, 0])

    sphere_64 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    sphere_64.translate((optimal_circle_64[0], optimal_circle_64[1], 0))
    sphere_64.paint_uniform_color([0, 0, 1])

    coord = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
    coord.paint_uniform_color([0,0,1])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="pointcloud", width=1280, height=720)
    vis.add_geometry(pcd)
    vis.add_geometry(sphere)
    vis.add_geometry(sphere_true)
    vis.add_geometry(sphere_64)

    options = vis.get_render_option()
    options.background_color=np.ones(3)
    options.point_size = 1.0
    options.show_coordinate_frame = False

    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()