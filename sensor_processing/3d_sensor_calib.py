import numpy as np
import math
import open3d as o3d
import random
import math

from sensor_extrinsic_calib_2d import find_optimal_circle, find_radius

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

file_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00207282.npy"
pointcloud = np.load(file_path)
pointcloud = pointcloud.reshape(-1,4)
pointcloud = pointcloud[:,:3]
pointcloud = pointcloud[(pointcloud[:,0] >= -5) & (pointcloud[:,0] <= 5)  & (pointcloud[:,1] >= -5) & (pointcloud[:,1] <= 5) & (pointcloud[:,2] < 0.25)]

centerpoint_2D = [0.133, -0.063, 0]
first_radius = 4.9

second_circle = [0.13367291395389114, -0.07696587487587106, 5.056592528942225]
second_centerpoint_2D = [0.13367291395389114, -0.07696587487587106, 0]

#first_circle_index = np.array([find_radius(point, centerpoint_2D) for point in pointcloud])
#second_circle_index = np.array([find_radius(point, second_centerpoint_2D) for point in pointcloud])
#filtered_points = pointcloud[(first_circle_index > first_radius) & (second_circle_index <= second_circle[2])]
index = np.array([find_radius(point, [0.133, -0.07, 0]) for point in pointcloud])
filtered_points = pointcloud[(index <= second_circle[2])]
ransac = RANSACRegressor(base_estimator=LinearRegression(), min_samples=int(len(filtered_points[:,0])/2), residual_threshold=0.2, random_state=0)
X = np.array(filtered_points[:,:2])
Z = np.array(filtered_points[:,2])
print(X.shape, Z.shape)
ransac.fit(X, Z)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
plane_coef = ransac.estimator_.coef_
intercept = ransac.estimator_.intercept_

print("평면 방정식: z = {:.2f} * x + {:.2f} * y + {:.2f}".format(plane_coef[0], plane_coef[1], intercept))
print(intercept)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.paint_uniform_color(np.array([0,0,0], dtype=np.float64))

sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
sphere.translate(centerpoint_2D)
sphere.paint_uniform_color([1, 0, 0])

sphere2 = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
sphere2.translate(second_centerpoint_2D)
sphere2.paint_uniform_color([0, 1, 0])

vis = o3d.visualization.Visualizer()
vis.create_window(window_name="pointcloud", width=1280, height=720)
vis.add_geometry(pcd)
vis.add_geometry(sphere)
vis.add_geometry(sphere2)
options = vis.get_render_option()
options.background_color=np.ones(3)
options.point_size = 1.0
options.show_coordinate_frame = False

vis.run()
vis.destroy_window()