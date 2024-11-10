import open3d as o3d
import numpy as np

# NumPy 배열로 .npy 파일 읽기
data = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64/00101304.npy"
#data = "/home/ailab/git/Team_1/baseline_code_and_model/3D_detection_car/02_baseline_code_and_model/sensor_processing/128to64_00200010.npy"
points = np.load(data)
points = points.reshape(-1,4)
points = points[:,:3]

print(points.shape)
# 포인트 클라우드 객체 생성
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color(np.array([0,0,0], dtype=np.float64))

centerpoint = pcd.get_center()
print(centerpoint)
sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)  # 구의 크기 설정
sphere.translate(centerpoint)  # 구를 중심점으로 이동
sphere.paint_uniform_color([1, 0, 0])  # 구의 색상을 빨간색으로 설정

sensor = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)  # 구의 크기 설정
sensor.translate((0.133, -0.070, 2.023))  # 구를 중심점으로 이동
sensor.paint_uniform_color([0, 1, 0])  # 구의 색상을 빨간색으로 설정

coord = o3d.geometry.TriangleMesh.create_sphere(radius=0.1, resolution=20)
coord.paint_uniform_color([0,0,1])
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="pointcloud", width=1280, height=720)
vis.add_geometry(pcd)
vis.add_geometry(sphere)
vis.add_geometry(coord)
vis.add_geometry(sensor)

options = vis.get_render_option()
options.background_color=np.ones(3)
options.point_size = 1.0
options.show_coordinate_frame = False

controller = vis.get_view_control()
controller.set_front(np.array([ 0.01150317424812144, -0.84550161745327712, 0.53384894105552894 ]))
controller.set_lookat(np.array([ -0.68655776977539062, 4.0639286041259766, 9.7854307889938354 ]))
controller.set_up(([ -0.015089781335444776, 0.53367668826487802, 0.84555395505069986 ]))
controller.set_zoom(0.19999999999999962)

vis.run()
vis.destroy_window()