import numpy as np
import math
import matplotlib.pyplot as plt

file_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00200010.npy"
pointcloud = np.load(file_path)
pointcloud = pointcloud.reshape(-1,4)
pointcloud[:,3] = 1

# 라이다 기준이었으면 구면 좌표계로 분류됐을텐데, 자차좌표계로 나오니까 ... ! 원통좌표계로 시도
def cartesian2spherical(point):
    radius = math.sqrt(point[0]**2 + point[1]**2)
    elevation = point[2]
    azimuth = math.atan2(point[1],point[0])
    return [radius, elevation, azimuth]

# 포인트 클라우드를 원통 좌표계로 변환
spherical_points = [cartesian2spherical(point) for point in pointcloud]
np_spherical = np.array(spherical_points)

print(np.max(pointcloud[:,2]))
print(np.min(pointcloud[:,2]))
print(np.min(np_spherical[:,0]))
elevation_resolution = (60/1280)
azimuth_resolution = math.radians(0.1)
num_elevation_bins = int(60/elevation_resolution) # -30 ~ 30m
num_azimuth_bins = int(2*np.pi/azimuth_resolution)
print(num_azimuth_bins, num_elevation_bins)
image = np.ones((num_elevation_bins, num_azimuth_bins, 3), dtype=np.uint8) * 255  # 흰색 배경 (RGB 값이 255)
intrinsic_matrix = np.array([[-1 / azimuth_resolution, 0, num_azimuth_bins / 2],
                             [0, -1 / elevation_resolution, num_elevation_bins / 2],
                             [0, 0, 1]], dtype=np.float32)

for point in np_spherical:
    radius, elevation, azimuth = point
    xyz = intrinsic_matrix @ np.array([azimuth, elevation, 1], dtype=np.float32)
    x = int(xyz[0] / xyz[2])
    y = int(xyz[1] / xyz[2])

    # 이미지 경계 내에 있는지 확인
    if 0 <= x < num_azimuth_bins and 0 <= y < num_elevation_bins:
        image[y, x] = (0, 0, 0)  # 검은색 점 추가
        # # 거리에 따라 밝기 조절 (가까울수록 밝음)
        # brightness = int(255 * (1 - min(radius /20 , 1)))
        # image[y, x] = (brightness, brightness, brightness)  # 회색조로 밝기 조절
print(f"num of point : {pointcloud.shape}")
# Matplotlib을 사용하여 이미지 플롯
plt.imshow(image)
plt.title('Plotted Image')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 여백 제거

plt.axis('off')
plt.show()