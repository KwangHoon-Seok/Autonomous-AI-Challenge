import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

RADIUS_MAX = 100

def count_elements(lst):
    return dict(Counter(lst))

table = pd.read_csv("Pandar128_Angle_Correction_File.csv", encoding="utf-8", error_bad_lines=False)

file_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100000.npy"
pointcloud = np.load(file_path)
pointcloud = pointcloud.reshape(-1,4)

channels_to_be_deleted_upper = np.linspace(2, 40, 20, dtype=int)
channels_to_be_deleted_lower = np.linspace(43, 127, 43, dtype=int)
channels_to_be_deleted = np.union1d(channels_to_be_deleted_upper, channels_to_be_deleted_lower)
print(channels_to_be_deleted)

class COORD_CHANGER:
    def __init__(self, centerpoint):
        self.centerpoint = centerpoint
        self.elevations = []

    def cartesian2spherical(self, point_):
        point = np.array(point_) - np.array(self.centerpoint)
        radius = math.sqrt(point[0]**2 + point[1]**2 + point[2]**2)
        elevation = math.degrees(math.atan2(point[2], math.sqrt(point[0]**2 + point[1]**2)))
        azimuth = math.atan2(point[1], point[0])
        return [radius, elevation, azimuth, point_[3]]

    def spherical2cartesian(self, point):
        radius, elevation_, azimuth, intensity = point
        elevation = (np.pi/2) - math.radians(elevation_)
        x = radius * math.sin(elevation) * math.cos(azimuth) + self.centerpoint[0]
        y = radius * math.sin(elevation) * math.sin(azimuth) + self.centerpoint[1]
        z = radius * math.cos(elevation) + self.centerpoint[2]
        return [x, y, z, intensity]

Lidar_position = [0.133, -0.070, 2.023, 0]
coord_changer = COORD_CHANGER(Lidar_position)
spherical_points_ = [coord_changer.cartesian2spherical(point) for point in pointcloud]
spherical_points = []

deleted = []

for point in spherical_points_:
    closest_channel = int(table.iloc[(table["Elevation"] - point[1]).abs().idxmin()]["Channel"])
    if not any(np.isclose(closest_channel, channels_to_be_deleted)):
        spherical_points.append(point)
    else:
        deleted.append(closest_channel)

deletion_result = count_elements(deleted)
print(deletion_result)

np_spherical = np.array(spherical_points)

elevation_resolution = math.radians(0.125)
azimuth_resolution = math.radians(0.1)

num_azimuth_bins = int(2*np.pi/azimuth_resolution)
num_elevation_bins = int((math.radians(15)-math.radians(-25))/elevation_resolution)
print(num_azimuth_bins, num_elevation_bins)

image = np.zeros((num_elevation_bins, num_azimuth_bins), dtype=np.float64)

intrinsic_matrix = np.array([[1 / azimuth_resolution, 0, num_azimuth_bins / 2],
                             [0, 1 / elevation_resolution, num_elevation_bins / 2],
                             [0, 0, 1]], dtype=np.float32)

for point in np_spherical:
    radius, elevation, azimuth, _ = point

    normalized_radius = radius/RADIUS_MAX

    elevation = (math.radians(elevation) - math.radians(-25)) * (math.radians(20) - math.radians(-20)) / (math.radians(15) - math.radians(-25)) + math.radians(-20)
    xyz = intrinsic_matrix @ np.array([azimuth, elevation, 1], dtype=np.float32)

    x = int(xyz[0] / xyz[2])
    y = int(xyz[1] / xyz[2])
    if 0 <= x < num_azimuth_bins and 0 <= y < num_elevation_bins:
        if image[y, x] < normalized_radius:
            image[y, x] = normalized_radius

cartesian_points = [coord_changer.spherical2cartesian(point) for point in spherical_points]
np_cartesian = np.array(cartesian_points)
np.save("128to64_00200010.npy", np_cartesian)
print(f"num of point : {np_cartesian.shape} / origin number : {pointcloud.shape}")

plt.imshow(image, cmap='magma', origin='lower')
plt.title('128to64 Image')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.axis('off')
plt.show()
