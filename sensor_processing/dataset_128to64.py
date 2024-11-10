import numpy as np
import math
import pandas as pd
import os
from multiprocessing import Pool, cpu_count

class COORD_CHANGER:
    def __init__(self, centerpoint):
        self.centerpoint = centerpoint

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

def find_128_indices(file):
    with open(file, "r") as f:
        modifying_indices = [line.strip() for line in f]
    return modifying_indices

def process_file(args):
    file_name, modifying_indices, already_modified, table, coord_changer, channels_to_be_deleted, folder_path, destination_folder_path, frame_count = args
    file_path = os.path.join(folder_path, file_name)

    if file_name.endswith(".npy"):
        try:
            pointcloud = np.load(file_path).reshape(-1, 4)
            print(f"Processing file: {file_name}")
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
            return

        file_number = file_name[:8]

        if file_number in modifying_indices and file_number not in already_modified and frame_count%3==0:
            spherical_points = [coord_changer.cartesian2spherical(point) for point in pointcloud]

            valid_indices = [
                i for i, point in enumerate(spherical_points)
                if int(table.iloc[(table["Elevation"] - point[1]).abs().idxmin()]["Channel"]) not in channels_to_be_deleted
            ]

            cartesian_points = pointcloud[valid_indices]
            np_cartesian = np.array(cartesian_points)
            destination_path = os.path.join(destination_folder_path, file_name)
            np.save(destination_path, np_cartesian)
            print(f"Saved processed data for {file_name}. Number of points: {np_cartesian.shape[0]}")
            print("-"*50)

        elif file_number in modifying_indices and file_number not in already_modified and frame_count%3!=0:
            frame_count += 1

        elif file_number not in already_modified:
            destination_path = os.path.join(destination_folder_path, file_name)
            np.save(destination_path, pointcloud)
            print(f"Copied unmodified data for {file_name}")
            print("-"*50)

        else:
            print("already modified!")
            print("-"*50)

def main():
    table = pd.read_csv("Pandar128_Angle_Correction_File.csv", encoding="utf-8", on_bad_lines="skip")
    folder_path =  "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points"
    destination_folder_path = "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64"
    modifying_index_files =  "/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/ImageSets/train_128.txt"

    modifying_indices = find_128_indices(modifying_index_files)

    already_modified = []
    modified = os.listdir(destination_folder_path)
    for file_name in modified:
        file_number = file_name[:8]
        already_modified.append(file_number)

    print(f"Number of 128-channel data: {len(modifying_indices)}")
    print(f"Number of data already modified: {len(already_modified)}")

    if not os.path.exists(destination_folder_path):
        os.makedirs(destination_folder_path)

    files = os.listdir(folder_path)
    channels_to_be_deleted_upper = np.linspace(2, 40, 20, dtype=int)
    channels_to_be_deleted_lower = np.linspace(43, 127, 43, dtype=int)
    channels_to_be_deleted = np.union1d(channels_to_be_deleted_upper, channels_to_be_deleted_lower)
    Lidar_position = [0.133, -0.070, 2.023, 0]
    coord_changer = COORD_CHANGER(Lidar_position)

    args = [(file_name, modifying_indices, already_modified, table, coord_changer, channels_to_be_deleted, folder_path, destination_folder_path, idx)
            for idx, file_name in enumerate(files)]

    with Pool(min(20, cpu_count())) as pool:
        pool.map(process_file, args)

if __name__ == "__main__":
    main()