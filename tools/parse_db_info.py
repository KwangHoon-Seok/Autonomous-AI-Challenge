import os
import numpy as np
import pickle

# Define paths
label_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels'
pointcloud_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64'
database_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_gt_database'
output_pkl_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_custom_av_dbinfos_train_64.pkl'

# Initialize the dictionary to hold all data
data_dict = {'Vehicle': [], 'Pedestrian': [], 'Cyclist': []}

# Function to load bounding boxes and object types from the label file
def load_bounding_boxes(label_file):
    boxes = []
    with open(label_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 8:
                # Parse bounding box and object type
                x, y, z = map(float, data[:3])
                dx, dy, dz, heading = map(float, data[3:7])
                object_type = data[7]
                boxes.append({
                    'box3d_lidar': np.array([x, y, z, dx, dy, dz, heading], dtype=np.float32),
                    'object_type': object_type
                })
    return boxes

# Process each file in the database
for npy_file in os.listdir(pointcloud_path):
    if npy_file.endswith('.npy'):
        file_id = npy_file.split('.')[0]

        # Load bounding boxes and point cloud data
        label_file_path = os.path.join(label_path, f'{file_id}.txt')
        if not os.path.exists(label_file_path):
            print(f"Label file {label_file_path} does not exist. Skipping.")
            continue

        boxes = load_bounding_boxes(label_file_path)

        for i, box in enumerate(boxes):
            object_type = box['object_type']
            box3d_lidar = box['box3d_lidar']
            bin_file_path = os.path.join(database_path, f'{file_id}_{object_type}_{i}.bin')

            if not os.path.exists(bin_file_path):
                print(f"Bin file {bin_file_path} does not exist. Skipping.")
                continue

            # Load the points in the bounding box to count
            points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
            num_points_in_gt = points.shape[0]

            # Append the information to the dictionary
            data_dict[object_type].append({
                'name': object_type,
                'path': f'Team_1_gt_database/{file_id}_{object_type}_{i}.bin',
                'gt_idx': i,
                'box3d_lidar': box3d_lidar,
                'num_points_in_gt': num_points_in_gt
            })
            
            print(f'file_id : {file_id}')
# Save the dictionary to a .pkl file
with open(output_pkl_path, 'wb') as f:
    pickle.dump(data_dict, f)

print(f"Saved data to {output_pkl_path}")
