import os
import pickle
import numpy as np

# Define paths
label_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels'
pointcloud_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64'
database_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_gt_database'
train_txt_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/ImageSets_Team_1_64/val.txt'
output_pkl_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_custom_av_infos_val_64.pkl'

# Load file_ids from train.txt
with open(train_txt_path, 'r') as f:
    file_ids = [line.strip() for line in f.readlines()]

# Function to load bounding boxes and object types from the label file
def load_bounding_boxes(label_file):
    boxes = []
    names = []
    with open(label_file, 'r') as f:
        for line in f:
            data = line.strip().split()
            if len(data) >= 8:
                x, y, z = map(float, data[:3])
                dx, dy, dz, heading = map(float, data[3:7])
                object_type = data[7]
                boxes.append([x, y, z, dx, dy, dz, heading])
                names.append(object_type)
    return np.array(boxes, dtype=np.float32), np.array(names, dtype='<U10')

# Function to count points in each bounding box .bin file
def count_points_in_bin(bin_file_path):
    points = np.fromfile(bin_file_path, dtype=np.float32).reshape(-1, 4)
    return points.shape[0]

# Create a list to store all entries
entries = []

# Process each file_id from train.txt
for file_id in file_ids:
    # Load bounding box and label data
    label_file_path = os.path.join(label_path, f'{file_id}.txt')
    if not os.path.exists(label_file_path):
        print(f"Label file {label_file_path} does not exist. Skipping.")
        continue

    gt_boxes_lidar, names = load_bounding_boxes(label_file_path)

    # Count points in each bounding box
    num_points_in_gt = []
    for idx, name in enumerate(names):
        bin_file_path = os.path.join(database_path, f'{file_id}_{name}_{idx}.bin')
        if not os.path.exists(bin_file_path):
            print(f"Bin file {bin_file_path} does not exist. Skipping.")
            continue
        
        num_points_in_gt.append(count_points_in_bin(bin_file_path))

    # Add difficulty level (all set to 0 as per the format)
    difficulty = np.zeros(len(names), dtype=np.int32)

    # Add entry to the list
    entry = {
        'point_cloud': {
            'num_features': 4,  # Assuming each point has [x, y, z, intensity]
            'lidar_idx': file_id
        },
        'annos': {
            'name': names,
            'gt_boxes_lidar': gt_boxes_lidar,
            'num_points_in_gt': np.array(num_points_in_gt, dtype=np.int32),
            'difficulty': difficulty
        }
    }
    entries.append(entry)

# Save the list to a .pkl file
with open(output_pkl_path, 'wb') as f:
    pickle.dump(entries, f)

print(f"Saved data to {output_pkl_path}")
