import numpy as np
import os
from multiprocessing import Pool, cpu_count

# Define paths
label_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels'
pointcloud_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64'
output_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_gt_database'

# Ensure output directory exists
os.makedirs(output_path, exist_ok=True)

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
                object_type = data[7]  # "Vehicle", "Pedestrian", or "Cyclist"
                boxes.append((x, y, z, dx, dy, dz, heading, object_type))
    return boxes

# Function to filter points within a bounding box
def filter_points_in_box(points, box):
    x, y, z, dx, dy, dz, heading, _ = box
    x_min, x_max = x - dx/2, x + dx/2
    y_min, y_max = y - dy/2, y + dy/2
    z_min, z_max = z - dz/2, z + dz/2
    # Apply bounding box filter
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    )
    return points[mask]

# Function to process a single file based on file_id derived from .npy files
def process_file(file_id):
    # Exclude test dataset with file_id in the 10000000 range
    if int(file_id) >= 10000000:
        print(f"Skipping test dataset file_id {file_id}")
        return
    
    label_file_path = os.path.join(label_path, f'{file_id}.txt')
    pointcloud_file_path = os.path.join(pointcloud_path, f'{file_id}.npy')
    
    # Check if the label file exists for the current file_id
    if not os.path.exists(label_file_path):
        print(f"Label file {label_file_path} does not exist. Skipping.")
        return
    
    # Load bounding boxes and point cloud data
    boxes = load_bounding_boxes(label_file_path)
    points = np.load(pointcloud_file_path)
    
    # Process each box and save points
    for i, box in enumerate(boxes):
        object_type = box[7]
        output_file = os.path.join(output_path, f'{file_id}_{object_type}_{i}.bin')
        
        # Check for duplicate file
        if os.path.exists(output_file):
            print(f"File {output_file} already exists. Skipping.")
            continue
        
        # Filter points and save if not empty
        filtered_points = filter_points_in_box(points, box)
        if filtered_points.size > 0:
            filtered_points.astype(np.float32).tofile(output_file)
            print(f"{output_file} to Team_1_gt_database")
        
        else :
            print("CANNOT ADD GT DATABASE")

# Main function to execute multiprocessing
def create_gt_database(num_cpus=None):
    # List of .npy files to process
    npy_files = [f for f in os.listdir(pointcloud_path) if f.endswith('.npy')]
    file_ids = [f.split('.')[0] for f in npy_files if int(f.split('.')[0]) < 10000000]  # Exclude test dataset file_ids
    
    # Set default CPU count if not provided
    num_cpus = num_cpus or cpu_count()
    print(f"Using {num_cpus} CPUs for processing.")
    
    # Execute in parallel
    with Pool(num_cpus) as pool:
        pool.map(process_file, file_ids)

# Specify the desired number of CPUs
desired_cpus = 24  # Adjust this number as needed
create_gt_database(num_cpus=desired_cpus)
