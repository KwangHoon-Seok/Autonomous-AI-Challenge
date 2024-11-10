import os
import shutil

# Define source and destination directories
source_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/gt_database'
destination_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_gt_database'

# Ensure destination directory exists
os.makedirs(destination_path, exist_ok=True)

# Range of files to copy (from 00000000 to 00004938)
start_id = 0
end_id = 4938

# Copy files in the specified range
for i in range(start_id, end_id + 1):
    file_id = f"{i:08d}"  # Format with leading zeros
    for filename in os.listdir(source_path):
        if filename.startswith(file_id) and filename.endswith('.bin'):
            src_file = os.path.join(source_path, filename)
            dest_file = os.path.join(destination_path, filename)
            shutil.copy2(src_file, dest_file)  # Copy file to new directory
            print(f"Copied {filename} to Team_1_gt_database")

print("Selected files successfully copied to Team_1_gt_database.")
