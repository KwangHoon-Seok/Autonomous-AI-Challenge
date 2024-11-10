import os
import shutil

# Define source and destination directories
source_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_128to64'
destination_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/ImageSets_Team_1_64'

# Ensure destination directory exists
os.makedirs(destination_path, exist_ok=True)

files = os.listdir(source_path)
datum = []

for file_name in files:
    file_number = file_name[:8]
    if int(file_number) < 10000000:
        datum.append(file_number)
    else:
        print(f"{file_number}")

txt_file_path = os.path.join(destination_path, "train_.txt")

with open(txt_file_path, 'w') as f:
    for name in datum:
        f.write(name + '\n')