import os

# Define the source and destination directories
original_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/gt_database'
new_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/Team_1_gt_database'

# Range of files to compare (from 00000000 to 00004938)
start_id = 0
end_id = 4938

# Track differences
differences = []

# Compare each file
for i in range(start_id, end_id + 1):
    file_id = f"{i:08d}"
    for filename in os.listdir(original_path):
        if filename.startswith(file_id) and filename.endswith('.bin'):
            original_file = os.path.join(original_path, filename)
            new_file = os.path.join(new_path, filename)

            # Check if new file exists
            if not os.path.exists(new_file):
                differences.append(f"{filename} missing in Team_1_gt_database.")
                continue

            # Compare file sizes
            if os.path.getsize(original_file) != os.path.getsize(new_file):
                differences.append(f"Size mismatch in {filename}")
                continue

            # Perform byte-by-byte comparison
            with open(original_file, 'rb') as f1, open(new_file, 'rb') as f2:
                if f1.read() != f2.read():
                    differences.append(f"Content mismatch in {filename}")

# Report results
if differences:
    print("Differences found in the following files:")
    for diff in differences:
        print(diff)
else:
    print("All files are identical between gt_database and Team_1_gt_database.")
