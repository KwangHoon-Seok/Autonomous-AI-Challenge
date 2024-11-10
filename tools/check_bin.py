import numpy as np

# 파일 경로 설정
bin_file_path = '/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/gt_database/00100899_Cyclist_113.bin'

# .bin 파일 읽기 함수 정의
def read_bin_file(file_path):
    # 포인트 클라우드 데이터를 float32 형태로 읽음 (각 포인트는 [x, y, z, intensity] 형태)
    points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
    return points

# .bin 파일 읽기 및 출력
points = read_bin_file(bin_file_path)
print(f"Loaded points shape: {points.shape}")
print(points)
