import numpy as np

# npy 파일 로드
point_cloud = np.load('/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/00100000.npy', allow_pickle=True)

# 배열의 shape 확인
print(point_cloud.shape)
