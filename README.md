# Autonomous-AI-Challenge
2024 자율주행 인공지능 챌린지(3D 동적객체 검출)
->주어진 LiDAR point를 사용하여 3D 객체 검출
### [Model Architecture]
<PV-RCNN>
![image](https://github.com/user-attachments/assets/e53c7e5b-3566-4add-8cb4-64124a6b31dc)

<PV-RCNN++>
![image](https://github.com/user-attachments/assets/5047ed61-b79d-4d37-8236-f45f54e0dc09)




[데이터 구성]
### 1. train.pkl - DataSet 및 Label에 대한 정확한 정보
   [
    {
        'point_cloud': {
            'num_features': 4,                 # 포인트 클라우드의 각 포인트당 특징 수 (보통 x, y, z, intensity)
            'lidar_idx': '00000001'             # 각 Lidar 스캔의 고유 식별자
        },
        'annos': {
            'name': np.array([
                'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 
                'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 
                'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 
                'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 'Vehicle', 
                'Vehicle'
            ]),  # 객체 이름의 배열
            'gt_boxes_lidar': np.array([
                [-0.047037, -19.070677, 1.126073, 4.835565, 1.934771, 1.537523, 2.653253],
                [3.590872, -12.771045, 1.322447, 5.00492, 2.018615, 1.601063, 2.605351],
                [55.624241, -11.98928, 2.889861, 3.596104, 1.803264, 1.603259, 0.463385],
                # ...
                [13.531695, 46.529221, 0.438992, 4.637841, 1.989035, 1.698529, -0.020413]
            ]),  # 각 객체에 대한 7D 바운딩 박스 좌표
            'num_points_in_gt': np.array([
                775, 2887, 31, 131, 3465, 2137, 520, 880, 222, 2726, 1195, 
                625, 367, 764, 37, 407, 36, 104, 315, 1438, 17, 25, 5, 19, 26
            ]),  # 각 바운딩 박스 내의 포인트 개수
            'difficulty': np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 난이도 레벨
        }
    },
    # ... 추가 항목들
]

### 2. /db_infos_pkl - gt 정보
{
    'Vehicle': [
        {
            'name': 'Vehicle',
            'path': 'gt_database/00000000_Vehicle_0.bin',
            'gt_idx': 0,
            'box3d_lidar': np.array([0.507079, -19.058336, 1.126073, 4.835565, 1.934771, 1.537523, 2.653356], dtype=np.float32),
            'num_points_in_gt': 846
        },
        {
            'name': 'Vehicle',
            'path': 'gt_database/00000001_Vehicle_1.bin',
            'gt_idx': 1,
            'box3d_lidar': np.array([...], dtype=np.float32),
            'num_points_in_gt': ...
        },
        # ... (다수의 Vehicle 객체)
    ],
    'Pedestrian': [
        {
            'name': 'Pedestrian',
            'path': 'gt_database/00000015_Pedestrian_8.bin',
            'gt_idx': 8,
            'box3d_lidar': np.array([18.770025, 6.158173, 0.86841, 0.898277, 0.802549, 1.698892, -1.564585], dtype=np.float32),
            'num_points_in_gt': 23
        },
        # ... (다수의 Pedestrian 객체)
    ],
    'Cyclist': [
        {
            'name': 'Cyclist',
            'path': 'gt_database/00000100_Cyclist_6.bin',
            'gt_idx': 6,
            'box3d_lidar': np.array([14.644975, -5.108981, 1.224325, 1.102101, 0.798372, 1.854168, 0.07854], dtype=np.float32),
            'num_points_in_gt': 346
        },
        # ... (다수의 Cyclist 객체)
    ]
}

### 3. gt_database - Label에 대한 정보
  -> Label의 class_name이 /label 에서 정해졌을 때 그 frame에서 물체 각각의 바운딩 박스에 있는 포인트들에 대한 정보

### 4. /points(.npy에 대한 정보)
  -> 한 frame당 모든 point
Shape of point cloud: (118674, 4)
Data type of point cloud: float32
Each point has 4 values.
First few points in the point cloud:
[[-2.9074268e+01 -1.9766258e-02  9.9172888e+00  3.8039219e-01]
 [-3.5565208e+01 -2.0658094e-02  9.0879440e+00  1.1764707e-01]
 [-3.2280708e+01 -1.5841596e-02  6.6269894e+00  3.9607847e-01]
 [-3.5376541e+01 -1.4064273e-02  5.1457205e+00  2.3137257e-01]
 [-2.9074268e+01 -1.9766258e-02  9.9172888e+00  3.8039219e-01]]

### 5. /labels 정보, 한 frame에 물체와 바운딩 박스 정보
  [x,y,z,dx,dy,dz,yaw]


## Result 
1. pv-rcnn(Epoch 10)
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP: 0.7724
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH: 0.7597
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APL: 0.7724
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP: 0.7566 
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH: 0.7442
OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APL: 0.7566
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP: 0.4173
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH: 0.2157
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APL: 0.4173
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP: 0.4024
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH: 0.2080
OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APL: 0.4024
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APL: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH: 0.0000
OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APL: 0.0000
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP: 0.6611
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH: 0.5866
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APL: 0.6611
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP: 0.6428
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH: 0.5704
OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APL: 0.6428

2. pv-rcnn(Epoch 30)
![image](https://github.com/user-attachments/assets/15dcc4f7-c311-4244-ae69-4b0870d84a4a)

3. pv-rcnn++Resnet(Epoch 30)
![image](https://github.com/user-attachments/assets/515593ae-7549-451a-9191-479243b57797)

---학습 환경 gpu: 3080 * 4---

## Range to Image - Result
![image](https://github.com/user-attachments/assets/d114791e-672e-4560-8ca3-d673d0f34edf)
