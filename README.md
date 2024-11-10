
Autonomous AI Challenge 2024
3D 동적 객체 검출 자율주행 인공지능 챌린지
목표: LiDAR 포인트 클라우드를 사용하여 3D 객체 검출 수행

모델 아키텍처 (Model Architecture)
PV-RCNN


PV-RCNN++


데이터 구성 (Data Configuration)
1. train.pkl - 데이터셋 및 라벨 정보
포인트 클라우드 (point_cloud)
num_features: 포인트 당 특징 수 (예: x, y, z, intensity)
lidar_idx: 각 LiDAR 스캔의 고유 식별자
주석 정보 (annos)
name: 객체 이름 배열 (예: 'Vehicle')
gt_boxes_lidar: 7D 바운딩 박스 좌표
num_points_in_gt: 각 바운딩 박스 내의 포인트 개수
difficulty: 난이도 레벨 배열
2. db_infos_pkl - gt 정보
각 객체 클래스별 정보 (예: Vehicle, Pedestrian, Cyclist)
구성 요소
name: 객체 이름
path: GT 데이터 파일 경로
gt_idx: GT 인덱스
box3d_lidar: 3D 바운딩 박스 좌표
num_points_in_gt: GT 내의 포인트 개수
3. gt_database - 라벨 정보
/label 파일에 정의된 클래스의 각 바운딩 박스에 포함된 포인트 정보 제공
4. points (.npy 파일)
각 프레임의 모든 포인트 데이터를 포함
Shape: (118674, 4)
Data Type: float32
포인트 예시:
css
코드 복사
[[-2.9074268e+01, -1.9766258e-02, 9.9172888e+00, 3.8039219e-01],
 [-3.5565208e+01, -2.0658094e-02, 9.0879440e+00, 1.1764707e-01],
 [-3.2280708e+01, -1.5841596e-02, 6.6269894e+00, 3.9607847e-01],
 ...]
5. labels 정보
각 프레임당 물체와 바운딩 박스 정보 ([x, y, z, dx, dy, dz, yaw])
성능 결과 (Results)
1. PV-RCNN (Epoch 10)
Object Type	Level 1 AP	Level 1 APH	Level 1 APL	Level 2 AP	Level 2 APH	Level 2 APL
Vehicle	0.7724	0.7597	0.7724	0.7566	0.7442	0.7566
Pedestrian	0.4173	0.2157	0.4173	0.4024	0.2080	0.4024
Sign	0.0000	0.0000	0.0000	0.0000	0.0000	0.0000
Cyclist	0.6611	0.5866	0.6611	0.6428	0.5704	0.6428
2. PV-RCNN (Epoch 30)


3. PV-RCNN++ with ResNet Backbone (Epoch 30)


학습 환경: GPU 3080 × 4
추가 결과 (Range to Image Transformation)


