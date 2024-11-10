import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import os
from tqdm import tqdm
import open3d as o3d
import pickle

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


def get_filename_without_extension(file_path):
    """파일 전체 경로에서 확장자를 제외한 파일명을 추출합니다."""
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext

        test_frame_ids_filename = os.path.join(root_path, "ImageSets/test.txt")
        with open(test_frame_ids_filename, 'r') as f:
            test_frame_ids = [line.strip() for line in f.readlines()]

        self.sample_file_list = sorted(
            [os.path.join(root_path, f"points/{frame_id}{ext}") for frame_id in test_frame_ids]
        )

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError(f"File extension {self.ext} is not supported")

        frame_id = get_filename_without_extension(self.sample_file_list[index])
        input_dict = {'points': points, 'frame_id': frame_id}

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='Argument parser for demo')
    parser.add_argument('--cfg_file', type=str, required=True, help='Specify the config for demo')
    parser.add_argument('--data_path', type=str, required=True, help='Specify the custom_av directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.npy', help='Specify the extension of your point cloud data file')
    parser.add_argument('--gt_path', type=str, required=True, help='Specify the ground truth labels directory')

    args = parser.parse_args()
    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def load_gt_labels(gt_path, frame_id):
    """
    GT 데이터에서 해당 프레임의 라벨 파일을 불러옵니다.
    Args:
        gt_path (str): GT 데이터 경로
        frame_id (str): 프레임 ID
    Returns:
        gt_boxes (np.ndarray): GT 바운딩 박스 (N, 7)
        gt_labels (list): GT 클래스 이름 리스트
    """
    gt_labels = []
    gt_boxes = []

    # 해당 frame_id로 시작하는 모든 파일 검색 (예: 00003377_Vehicle_0.bin)
    gt_files = glob.glob(os.path.join(gt_path, f"{frame_id}_*.bin"))

    for gt_file in gt_files:
        # 파일 이름에서 클래스 이름 추출 (예: 00003377_Vehicle_0.bin -> Vehicle)
        class_name = gt_file.split('_')[1]

        # 바이너리 파일에서 3D 바운딩 박스 정보 로드 (이 부분은 데이터 형식에 따라 다를 수 있음)
        bbox = np.fromfile(gt_file, dtype=np.float32)  # .bin 파일을 float32로 읽어옴

        if bbox.shape[0] == 7:  # 3D 바운딩 박스는 [x, y, z, l, w, h, yaw]로 구성
            gt_boxes.append(bbox)
            gt_labels.append(class_name)
        else:
            print(f"Unexpected bbox shape in file {gt_file}")

    return np.array(gt_boxes), gt_labels

def create_3d_box(box, color=(1, 0, 0)):
    """
    3D 바운딩 박스를 생성합니다.
    Args:
        box (np.ndarray): 바운딩 박스 정보 (x, y, z, l, w, h, yaw)
        color (tuple): 바운딩 박스의 색상 (r, g, b)
    Returns:
        o3d.geometry.LineSet: Open3D 3D 바운딩 박스 객체
    """
    x, y, z, l, w, h, yaw = box

    # 바운딩 박스 좌표 (점군 좌표계에 맞게 설정)
    box3d = np.array([
        [l / 2, w / 2, 0],
        [-l / 2, w / 2, 0],
        [-l / 2, -w / 2, 0],
        [l / 2, -w / 2, 0],
        [l / 2, w / 2, h],
        [-l / 2, w / 2, h],
        [-l / 2, -w / 2, h],
        [l / 2, -w / 2, h]
    ])

    # 회전 적용 (yaw 각도는 z축을 기준으로 회전)
    rot_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 회전 및 변환 적용
    box3d = np.dot(box3d, rot_matrix.T)  # 회전
    box3d += np.array([x, y, z])  # 위치 이동 (중심점 적용)

    # 박스를 그릴 선 정보 (12개)
    lines = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]

    # Open3D LineSet 객체 생성
    bbox = o3d.geometry.LineSet()
    bbox.points = o3d.utility.Vector3dVector(box3d)
    bbox.lines = o3d.utility.Vector2iVector(lines)

    # 각 라인에 색상 적용
    colors = [color for _ in range(len(lines))]
    bbox.colors = o3d.utility.Vector3dVector(colors)

    return bbox



def draw_boxes(points, gt_boxes, pred_boxes):
    """
    Ground Truth 및 예측 바운딩 박스를 점군 데이터와 함께 시각화합니다.
    Args:
        points (np.ndarray): 포인트 클라우드 데이터
        gt_boxes (np.ndarray): GT 바운딩 박스 (N, 7)
        pred_boxes (np.ndarray): 모델 예측 바운딩 박스 (N, 7)
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 포인트 클라우드 시각화
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])  # (x, y, z) 좌표만 사용
    vis.add_geometry(pcd)

    # GT 바운딩 박스 시각화 (파란색)
    for box in gt_boxes:
        bbox = create_3d_box(box, color=(0, 0, 1))  # 파란색
        vis.add_geometry(bbox)
    # 예측 바운딩 박스 시각화 (빨간색)
    for box in pred_boxes:
        bbox = create_3d_box(box, color=(1, 0, 0))  # 빨간색
        vis.add_geometry(bbox)

    vis.run()
    vis.destroy_window()



def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    # 데이터셋 및 모델 준비
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="Processing dataset")):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_dict = pred_dicts[0]
            pred_boxes = pred_dict['pred_boxes'].cpu().numpy()
            frame_id = data_dict['frame_id'][0]

            # Ground Truth 불러오기
            gt_boxes, gt_labels = load_gt_labels(args.gt_path, frame_id)

            # 포인트 클라우드 데이터 불러오기
            points = data_dict['points'].cpu().numpy()

            # GT와 예측 바운딩 박스 시각화
            draw_boxes(points, gt_boxes, pred_boxes)

if __name__ == '__main__':
    main()
