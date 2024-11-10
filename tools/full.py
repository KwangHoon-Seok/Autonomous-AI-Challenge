import argparse
import glob
from pathlib import Path
import numpy as np
import torch
import os
from tqdm import tqdm
import open3d as o3d
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

try:
    from visual_utils import open3d_vis_utils as V  # Open3D 시각화 유틸리티
    OPEN3D_FLAG = True
except ImportError:
    from visual_utils import visualize_utils as V  # Mayavi 시각화 유틸리티
    OPEN3D_FLAG = False

import pickle

def get_filename_without_extension(file_path):
    """파일 전체 경로에서 확장자를 제외한 파일명을 추출합니다."""
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension = os.path.splitext(filename_with_extension)[0]
    return filename_without_extension

def load_gt_labels(gt_path):
    """GT(Label) 파일을 로드하는 함수"""
    gt_labels = []
    with open(gt_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 7:
                box = [float(x) for x in parts[:7]]  # 좌표와 크기 정보
                gt_labels.append(box)
    return np.array(gt_labels)

def add_bounding_box_text(vis, boxes, scores):
    """예측된 바운딩 박스 위에 점수 출력"""
    if scores is not None:
        for box, score in zip(boxes, scores):
            center = box[:3]  # 바운딩 박스의 중심 좌표
            text_3d = o3d.geometry.Text3D.create_text(f'Score: {score:.2f}', center=center, font_size=12, color=(1, 0, 0))
            vis.add_geometry(text_3d)

def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None):
    """포인트 클라우드, GT 박스 및 예측 박스를 시각화"""
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 포인트 클라우드 시각화
    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])
    vis.add_geometry(pts)

    # GT 박스 시각화
    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, color=(0, 0, 1))  # 파란색 GT 박스

    # 예측 박스 및 점수 시각화
    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, color=(0, 1, 0))  # 초록색 예측 박스
        add_bounding_box_text(vis, ref_boxes, ref_scores)  # 점수 출력

    vis.run()
    vis.destroy_window()

def draw_box(vis, boxes, color):
    """바운딩 박스 시각화"""
    for box in boxes:
        center = box[:3]
        extents = box[3:6]
        rotation = box[6]
        obb = o3d.geometry.OrientedBoundingBox(center, np.eye(3), extents)
        obb.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, rotation]))
        obb.color = color
        vis.add_geometry(obb)
    return vis

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger)
        self.root_path = root_path
        self.ext = ext

        test_frame_ids_filename = os.path.join(root_path, "ImageSets/val.txt")
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

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()

    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    
    # 모델 가중치 로드 확인
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    logger.info(f"모델의 가중치가 {args.ckpt}에서 성공적으로 로드되었습니다.")
    
    model.cuda()
    model.eval()

    with torch.no_grad():
        for idx, data_dict in enumerate(tqdm(demo_dataset, desc="Processing dataset")):
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)
            pred_dict = pred_dicts[0]

            frame_id = data_dict['frame_id'][0]

            # GT Label 로드
            gt_labels_path = f'/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/labels/{frame_id}.txt'
            gt_boxes = load_gt_labels(gt_labels_path)

            # pred_scores 확인
            if 'pred_scores' not in pred_dict:
                logger.error(f"pred_scores가 예측 결과에 없습니다. pred_dict keys: {pred_dict.keys()}")
            else:
                pred_scores = pred_dict['pred_scores'].cpu().numpy() if isinstance(pred_dict['pred_scores'], torch.Tensor) else pred_dict['pred_scores']

            # 시각화
            V.draw_scenes(
                points=data_dict['points'][:, 1:], 
                gt_boxes=gt_boxes, 
                ref_boxes=pred_dict['pred_boxes'], 
                ref_scores=pred_scores,  # 예측 점수 전달
                ref_labels=pred_dict['pred_labels']
            )

if __name__ == '__main__':
    main()
