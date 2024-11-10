import argparse
import glob
from pathlib import Path

try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        # data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]
        data_file_list = list(Path(root_path).rglob('*.npy'))
        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='/home/ailab/git/Team_1/OpenPCDet/tools/cfgs/kitti_models/voxel_rcnn_car.yaml',
                        help='specify the config for demo')
    # parser.add_argument('--data_path', type=str, default='demo_data',
    #                     help='specify the point cloud data file or directory')
    parser.add_argument('--data_path', type=str, default='/home/ailab/AILabDataset/01_Open_Dataset/39_AutoDna/3d_object_detection/3d_mod_av_db/points/',
                    help='specify the point cloud data directory')

    # 학습된 가중치 파일
    # parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')

    parser.add_argument('--ckpt', type=str, default='/home/ailab/git/Team_1/baseline_code_and_model/3D_detection_car/02_baseline_code_and_model/tools/voxel_rcnn_car_84.54.pth',
                    help='specify the pretrained model')

    # 데이터 파일 확장자 설정
    # parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')
    parser.add_argument('--ext', type=str, default='.npy', help='specify the extension of your point cloud data file')


    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg

def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # 데이터셋 로드 경로가 디렉터리인지 확인
    if not Path(args.data_path).is_dir():
        logger.error(f"Data path is not a directory: {args.data_path}")
        return

    # 경로 내의 파일 리스트 디버깅
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(args.data_path), ext=args.ext, logger=logger
    )

    logger.info(f"Searching for files in: {args.data_path}")
    logger.info(f"Found files: {demo_dataset.sample_file_list}")

    logger.info(f'Total number of samples: {len(demo_dataset)}')
    if len(demo_dataset) == 0:
        logger.error("No samples found in the specified data path!")
        return

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
    model.cuda()
    model.eval()
    with torch.no_grad():
        for idx, data_dict in enumerate(demo_dataset):
            logger.info(f'Visualized sample index: {idx + 1}')
            data_dict = demo_dataset.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
            )

            if not OPEN3D_FLAG:
                mlab.show(stop=True)

    logger.info('Demo done.')


# def main():
#     args, cfg = parse_config()
#     logger = common_utils.create_logger()
#     logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
#     demo_dataset = DemoDataset(
#         dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
#         root_path=Path(args.data_path), ext=args.ext, logger=logger
#     )
#     logger.info(f'Total number of samples: \t{len(demo_dataset)}')

#     model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=demo_dataset)
#     model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
#     model.cuda()
#     model.eval()
#     with torch.no_grad():
#         for idx, data_dict in enumerate(demo_dataset):
#             logger.info(f'Visualized sample index: \t{idx + 1}')
#             data_dict = demo_dataset.collate_batch([data_dict])
#             load_data_to_gpu(data_dict)
#             pred_dicts, _ = model.forward(data_dict)

#             V.draw_scenes(
#                 points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
#                 ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels']
#             )

#             if not OPEN3D_FLAG:
#                 mlab.show(stop=True)

#     logger.info('Demo done.')


if __name__ == '__main__':
    main()
