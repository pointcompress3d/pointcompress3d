#!/usr/bin/env python3

from depoco.trainer import DepocoNetTrainer
from ruamel import yaml
import argparse
import time
import depoco.utils.point_cloud_utils as pcu
import os
import numpy as np
from open3d import *

if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./encode.py")
    parser.add_argument(
        '--config_cfg', '-cfg',
        type=str,
        required=False,
        default='network_files/e3/e3.yaml',
        help='configitecture yaml cfg file. See /config/config for sample. No default!',
    )
    parser.add_argument(
        '--file_ext', '-fe',
        type=str,
        required=False,
        default='test/bin',
        help='Extends the output file name by the given string',
    )
    
    
    parser.add_argument(
        '--file_path', '-fp',
        type=str,
        required=False,
        help='Pass a binary point cloud file path'
    )

    # TODO: add filepath to 1 point cloud
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    yaml_parser = yaml.YAML(typ='safe', pure=True)
    with open(FLAGS.config_cfg, 'r') as file:
        config = yaml_parser.load(file)
    print('loaded yaml flags')
    print('config:', FLAGS.config_cfg)
    trainer = DepocoNetTrainer(config)
    trainer.decode(load_model=True, best_model=True)
    # decode, filename = trainer.decode_single(
    #     path=FLAGS.file_path, 
    #     load_model=True, 
    #     best_model=True
    # )

    # pcd = geometry.PointCloud()
    # np_points = decode.cpu().numpy().astype(np.float16)
    # np_points = np.unique(np_points, axis=0)
    # pcd.points = utility.Vector3dVector(np_points)
    # filename_without_extension = os.path.basename(filename).split(".")[0]
    # io.write_point_cloud(f'/data/test/decoded/{filename_without_extension}.pcd', pcd)