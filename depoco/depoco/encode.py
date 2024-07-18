#!/usr/bin/env python3

from depoco.trainer import DepocoNetTrainer
from ruamel import yaml
import argparse
import time
import depoco.utils.point_cloud_utils as pcu
import os
import pickle

# TODO: load only 1 file instead of whole folder
if __name__ == "__main__":
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

    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    yaml_parser = yaml.YAML(typ='safe', pure=True)
    with open(FLAGS.config_cfg, 'r') as file:
        config = yaml_parser.load(file)
    print('loaded yaml flags')
    print('config:', FLAGS.config_cfg)
    trainer = DepocoNetTrainer(config)
    # TODO: load only 1 point cloud
    # TODO: pass 1 point cloud to encode() method
    trainer.encode(load_model=True, best_model=True)
    # compressed, filename = point_cloud_encoded = trainer.encode_single(
    #     path=FLAGS.file_path, 
    #     load_model=True, 
    #     best_model=True
    # )
    
    # filename_without_extension = os.path.basename(filename).split(".")[0]
    # with open(f"/data/test/encoded/{filename_without_extension}.bin", "wb") as file:
    #     pickle.dump(compressed, file)