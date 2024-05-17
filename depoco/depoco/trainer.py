#!/usr/bin/env python3
from open3d import *

import depoco.datasets.submap_handler as submap_handler
import depoco.evaluation.evaluator as evaluator
import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import islice
import pickle

import os
from torch.utils.tensorboard import SummaryWriter
from ruamel import yaml
import argparse
import depoco.architectures.network_blocks as network
import chamfer3D.dist_chamfer_3D

import depoco.utils.point_cloud_utils as pcu
import subprocess

# import depoco.utils.checkpoint as chkpt
import depoco.architectures.loss_handler as loss_handler
from tqdm.auto import trange, tqdm
from datasets.submap_handler import SubMap
import json

class DepocoNetTrainer():
    def __init__(self, config):
        t_start = time.time()
        # parameters
        # config['git_commit_version'] = str(subprocess.check_output(
        #     ['git', 'rev-parse', '--short', 'HEAD']).strip())
        self.config = config
        self.experiment_id = self.config["train"]["experiment_id"]
        # self.submaps = submap_handler.SubmapHandler(self.config)
        t_sm = time.time()
        self.submaps = submap_handler.SubMapParser(config)
        print(f'Loaded Submaps ({time.time()-t_sm}s')

        self.max_nr_pts = self.config["train"]["max_nr_pts"]
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")

        ### Load Encoder and Decoder ####
        t_model = time.time()
        self.encoder_model = None
        self.decoder_model = None
        self.getModel(config)
        print(f'Loaded Model ({time.time()-t_model}s)')

        ##################################
        ########## Loss Attributes #######
        ##################################
        self.cham_loss = chamfer3D.dist_chamfer_3D.chamfer_3DDist()
        self.pairwise_dist = nn.PairwiseDistance()
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.w_transf2map = self.config['train']['loss_weights']['transf2map']
        self.w_map2transf = self.config['train']['loss_weights']['map2transf']
        self.path = '/data'

        print(f'Init Trainer ({time.time()-t_start}s)')

    def getModel(self, config: dict):
        """Loads the model specified in self.config
        """
        arch = self.config["network"]
        print(f"network architecture: {arch}")
        self.encoder_model = network.Network(
            config['network']['encoder_blocks'])
        self.decoder_model = network.Network(
            config['network']['decoder_blocks'])
        print(self.encoder_model)
        print(self.decoder_model)

    def loadModel(self, best: bool = True, device='cuda:0', out_dir=None):
        if out_dir is None:
            out_dir = self.config["network"]['out_dir']
        enc_path = out_dir+self.experiment_id+'/enc'
        dec_path = out_dir+self.experiment_id+'/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        print("load", enc_path, ",", dec_path)
        if(os.path.isfile(enc_path) and os.path.isfile(dec_path)):
            # if(os.path.isfile(model_file)):
            self.encoder_model.load_state_dict(torch.load(
                enc_path, map_location=lambda storage, loc: storage))

            self.decoder_model.load_state_dict(torch.load(
                dec_path, map_location=lambda storage, loc: storage))
            self.encoder_model.to(device)
            self.decoder_model.to(device)
        else:
            print(10*'!', 'Cannot load model', 10*'!')

    def saveModel(self, best: bool = False):
        out_dir = self.config["network"]['out_dir']
        enc_path = out_dir+self.experiment_id+'/enc'
        dec_path = out_dir+self.experiment_id+'/dec'
        enc_path += '_best.pth' if best else '.pth'
        dec_path += '_best.pth' if best else '.pth'
        torch.save(self.encoder_model.state_dict(), enc_path)
        torch.save(self.decoder_model.state_dict(), dec_path)

    def saveYaml(self, out_dir="network_files/"):
        config_path = out_dir+self.experiment_id+'/'+self.experiment_id+'.yaml'
        if not os.path.exists(out_dir+self.experiment_id):
            os.makedirs(out_dir+self.experiment_id, exist_ok=True)
        with open(config_path, 'w') as f:
            saver = yaml.YAML()
            saver.dump(self.config, f)

    def test(self, best=True):
        # TEST
        return self.evaluate(self.submaps.getTestSet(),
                             load_model=True,
                             best_model=best,
                             reference_points='map',
                             compute_memory=True, evaluate=True)

    def getNetworkParams(self):
        return list(self.encoder_model.parameters()) + list(self.decoder_model.parameters())

    def getScheduler(self, optimizer, len_data_loader, batch_size):
        number_epochs = self.config['train']['max_epochs']
        steps_per_epoch = int(len_data_loader / batch_size)
        max_lr = self.config["train"]["optimizer"]["max_lr"]
        div_factor = max_lr / \
            self.config["train"]["optimizer"]["start_lr"]
        final_div_factor = self.config["train"]["optimizer"]["start_lr"] / \
            self.config["train"]["optimizer"]["end_lr"]
        pct_start = self.config["train"]['optimizer']["pct_incr_cycle"]
        anneal_strategy = self.config["train"]['optimizer']["anneal_strategy"]
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=steps_per_epoch, epochs=number_epochs, pct_start=pct_start, anneal_strategy=anneal_strategy, div_factor=div_factor, final_div_factor=final_div_factor)

    def getLogWriter(self, logdir):
        if(os.path.isdir(logdir)):
            filelist = [f for f in os.listdir(logdir)]
            for f in filelist:
                os.remove(os.path.join(logdir, f))
        return SummaryWriter(logdir)

    def train(self, verbose=True):
        ###### Setup ######
        if self.config['train']['load_pretrained']:
            self.loadModel(best=True)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder_model.to(self.device)
        self.decoder_model.to(self.device)
        number_epochs = self.config['train']['max_epochs']
        output_path = self.config['network']['out_dir']
        writer = self.getLogWriter(
            output_path+'log/'+self.experiment_id+"/")
        batch_size = min(
            (self.config["train"]["batch_size"], self.submaps.getTrainSize()))

        ###### Init optimizer ########
        lr = self.config["train"]["optimizer"]["max_lr"]
        optimizer = optim.Adam(self.getNetworkParams(), lr=lr, amsgrad=False)
        scheduler = self.getScheduler(
            optimizer, self.submaps.getTrainSize(), batch_size)
        self.saveYaml(out_dir=output_path)

        time_start = time.time()
        optimizer.zero_grad()
        r_batch = 0
        best_loss = 1e10
        n_pct_time = time.time()

        validation_it = 0
        nr_batches = int(self.submaps.getTrainSize()/batch_size)
        ####################################
        ####### Start training #############
        ####################################
        print('Start Training: #self.submaps: %d, #batches: %f' %
              (self.submaps.getTrainSize(), nr_batches))
        for epoch in range(number_epochs):
            running_loss = 0
            batch = 0
            for i, input_dict in enumerate(self.submaps.getTrainSet()):
                if i >= (nr_batches * batch_size):  # drop last batch
                    continue
                ######## Preprocess #######
                input_dict['points'] = input_dict['points'].to(self.device)
                input_dict['features'] = input_dict['features'].to(self.device)
                input_points = input_dict['points']

                ####### Encoding and decoding #########
                t1 = time.time()
                out_dict = self.encoder_model(input_dict.copy())
                out_dict = self.decoder_model(out_dict)
                translation = out_dict['features'][:, :3]

                samples = out_dict['points']
                samples_transf = samples+translation
                #####################
                ###### Loss #########
                #####################
                loss = self.getTrainLoss(input_points, samples, translation)
                loss += loss_handler.linDeconvRegularizer(
                    self.decoder_model,
                    weight=self.config['train']['loss_weights']['upsampling_reg'],
                    gt_points=input_points)

                # print(loss)
                loss.backward()
                running_loss += loss.item()
                ########################################
                ######### Gradient Accumulation ########
                ########################################
                if ((i % batch_size) == (batch_size-1)):
                    r_batch += 1
                    batch += 1
                    optimizer.step()

                    optimizer.zero_grad()
                    running_loss /= batch_size
                    scheduler.step()

                    curr_lr = scheduler.get_lr()[0]

                    # Write Log
                    writer.add_scalar('learning loss',
                                      running_loss,
                                      r_batch)
                    writer.add_scalar('learning rate',
                                      curr_lr,
                                      r_batch)
                    if verbose:
                        print('[%d, %5d] loss: %.5f, time: %.1f lr: %.5f' %
                              (epoch + 1, batch, running_loss, time.time()-time_start, curr_lr))

                    time_start = time.time()
                    self.saveModel(best=False)

                    running_loss = 0

            #############################
            ##### validation ############
            #############################
            if (epoch % self.config['train']['validation']['report_rate']) is (self.config['train']['validation']['report_rate']-1):
                ts_val = time.time()
                valid_dict = self.evaluate(
                    dataloader=self.submaps.getValidSet())
                chamf_dist = valid_dict['reconstruction_error']
                writer.add_scalar('v: chamfer distance',
                                  chamf_dist,
                                  r_batch)
                if chamf_dist < best_loss:
                    self.saveModel(best=True)
                    best_loss = chamf_dist
                if verbose:
                    print('[%d, valid] rec_err: %.5f, time: %.1f' %
                          (epoch + 1, chamf_dist, time.time()-ts_val))

                validation_it += 1
            ###########################################
            ##### verbose every 10 percent ############
            ###########################################
            if(pcu.isEveryNPercent((epoch), max_it=number_epochs, percent=10)):
                n_pct = (epoch+1)/number_epochs*100
                time_est = (time.time() - n_pct_time)/n_pct*(100-n_pct)
                print("%4d%s in %ds, estim. time left: %ds (%dmin), best loss: %.5f" % (
                    n_pct, "%", time.time() - n_pct_time, time_est, time_est/60, best_loss))

    def getTrainLoss(self, gt_points: torch.tensor, samples, translations,):
        loss = torch.tensor(
            0.0, dtype=torch.float32, device=self.device)  # init loss
        samples_transf = samples + translations

        # Chamfer Loss between input and samples+T
        d_map2transf, d_transf2map, idx3, idx4 = self.cham_loss(
            gt_points.unsqueeze(0), samples_transf.unsqueeze(0))
        loss += (self.w_map2transf * d_map2transf.mean() +
                 self.w_transf2map * d_transf2map.mean())
        return loss

    def evaluate(self, dataloader,
                 load_model=False,
                 best_model=False,
                 reference_points='points',
                 compute_memory=False,
                 evaluate=False):
        loss_evaluator = evaluator.Evaluator(self.config)
        self.encoder_model.eval()
        self.decoder_model.eval()
        t_encoding = []
        t_decoding = []
        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.encoder_model.to(self.device)
                self.decoder_model.to(self.device)
                print('loaded best:', best_model)
            for i, input_dict in enumerate(tqdm(dataloader)):
                map_idx = input_dict['idx']
                # print('map:', map_idx)
                scale = input_dict['scale']
                input_dict['features'] = input_dict['features'].to(self.device)
                input_dict['points'] = input_dict['points'].to(self.device)
                
                ####### Cast to float16 if necessary #######
                t_enc_start = time.time()
                out_dict = self.encoder_model(input_dict.copy())
                t_enc_end = time.time()
                t_encoding.append(t_enc_end - t_enc_start)
                if self.config['evaluation']['float16']:
                    out_dict['points'] = out_dict['points'].half().float()
                    out_dict['features'] = out_dict['features'].half().float()
                ####### Compute  Memory #######
                if compute_memory:
                    nbytes = 2 if self.config['evaluation']['float16'] else 4
                    mem = (out_dict['points'].numel() +
                           out_dict['features'].numel())*nbytes
                    loss_evaluator.eval_results['memory'].append(mem)
                    loss_evaluator.eval_results['bpp'].append(
                        mem/input_dict['map'].shape[0]*8)
                ############# Decoder ##################
                t_dec_start = time.time()
                out_dict = self.decoder_model(out_dict)
                t_dec_end = time.time()
                t_decoding.append(t_dec_start - t_dec_end)

                translation = out_dict['features'][:, :3]
                samples = out_dict['points']
                samples_transf = samples+translation

                ###################################
                gt_points = input_dict[reference_points].to(self.device)

                # Scale to metric space
                samples_transf *= scale
                samples *= scale
                translation *= scale
                gt_points *= scale

                # Save the result into file
                pcd = geometry.PointCloud()
                np_points = samples_transf.cpu().numpy().astype(np.float16)
                np_points = np.unique(np_points, axis=0)
                pcd.points = utility.Vector3dVector(np_points)
                filename_without_extension = os.path.basename(input_dict['filename']).split(".")[0]
                io.write_point_cloud(f'/data/test/decoded/{filename_without_extension}.pcd', pcd, write_ascii=True)

            #     reconstruction_error = loss_evaluator.chamferDist(
            #         gt_points=gt_points, source_points=samples_transf)
            #     loss_evaluator.eval_results['mapwise_reconstruction_error'].append(
            #         reconstruction_error.item())

            #     if evaluate:  # Full evaluation for testing
            #         feat_ind = np.cumsum(
            #             [0]+self.config['grid']['feature_dim'])
            #         normal_idx = pcu.findList(
            #             self.config['grid']['features'], value='normals')
            #         normal_idx = (feat_ind[normal_idx], feat_ind[normal_idx+1])
            #         gt_normals = input_dict[reference_points +
            #                                 '_attributes'][:, normal_idx[0]:normal_idx[1]].cuda()
            #         loss_evaluator.evaluate(
            #             gt_points=gt_points, source_points=samples_transf, gt_normals=gt_normals)
            # chamfer_dist = loss_evaluator.getRunningLoss()
            # loss_evaluator.eval_results['reconstruction_error'] = chamfer_dist
        self.encoder_model.train()
        self.decoder_model.train()
        print(f"Encoding time: {np.mean(t_encoding)} | Decoding time: {np.mean(t_decoding)}")

        return loss_evaluator.eval_results
    
    def construct(self, submap, nr_points=200000, init_ones=True, feature_cols=[]):
        out_dict = {'idx': torch.tensor(0)}
        submap.initialize()
        if submap.cols <= 3:
            out_dict['points'] = submap.getRandPoints(
                nr_points, seed=0)
            out_dict['map'] = submap.getPoints()
            if init_ones:
                out_dict['features'] = np.ones(
                    (out_dict['points'].shape[0], 1), dtype='float32')
                out_dict['features'] = torch.from_numpy(out_dict['features'])
        else:
            points = submap.getRandPoints(
                nr_points, seed=0)
            out_dict['points'] = points[:, :3]
            out_dict['points_attributes'] = points[:, 3:]
            map_ = submap.getPoints()
            out_dict['map'] = map_[:, :3]
            out_dict['map_attributes'] = map_[:, 3:]
            if init_ones:
                out_dict['features'] = np.hstack(
                    (np.ones((points.shape[0], 1), dtype='float32'), out_dict['points_attributes'][:, feature_cols]))
            else:
                out_dict['features'] = out_dict['points_attributes'][:, feature_cols]
        out_dict['features_original'] = out_dict['features']
        out_dict['scale'] = submap.getScale()
        out_dict['filename'] = submap.getFileName()
        return out_dict


    def encode_single(self, path, load_model=False, best_model=False,
                      reference_points='points'):
        submap = SubMap(path, file_cols=3,
                            on_the_fly=True,grid_size=16)

        input_dict = self.construct(submap = submap)
        #input_dict is the source after loading data
        self.encoder_model.eval()
        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.encoder_model.to(self.device)
                print('loaded best:', best_model)
            # Convert all numeric array and numeri to tensor
            input_dict['points'] = torch.from_numpy(input_dict['points'])
            input_dict['map'] = torch.from_numpy(input_dict['map'])
            input_dict['scale'] = torch.tensor(input_dict['scale'])
            
            # Move to gpu
            input_dict['features'] = input_dict['features'].to(self.device)
            input_dict['points'] = input_dict['points'].to(self.device)
            
            # input_dict['features'] = torch.from_numpy(input_dict['features'])
            # input_dict['features_original'] = torch.from_numpy(input_dict['features_original'])

            t_enc_start = time.time()
            out_dict = self.encoder_model(input_dict.copy())
            t_enc_end = time.time()

            if self.config['evaluation']['float16']:
                out_dict['points'] = out_dict['points'].half().float()
                out_dict['features'] = out_dict['features'].half().float()
            
            saved_dict = {}
            saved_dict['points'] = out_dict['points']
            saved_dict['features'] = out_dict['features']
            saved_dict['scale'] = out_dict['scale']
        self.encoder_model.train()
        return saved_dict, path
    
    def decode_single(self, path, load_model=False, best_model=False,
                      reference_points='points'):
        self.decoder_model.eval()
        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.decoder_model.to(self.device)
            
            with open(f"{path}", "rb") as f:
                    out_dict = pickle.load(f)
            out_dict = self.decoder_model(out_dict)
            translation = out_dict['features'][:, :3]
            samples = out_dict['points']
            samples_transf = samples+translation

            # Scale to metric space
            scale = out_dict['scale']
            samples_transf *= scale
            samples *= scale
            translation *= scale

        return samples_transf, path

    # TODO: check whether this method is also needed for training
    # if yes, then create a new method that takes only 1 point cloud as parameter
    def create_folder(self, type, max_nr_pts, size):
        size = 'x'.join(map(str, size))
        path = f"{self.path}/{type}/{max_nr_pts}/{size}"
        return path

    def encode(self, load_model=False, best_model=False,
                 reference_points='points', compute_memory=True):
        
        loss_evaluator = evaluator.Evaluator(self.config)
        dataloader = self.submaps.getTestSet()
        t_encoding = []
        self.encoder_model.eval()
        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.encoder_model.to(self.device)
                print('loaded best:', best_model)

            # TODO: remove for loop
            for i, input_dict in enumerate(tqdm(dataloader)):
                map_idx = input_dict['idx']
                input_dict['features'] = input_dict['features'].to(self.device)
                input_dict['points'] = input_dict['points'].to(self.device)
                
                ####### Cast to float16 if necessary #######
                t_enc_start = time.time()
                out_dict = self.encoder_model(input_dict.copy())
                t_enc_end = time.time()
                t_encoding.append(t_enc_end - t_enc_start)
                if self.config['evaluation']['float16']:
                    out_dict['points'] = out_dict['points'].half().float()
                    out_dict['features'] = out_dict['features'].half().float()
                
                saved_dict = {}
                saved_dict['points'] = out_dict['points']
                saved_dict['features'] = out_dict['features']
                saved_dict['scale'] = out_dict['scale']
                saved_dict['normalizer'] = out_dict['normalizer']
                saved_dict['encoding_speed'] = t_enc_end - t_enc_start
                # translation = out_dict['features'][:, :3]
                # samples = out_dict['points']
                # samples_transf = samples+translation

                # torch.save(samples_transf, f"/data/test/encoded/{i}.bin")
                filename_without_extension = os.path.basename(input_dict['filename']).split(".")[0]
                write_path = self.create_folder(
                    "encoded", 
                    self.config['train']['max_nr_pts'],
                    self.config['grid']['size']
                )

                # Ensure the directory exists or create it
                if not os.path.exists(write_path):
                    os.makedirs(write_path)

                print(f"Saved to {write_path}")
                with open(f"{write_path}/{filename_without_extension}.bin", "wb") as file:
                    pickle.dump(saved_dict, file)

                ####### Compute  Memory #######
                if compute_memory:
                    nbytes = 2 if self.config['evaluation']['float16'] else 4
                    mem = (out_dict['points'].numel() +
                           out_dict['features'].numel())*nbytes
                    loss_evaluator.eval_results['memory'].append(mem)
                    loss_evaluator.eval_results['bpp'].append(
                        mem/input_dict['map'].shape[0]*8)
                #############$$$$$$$$$##################
        self.encoder_model.train()
        print(f"Encoding mean time: {np.mean(t_encoding)}")
        print(f"Bpp: {np.mean(loss_evaluator.eval_results['bpp'])}")
        return saved_dict

    # Reading JSON data from a file
    def read_json_file(self, filename):
        with open(filename, 'r') as file:
            data = json.load(file)
        return data

    # Writing JSON data to a file
    def write_json_file(self, data, filename):
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def decode(self, load_model=False, best_model=False,
                 reference_points='points', compute_memory=False, evaluate=False):
        
        loss_evaluator = evaluator.Evaluator(self.config)
        t_decoding = []
        self.decoder_model.eval()

        # Get a list of all files in the directory
        read_path = self.create_folder(
            "encoded", 
            self.config['train']['max_nr_pts'],
            self.config['grid']['size']
        )
        write_path = self.create_folder(
            "decoded", 
            self.config['train']['max_nr_pts'],
            self.config['grid']['size']
        )
        files = os.listdir(read_path)
        encoded_bin_files = [os.path.join(read_path, file) for file in files if file.endswith(".bin")]

        performance_file_name = f"{write_path}/performance.json"
        if not os.path.exists(performance_file_name):
            os.makedirs(os.path.dirname(performance_file_name), exist_ok=True)
            performance_dump = {}
        else:
            performance_dump = self.read_json_file(performance_file_name)

        with torch.no_grad():
            if load_model:
                self.loadModel(best=best_model)
                self.decoder_model.to(self.device)
                print('loaded best:', best_model)
            
            for i, file in enumerate(tqdm(encoded_bin_files)):
                # out_dict['points'] = torch.load(file)
                with open(file, "rb") as f:
                    out_dict = pickle.load(f)
                t_dec_start = time.time()
                out_dict = self.decoder_model(out_dict)
                t_dec_end = time.time()
                t_decoding.append(t_dec_end - t_dec_start)

                translation = out_dict['features'][:, :3]
                samples = out_dict['points']
                samples_transf = samples+translation

                # Scale to metric space
                scale = out_dict['scale']
                # samples_transf *= scale

                samples_transf = (samples_transf * out_dict['normalizer']['dif'].cuda())+out_dict['normalizer']['min'].cuda()
                samples *= scale
                translation *= scale

                # Save the result into file
                pcd = geometry.PointCloud()
                np_points = samples_transf.cpu().numpy().astype(np.float16)
                np_points = np.unique(np_points, axis=0)

                pcd.points = utility.Vector3dVector(np_points)
                filename_without_extension = os.path.basename(file).split(".")[0]
                io.write_point_cloud(f'{write_path}/{filename_without_extension}.pcd', pcd, write_ascii=True)

                name = os.path.basename(filename_without_extension)
                performance_dump[name] = {
                    'encoding_time': out_dict['encoding_speed'],
                    'decoding_time': t_dec_end - t_dec_start
                }
            self.write_json_file(performance_dump, performance_file_name)

        self.decoder_model.train()
        print(f"Decoding mean time: {np.mean(t_decoding)}")

    def encodeDecode(self, input_dict, float_16=True):
        map_idx = input_dict['idx']
        # print('map:', map_idx)
        scale = input_dict['scale']
        input_dict['features'] = input_dict['features'].to(self.device)
        input_dict['points'] = input_dict['points'].to(self.device)

        ####### Cast to float_16 if necessary #######
        out_dict = self.encoder_model(input_dict.copy())
        if float_16:
            out_dict['points'] = out_dict['points'].half().float()
            out_dict['features'] = out_dict['features'].half().float()
        nr_emb = out_dict['points'].shape
        ############# Decoder ##################
        out_dict = self.decoder_model(out_dict)
        translation = out_dict['features'][:, :3]
        samples = out_dict['points']
        samples_transf = samples+translation

        samples_transf *= scale
        return samples_transf, nr_emb


if __name__ == "__main__":
    print('Hello')
    parser = argparse.ArgumentParser("./sample_net_trainer.py")
    parser.add_argument(
        '--config', '-cfg',
        type=str,
        required=False,
        default='config/depoco.yaml',
        help='configitecture yaml cfg file. See /config/config for example',
    )
    FLAGS, unparsed = parser.parse_known_args()

    print('passed flags')
    config = yaml.safe_load(open(FLAGS.config, 'r'))
    print('loaded yaml flags')
    trainer = DepocoNetTrainer(config)
    print('initialized  trainer')
    trainer.train(verbose=True)
