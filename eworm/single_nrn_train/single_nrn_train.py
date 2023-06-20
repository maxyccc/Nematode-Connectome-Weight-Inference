import numpy
import torch
import time
import os
import re
import sys
import argparse
from functools import reduce
import os.path as path
import _pickle as pickle
import numpy.random as nr
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train.datamaker import *
from eworm.single_nrn_train.trainer import *


parser = argparse.ArgumentParser()
parser.add_argument('--cell_index', default=1, type=int,
                    help='index of neuron working on')
parser.add_argument('--task_name', default=None, type=str,
                    help="task name")
parser.add_argument('--dataset_name', default=None, type=str,
                    help="dataset name")
parser.add_argument('--config_dir', default=None, type=str,
                    help="config .json file path")
parser.add_argument('--circuit_dir', default=None, type=str,
                    help="circuit .pkl file path")
parser.add_argument('--device', default=0, type=int,
                    help="specific device to run")
args = parser.parse_args()

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    nr.seed(42)
    """
    ######### PART 0 PREPARATION #########
    """
    data_squeeze = int(re.search("\\d*x", args.dataset_name).group()[:-1])
    model_squeeze = int(int(re.search("\\d*x", args.task_name).group()[:-1])/data_squeeze)
    generate_config = {
        "total_file": 32,
        "data_per_file": 1000,
        "thread_num": 32,
        'squeeze_ratio': data_squeeze}
    data_config = {
        "factory_name": "random_data_factory",
        "args": {
            "window_range": (10, 1000),
            "volt_range": (-95, 45),
            "noise_settings": ((40, 5), (180, 60)),
            "reverse_noise_setting": None,
            "sparse_active": True}}
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    sim_config = {"dt": 0.5, "tstop": 5000, "v_init": -65, "secondorder": 0}
    config = func.load_json(path.join(path.dirname(__file__), "config.json")) if args.config_dir is None else \
        func.load_json(args.config_dir)
    ref_circuit = pickle.load(open("./ref_circuit.pkl", "rb")) if args.circuit_dir is None else \
        pickle.load(open(args.circuit_dir, "rb"))
    cell = ref_circuit.cell(args.cell_index)
    cell_config = {
        "cell_index": cell.index,
        "cell_name": cell.name}
    for vis_config in (generate_config, data_config, weight_config, sim_config, cell_config):
        print(vis_config)
    print(f"Cell {cell.name}, in: {len(cell.pre_connections)}, out: {len(cell.post_connections)}, "
          f"segments:{len(cell.segments)}")

    """
    ######### PART I MAKE DATA #########
    """
    dataset_dir = path.join(path.dirname(__file__), "data", args.dataset_name)
    if not path.exists(path.join(dataset_dir, cell.name, f"{cell.name}_{generate_config['total_file'] - 1}.dat")):
        print(f"Make {cell.name} Dataset!")
        data_make(ref_circuit, cell_config, sim_config, config, dataset_dir,
                  generate_config, data_config, weight_config)
        print(f"{cell.name} Dataset Generated!")
    # exit()
    """
    ######### PART II TRAIN PROCESS #########
    """
    model_noise = int(re.search("\\d*inoise", args.task_name).group()[:-6])
    train_config = {
        "batch_size": 64,
        "n_cpu": 0,
        "num_epoch": 129,
        "lr": 1e-4,
        "window_size": 500,
        "squeeze": model_squeeze,
        "dataset_dir": dataset_dir,
        "device": f"cuda:{args.device}" if torch.cuda.is_available() else "cpu",
        "weight_config": weight_config,
        "loss_thresh": 0.2,
        "loss_type": "MSE",
        "noise": model_noise,
        "validation": False}
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": None,
        "args": {
            "n_layers": 1,
            "in_channel": len(cell.pre_connections),
            "out_channel": len(np.unique([post_cnt.pre_segment.index for post_cnt in cell.post_connections]))}}
    trainer_dir = path.join(path.dirname(__file__), "output", args.task_name)
    for vis_config in (train_config, model_config):
        print(vis_config)
    train(cell, train_config, model_config, trainer_dir)

    """
    ######### PART III VALIDATION PROCESS #########
    """
    train_config["validation"], train_config["num_epoch"] = True, 129
    train(cell, train_config, model_config, trainer_dir)
