import os
import sys
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train import *
import time
import torch
import numpy as np
import _pickle as pickle
import numpy.random as nr
import os.path as path
import matplotlib.pyplot as plt

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    nr.seed(45)
    seeds = nr.randint(0, 1000, 10)
    flag = 'weight'
    weight_default = 33
    input_default = 42

    # circuit_name = "AVB_simple"
    # config_name = "config_AVB_simple"
    circuit_name = "full109"
    config_name = "config_full109"
    # circuit_name = "head20"
    # config_name = "config_head20"
    # circuit_name = "command_circuit"
    # config_name = "config_small"
    config_dir = path.join(path.dirname(__file__), "configs", config_name + ".json")
    circuit_dir = path.join(path.dirname(__file__), "circuits")
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    sample_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.6}
    # data_config = {
    #     "factory_name": "special_wave_data_factory",
    #     "args": {
    #         "window_range": (400, 1000),
    #         "volt_range": (-30, -35),
    #         "wave_amp": (40, 45),
    #         "noise_settings": ((20, 60), (5, 20)),
    #         "fire_rate": 0.3,
    #         "wave_type": 'sin',
    #         "op_type": 'or'}}
    # data_config = {
    #     "factory_name": "random_data_factory",
    #     "args": {
    #         "window_range": (10, 1000),
    #         "volt_range": (-75, 35),
    #         "noise_settings": ((80, 60), (20, 20), (180, 60), (40, 15)),
    #         "reverse_noise_setting": None}}
    data_config = {
        "factory_name": "special_wave_data_factory",
        "args": {
            "window_range": (400, 1000),
            "volt_range": (-30, -35),
            "wave_amp": (40, 45),
            "noise_settings": ((20, 60), (5, 20)),
            "fire_rate": 0.3,
            "wave_type": 'sin',
            "fire_pattern": "random",
            "op_type": 'xor'}}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    dataset_squeeze, addtional_squeeze = 20, 1
    inoise = 0
    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, task_name, dataset_name, circuit_name)
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    io_config = config['io']

    # gpus = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    gpus = [torch.device("cuda:1")]
    print(gpus)
    if flag == "weight":
        seeds_pair = zip(seeds, [input_default for _ in range(len(seeds))])
        test_name = f"{circuit_name}_input_{input_default}"
    else:
        seeds_pair = zip([weight_default for _ in range(len(seeds))], seeds)
        test_name = f"{circuit_name}_weight_{weight_default}"
    for weight_seed, input_seed in seeds_pair:
        nr.seed(weight_seed)
        new_weights = func.circuit_weight_sample(sample_weight_config, del_circuit)
        abs_circuit.update_connections(new_weights)
        del_circuit = transform.abstract2detailed(abs_circuit, config)
        art_circuit = artificial_circuit.ArtificialCircuit(abs_circuit, weight_config, model_config, gpus)

        working_directory = path.join(path.dirname(__file__), "inspect", test_name)
        os.makedirs(working_directory, exist_ok=True)
        vis.visualize_circuit(abs_circuit, save_dir=path.join(working_directory, circuit_name + f"_weight_{weight_seed}_connectome.jpg"), layout="circular")

        nr.seed(input_seed)
        test_input = eval("data_factory." + data_config['factory_name'])(
            num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
        tensor_test_input = torch.tensor(data_factory.squeeze_trace(test_input, dataset_squeeze * addtional_squeeze),
                                         dtype=torch.float32).unsqueeze(0)
        print(test_input.shape, tensor_test_input.shape)
        _ = art_circuit(tensor_test_input, io_config["input_cells"], io_config["output_cells"])
        network_inception.visualize_circuit_history(
            art_circuit, sim_config, path.join(working_directory, f"{circuit_name}_weight_{weight_seed}_input_{input_seed}_art.jpg"))
        _ = del_circuit.simulation(sim_config, test_input, io_config["input_cells"], io_config["output_cells"],
                                   make_history=True)
        network_inception.visualize_circuit_history(
            del_circuit, sim_config, path.join(working_directory, f"{circuit_name}_weight_{weight_seed}_input_{input_seed}_del.jpg"))
