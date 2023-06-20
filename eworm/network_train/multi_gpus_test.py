import os
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train.artificial_circuit import ArtificialCircuit
from eworm.network_train.network_inception import single_nrn_preparation, visualize_circuit_history
import time
import torch
import numpy as np
import _pickle as pickle
import numpy.random as nr
import os.path as path
import matplotlib.pyplot as plt
import timeit
plt.switch_backend('Agg')


def test_statement(devices, abs_circuit, weight_config, model_config, squeeze_input, io_config, pipeline_test):
    art_circuit = ArtificialCircuit(abs_circuit, weight_config, model_config, devices)
    _ = art_circuit(torch.tensor(squeeze_input, dtype=torch.float32),
                    io_config["input_cells"], io_config["output_cells"], pipeline_test=pipeline_test)


if __name__ == "__main__":
    dataset_squeeze, addtional_squeeze = 4, 5
    inoise = 3
    circuit_name = "command_circuit"
    config_name = "config_small"
    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega1"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    io_config = {
        "input_cells": ["AVBL", "AVAL", "AVAR", "PVCL", "AVBR"],
        # "input_cells": ["AVAR", "AVBR"],
        "output_cells": []}
    for vis_config in (circuit_name, config_name, task_name, dataset_name, io_config):
        print(vis_config)
    config, _, del_circuit, pretrain_dir = single_nrn_preparation(config_name, io_config, task_name,
                                                                            dataset_name, circuit_name)
    nr.seed(42)
    fake_weight_config = {
        "syn": (0.1, 10),
        "gj": (1e-4, 6e-4),
        "inh_prob": 0.3}
    new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
    del_circuit.update_connections(new_weights)
    abs_circuit = transform.detailed2abstract(del_circuit)
    for connection in abs_circuit.connections:
        if connection.pre_segment is not None:
            print(connection.pre_cell.name, connection.post_cell.name, connection.weight, connection.category)
    data_config = {
        "factory_name": "random_data_factory",
        "args": {
            "window_range": (10, 1000),
            "volt_range": (-95, 35),
            "noise_settings": ((80, 60), (20, 20), (180, 60), (40, 15)),
            "reverse_noise_setting": None}}
    weight_config = {
        "syn": (0.1, 10),
        "gj": (1e-4, 1e-3),
        "inh_prob": 0.3}
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    batch_size = 64
    squeeze_input = []
    for _ in range(batch_size):
        input_traces = eval("data_factory." + data_config['factory_name'])(
            num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
        squeeze_input.append(data_factory.squeeze_trace(input_traces, dataset_squeeze*addtional_squeeze))
    squeeze_input = np.stack(squeeze_input, axis=0)

    # inference
    num_repeat = 10
    gpus = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    print(gpus)
    # warm up
    statement = "test_statement(gpus, abs_circuit, weight_config, model_config, squeeze_input, io_config, False)"
    _ = timeit.repeat(statement, number=1, repeat=2, globals=globals())

    devices = [torch.device("cpu")]
    statement = "test_statement(devices, abs_circuit, weight_config, model_config, squeeze_input, io_config, False)"

    cpu_run_times = timeit.repeat(statement, number=1, repeat=num_repeat, globals=globals())
    cpu_mean, cpu_std = np.mean(cpu_run_times), np.std(cpu_run_times)
    print(f"Cpu mean:{cpu_mean}, std:{cpu_std}")

    cuda_mean, cuda_std = [], []
    for cuda_index in range(torch.cuda.device_count()):
        devices = [torch.device(f"cuda:{gpu_index}") for gpu_index in range(cuda_index+1)]
        statement = "test_statement(devices, abs_circuit, weight_config, model_config, squeeze_input, io_config, False)"

        gpu_run_times = timeit.repeat(statement, number=1, repeat=num_repeat, globals=globals())
        cuda_mean.append(np.mean(gpu_run_times))
        cuda_std.append(np.std(gpu_run_times))
        print(f"Gpu#{cuda_index} mean:{cuda_mean[-1]}, std:{cuda_std[-1]}")

    real_cuda_mean, real_cuda_std = [], []
    for cuda_index in range(torch.cuda.device_count()):
        devices = [torch.device(f"cuda:{gpu_index}") for gpu_index in range(cuda_index+1)]
        statement = "test_statement(devices, abs_circuit, weight_config, model_config, squeeze_input, io_config, True)"

        gpu_run_times = timeit.repeat(statement, number=1, repeat=num_repeat, globals=globals())
        real_cuda_mean.append(np.mean(gpu_run_times))
        real_cuda_std.append(np.std(gpu_run_times))
        print(f"Real Gpu#{cuda_index} mean:{real_cuda_mean[-1]}, std:{real_cuda_std[-1]}")

    means, stds = cuda_mean + real_cuda_mean + [cpu_mean], cuda_std + real_cuda_std + [cpu_std]
    labels = [f"{cuda_index+1} GPU Seq" for cuda_index in range(torch.cuda.device_count())] +\
             [f"{cuda_index+1} GPU Par" for cuda_index in range(torch.cuda.device_count())] + ["CPU"]
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig("./tradeoff_par_6_no_computation.png", dpi=300)
    plt.close(fig)