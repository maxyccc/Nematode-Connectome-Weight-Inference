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


def random_circuit_target_test_factory(num, mask_rate, ref_circuit, fake_weight_config, config_name,
                                       data_config, model_config, sim_config, io_config):
    new_weights = func.circuit_weight_sample(fake_weight_config, ref_circuit)
    ref_circuit.update_connections(new_weights)
    del_circuit = transform.abstract2detailed(
        ref_circuit, func.load_json(path.join(path.dirname(__file__), "configs", config_name + ".json")))
    squeeze_input, squeeze_output = [], [[] for _ in range(len(del_circuit.connections))]
    total_squeeze = model_config['squeeze']
    for data_idx in range(num):
        print(f"data generating... {data_idx + 1}/{num} ")
        input_traces = eval("data_factory." + data_config['factory_name'])(
            num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
        _ = del_circuit.simulation(sim_config, input_traces, io_config["input_cells"], io_config["output_cells"],
                                   make_history=True)
        for cnt_idx, connection in enumerate(del_circuit.connections):
            squeeze_output[cnt_idx].append(data_factory.squeeze_trace(connection.history, total_squeeze))
        squeeze_input.append(data_factory.squeeze_trace(input_traces, total_squeeze))
    tensor_input = torch.tensor(np.stack(squeeze_input, axis=0), dtype=torch.float32)
    tensor_output = []
    for cnt_idx, connection in enumerate(del_circuit.connections):
        tensor_output.append(torch.tensor(np.stack(squeeze_output[cnt_idx], axis=0), dtype=torch.float32))

    history_mask = np.random.rand(len(del_circuit.connections)) < mask_rate
    test_input = eval("data_factory." + data_config['factory_name'])(
        num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
    _ = del_circuit.simulation(sim_config, test_input, io_config["input_cells"], io_config["output_cells"],
                               make_history=True)
    test_squeeze_input = torch.tensor(data_factory.squeeze_trace(test_input, total_squeeze),
                                      dtype=torch.float32).unsqueeze(0)

    return tensor_input, tensor_output, history_mask, del_circuit, test_squeeze_input, test_input


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    dataset_squeeze, addtional_squeeze = 20, 1
    inoise = 0
    # circuit_name = "command_circuit"
    # config_name = "config_small"
    # circuit_name = "head20"
    # config_name = "config_head20"
    circuit_name = "full109"
    config_name = "config_full109"

    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    # io_config = {
    #     "input_cells": ["AVBL", "AVBR", 'AVAL', 'AVAR'],
    #     "output_cells": ["AVBL", "AVBR", 'AVAL', 'AVAR', 'PVCL', 'PVCR']}

    # io_config = {
    #     "input_cells": ["AWAL", "AWAR", 'AWCL', 'AWCR'],
    #     "output_cells": ["SAADL", "SAADR", "SAAVL", "SAAVR"]}
    io_config = {
        "input_cells": [
            "AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR", "ALNL", "ALNR", "PLML", "PHAL", "PHAR", "URYDL", "URYDR",
            "URYVL", "URYVR"],
        "output_cells": [
            "AWAL", "AWAR", "AWCL", "AWCR", "ASKL", "ASKR", "ALNL", "ALNR", "PLML", "DVA", "PHAL", "PHAR", "URYDL",
            "URYDR", "URYVL", "URYVR", "AIYL", "AIYR", "AIAL", "AIAR", "AIZL", "AIZR", "RIS", "ALA", "AVFL", "AVFR",
            "PVNL", "PVNR", "DVB", "RIBL", "RIBR", "AIBL", "AIBR", "SAADL", "SAADR", "SAAVL", "SAAVR", "DVC", "RIML",
            "RIMR", "AVEL", "AVER", "RID", "AVBL", "AVBR", "AVAL", "AVAR", "PVCL", "PVCR", "RMEL", "RMER", "RMED",
            "RMEV", "RMDDL", "RMDDR", "RMDL", "RMDR", "RMDVL", "RMDVR", "RIVL", "RIVR", "SABD", "SABVL", "SABVR",
            "SMDDL", "SMDDR", "SMDVL", "SMDVR", "SMBDL", "SMBDR", "SMBVL", "SMBVR", "SIADL", "SIADR", "SIAVL", "SIAVR",
            "DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09", "PDA", "DB01", "DB02", "DB03",
            "DB04", "DB05", "DB06", "DB07", "AS10", "DD01", "DD02", "DD03", "DD04", "DD05", "DD06", "VA01", "VA02",
            "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12", "VB01", "VB02", "VB03",
            "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11", "VD01", "VD02", "VD03", "VD04", "VD05",
            "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13"]}
    for vis_config in (circuit_name, config_name, task_name, dataset_name, io_config):
        print(vis_config)
    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, io_config, task_name, dataset_name, circuit_name)
    print("Loading Complete!")

    # weight_config = {
    #     "syn": (5e-2, 5e-1),
    #     "gj": (1e-5, 1e-4),
    #     "inh_prob": 0.6}
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    fake_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.7, }
    # data_config = {
    #     "factory_name": "special_wave_data_factory",
    #     "args": {
    #         "window_range": (400, 1000),
    #         "volt_range": (-30, -35),
    #         "wave_amp": (40, 45),
    #         "noise_settings": ((20, 60), (5, 20)),
    #         "fire_rate": 0.3,
    #         "wave_type": 'sin',
    #         "fire_pattern": "random",
    #         "op_type": 'or'}}
    data_config = {
        "factory_name": "random_data_factory",
        "args": {
            "window_range": (10, 1000),
            "volt_range": (-95, 45),
            "noise_settings": ((40, 5), (180, 60)),
            "reverse_noise_setting": None}}
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    batch_size = 32
    mask_rate = 0.2
    test_name = f"diff_weight_approach_test_bsz{batch_size}_mask{mask_rate}"
    print(test_name)

    nr.seed(43)
    train_input, train_target, history_mask, test_del_circuit, test_squeeze_input, test_input = \
        random_circuit_target_test_factory(batch_size, mask_rate, abs_circuit, fake_weight_config, config_name,
                                           data_config, model_config, sim_config, io_config)
    print("Dataset Generated!")

    # new start
    new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
    new_abs_circuit = transform.detailed2abstract(del_circuit)
    new_abs_circuit.update_connections(new_weights)
    gpus = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    print(gpus)
    new_art_circuit = artificial_circuit.ArtificialCircuit(new_abs_circuit, weight_config, model_config, gpus)

    working_directory = path.join(path.dirname(__file__), "log", task_name, test_name)
    os.makedirs(working_directory, exist_ok=True)
    network_inception.visualize_circuit_history(test_del_circuit, sim_config,
                                                path.join(working_directory, 'test_gt.jpg'))
    new_art_circuit(train_input, io_config["input_cells"], io_config["output_cells"])
    for cnt_idx, connection in enumerate(new_art_circuit.circuit.connections):
        if history_mask[cnt_idx]:
            connection.history = train_target[cnt_idx]
        elif connection.pre_segment is not None:
            connection.history = None
    network_inception.visualize_circuit_history(new_art_circuit, sim_config,
                                                path.join(working_directory, 'train_gt.jpg'))
    print("Training Start!")

    # main part train process
    torch.backends.cudnn.benchmark = True
    new_art_circuit.mode_switch("train")
    train_config = {
        "loss": "MSE",
        "lr": 5e-3,
        "num_epoch": 1025,
        "window_len": 200,
    }
    optimizer = torch.optim.Adam(new_art_circuit.fetch_meta_weights(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss() if train_config['loss'] == "MSE" else torch.nn.L1Loss()
    half_window = int(train_config['window_len'] // 2)
    loss_rec, loss_mean_rec, num_window = [], [], int(np.ceil(train_input.shape[-1] / half_window)) - 1
    for epoch in range(train_config['num_epoch']):
        avg_loss = 0

        for window_idx in range(num_window):
            _ = new_art_circuit(train_input[..., window_idx * half_window:(window_idx + 2) * half_window],
                                io_config["input_cells"], io_config["output_cells"])
            epoch_loss = 0
            optimizer.zero_grad()
            for connection_id, connection in enumerate(new_art_circuit.circuit.connections):
                if history_mask[connection_id]:
                    loss = criterion(connection.history, train_target[connection_id][..., window_idx * half_window:(window_idx + 2) * half_window].to(connection.history.device))
                    epoch_loss += loss.cpu()
            epoch_loss.backward()
            optimizer.step()
            avg_loss += epoch_loss.cpu()
        avg_loss /= num_window
        print(f"Epoch {epoch}/{train_config['num_epoch']} Loss {float(avg_loss)}")
        loss_rec.append(float(avg_loss))
        loss_mean_rec.append(np.mean(loss_rec[-100:]))
        if epoch & (epoch - 1) == 0:
            with torch.no_grad():
                network_inception.artificial_circuit_inception(
                    new_art_circuit, test_squeeze_input, test_input, config_name, io_config, sim_config, model_config,
                    path.join(working_directory, f"epoch #{epoch}"), 'ad', "circuit_ckp", prefix="test_")
                new_art_circuit(train_input, io_config["input_cells"], io_config["output_cells"])
                network_inception.visualize_circuit_history(new_art_circuit, sim_config,
                                                            path.join(working_directory, f"epoch #{epoch}",
                                                                      'train_ckp.jpg'))
            vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"))









