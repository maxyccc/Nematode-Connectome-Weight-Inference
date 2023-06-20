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


def functional_test_factory(num, amp_rate, ref_circuit, config_name, data_config, model_config, sim_config, io_config):
    assert data_config['factory_name'] == 'special_wave_data_factory'
    del_circuit = transform.abstract2detailed(
        ref_circuit, func.load_json(path.join(path.dirname(__file__), "configs", config_name + ".json")))
    total_squeeze = model_config['squeeze']

    # make test dataset and history mask
    test_input, test_target, _ = eval("data_factory." + data_config['factory_name'])(
        num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], with_output=True, **data_config['args'])
    _ = del_circuit.simulation(sim_config, test_input, io_config["input_cells"], io_config["output_cells"],
                               make_history=True)
    assert test_target.shape[0] == 1
    test_target = np.repeat(test_target, int(amp_rate), axis=0)
    history_mask = np.zeros(len(del_circuit.connections), dtype=bool)
    while np.sum(history_mask) < test_target.shape[0]:
        cnt_idx = nr.choice(len(del_circuit.connections))
        choosen_cnt = del_circuit.connections[cnt_idx]
        if (choosen_cnt.pre_segment is not None) and (choosen_cnt.category == 'syn'):
            history_mask[cnt_idx] = True
    test_squeeze_input = torch.tensor(data_factory.squeeze_trace(test_input, total_squeeze),
                                      dtype=torch.float32).unsqueeze(0)

    for cnt_idx, connection in enumerate(del_circuit.connections):
        if history_mask[cnt_idx]:
            connection.history = test_target[0]
        elif connection.pre_segment is not None:
            connection.history = None

    # make train dataset
    squeeze_input, squeeze_output, squeeze_sign = [], [[] for _ in range(len(del_circuit.connections))],  [[] for _ in range(len(del_circuit.connections))]
    for data_idx in range(num):
        print(f"data generating... {data_idx+1}/{num} ")
        train_input, train_target, train_sign = eval("data_factory." + data_config['factory_name'])(
            num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], with_output=True, **data_config['args'])
        for cnt_idx, connection in enumerate(del_circuit.connections):
            if history_mask[cnt_idx]:
                squeeze_output[cnt_idx].append(data_factory.squeeze_trace(train_target[0], total_squeeze))
                squeeze_sign[cnt_idx].append(data_factory.squeeze_trace(train_sign[0], total_squeeze))
        squeeze_input.append(data_factory.squeeze_trace(train_input, total_squeeze))

    tensor_input = torch.tensor(np.stack(squeeze_input, axis=0), dtype=torch.float32)
    tensor_output, tensor_sign = [], []
    for cnt_idx, connection in enumerate(del_circuit.connections):
        if history_mask[cnt_idx]:
            tensor_output.append(torch.tensor(np.stack(squeeze_output[cnt_idx], axis=0), dtype=torch.float32))
            tensor_sign.append(torch.tensor(np.stack(squeeze_sign[cnt_idx], axis=0), dtype=torch.float32))
        else:
            tensor_output.append(None)
            tensor_sign.append(None)
    return tensor_input, tensor_output, tensor_sign, history_mask, del_circuit, test_squeeze_input, test_input


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

    for vis_config in (circuit_name, config_name, task_name, dataset_name):
        print(vis_config)
    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, task_name, dataset_name, circuit_name)
    io_config = config['io']
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
        "inh_prob": 0.6, }
    data_config = {
        "factory_name": "special_wave_data_factory",
        "args": {
            "window_range": (400, 500),
            "volt_range": (-60, -55),
            "wave_amp": (10, 15),
            "noise_settings": ((20, 60), (5, 20)),
            "fire_rate": 0.3,
            "wave_type": 'sin',
            "fire_pattern": "periodic",
            "op_type": 'or'}}
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    batch_size = 32
    amp_rate = 8
    test_name = f"functional_test_bsz{batch_size}_amp{amp_rate}_{data_config['args']['fire_pattern']}_{data_config['args']['op_type']}"
    print(test_name)

    nr.seed(43)
    train_input, train_target, train_sign, history_mask, test_del_circuit, test_squeeze_input, test_input = \
        functional_test_factory(batch_size, amp_rate, abs_circuit, config_name, data_config, model_config, sim_config, io_config)
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
    network_inception.visualize_circuit_history(test_del_circuit, sim_config, path.join(working_directory, 'test_gt.jpg'))
    new_art_circuit(train_input, io_config["input_cells"], io_config["output_cells"])
    for cnt_idx, connection in enumerate(new_art_circuit.circuit.connections):
        if history_mask[cnt_idx]:
            connection.history = train_target[cnt_idx]
        elif connection.pre_segment is not None:
            connection.history = None
    network_inception.visualize_circuit_history(new_art_circuit, sim_config, path.join(working_directory, 'train_gt.jpg'))
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
            _ = new_art_circuit(train_input[..., window_idx*half_window:(window_idx+2)*half_window],
                                io_config["input_cells"], io_config["output_cells"])
            epoch_loss = 0
            optimizer.zero_grad()
            for connection_id, connection in enumerate(new_art_circuit.circuit.connections):
                if history_mask[connection_id]:
                    history_sign = train_sign[connection_id][..., window_idx*half_window:(window_idx+2)*half_window]
                    loss = -1*torch.sum(connection.history*history_sign.to(connection.history.device))
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
                                                            path.join(working_directory, f"epoch #{epoch}", 'train_ckp.jpg'))
            vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"), log=False)









