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


def cpg_test_factory(ref_circuit, muscle_list, config_name, data_config, model_config, sim_config, io_config):
    assert data_config['factory_name'] == 'special_wave_data_factory'
    del_circuit = transform.abstract2detailed(
        ref_circuit, func.load_json(path.join(path.dirname(__file__), "configs", config_name + ".json")))
    total_squeeze = model_config['squeeze']

    # make test dataset and history mask
    test_input, test_target, test_sign = eval("data_factory." + data_config['factory_name'])(
        num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], with_output=True,
        **data_config['args'])
    _ = del_circuit.simulation(sim_config, test_input, io_config["input_cells"], io_config["output_cells"],
                               make_history=True)
    assert test_target.shape[0] == 1
    test_target = np.repeat(test_target, len(muscle_list), axis=0)
    history_mask = np.zeros(len(del_circuit.connections), dtype=bool)
    for cnt_idx, connection in enumerate(del_circuit.connections):
        if (connection.post_segment is None) and (connection.pre_cell.name in muscle_list) and (connection.category == 'syn'):
            history_mask[cnt_idx] = True

    test_squeeze_input = torch.tensor(data_factory.squeeze_trace(test_input, total_squeeze),
                                      dtype=torch.float32).unsqueeze(0)

    for cnt_idx, connection in enumerate(del_circuit.connections):
        if history_mask[cnt_idx]:
            connection.history = test_target[0]
        elif connection.pre_segment is not None:
            connection.history = None
    test_sign = torch.tensor(data_factory.squeeze_trace(test_sign, total_squeeze), dtype=torch.float32)
    return test_sign, history_mask, del_circuit, test_squeeze_input, test_input


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
    window_range = 300
    data_config = {
        "factory_name": "special_wave_data_factory",
        "args": {
            "window_range": (window_range, window_range + 100),
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
    muscle_list = ["DA01", "DA02", "DA03", "DA04", "DA05", "DA06", "DA07", "DA08", "DA09",
                   "DB01", "DB02", "DB03", "DB04", "DB05", "DB06", "DB07",
                   "AS01", "AS02", "AS03", "AS04", "AS05", "AS06", "AS07", "AS08", "AS09", "AS10", "AS11",
                   "DD01", "DD02", "DD03", "DD04", "DD05", "DD06",
                   "VA01", "VA02", "VA03", "VA04", "VA05", "VA06", "VA07", "VA08", "VA09", "VA10", "VA11", "VA12",
                   "VB01", "VB02", "VB03", "VB04", "VB05", "VB06", "VB07", "VB08", "VB09", "VB10", "VB11",
                   "VD01", "VD02", "VD03", "VD04", "VD05", "VD06", "VD07", "VD08", "VD09", "VD10", "VD11", "VD12", "VD13"]
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    test_name = f"CPG_test_window{window_range}"
    print(test_name)

    nr.seed(43)
    test_sign, history_mask, test_del_circuit, test_squeeze_input, test_input = \
        cpg_test_factory(abs_circuit, muscle_list, config_name, data_config, model_config, sim_config, io_config)
    print("Dataset Generated!")

    # new start
    new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
    new_abs_circuit = transform.detailed2abstract(del_circuit)
    new_abs_circuit.update_connections(new_weights)
    if torch.cuda.is_available():
        gpus = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    else:
        gpus = ['cpu']
    print(gpus)
    new_art_circuit = artificial_circuit.ArtificialCircuit(new_abs_circuit, weight_config, model_config, gpus)

    working_directory = path.join(path.dirname(__file__), "log", task_name, test_name)
    os.makedirs(working_directory, exist_ok=True)
    network_inception.visualize_circuit_history(test_del_circuit, sim_config,
                                                path.join(working_directory, 'test_gt.jpg'))
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
    loss_rec, loss_mean_rec, num_window = [], [], int(np.ceil(test_squeeze_input.shape[-1] / half_window)) - 1
    for epoch in range(train_config['num_epoch']):
        avg_loss = 0

        for window_idx in range(num_window):
            _ = new_art_circuit(test_squeeze_input[..., window_idx * half_window:(window_idx + 2) * half_window],
                                io_config["input_cells"], io_config["output_cells"])
            epoch_loss = 0
            optimizer.zero_grad()
            for connection_id, connection in enumerate(new_art_circuit.circuit.connections):
                if history_mask[connection_id]:
                    history_sign = test_sign[..., window_idx * half_window:(window_idx + 2) * half_window]
                    loss = -1 * torch.sum(connection.history * history_sign.to(connection.history.device))
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

            vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"), log=False)
