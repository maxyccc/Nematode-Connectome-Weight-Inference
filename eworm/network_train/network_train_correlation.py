import os
import sys
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train import *
import time
import torch
import torch.nn as nn
import numpy as np
import _pickle as pickle
import numpy.random as nr
import os.path as path
import matplotlib.pyplot as plt


def meta2input(meta_data):
    return torch.sigmoid(meta_data) * 90 - 70.


def input2meta(input_data):
    input_data = np.clip(input_data, -70 + .1, 20 - .1)
    return -np.log((1 / ((input_data + 70) / 90)) - 1)


def correlation_data_reader(correlation_config, test_config, ref_circuit):
    ori_ca_traces = func.read_txt_array(correlation_config['ca_traces_dir']).transpose()
    cell_names = func.read_txt_list(correlation_config['cell_names_dir'])[0]
    time_series = func.read_txt_array(correlation_config['time_series_dir'])[:, 0]
    ca_traces = []
    for i in range(ori_ca_traces.shape[0]):
        ca_traces.append(
            np.interp(np.arange(*correlation_config['clip_time'], test_config['dt']), time_series, ori_ca_traces[i]))
    history_mask = np.zeros(len(ref_circuit.connections), dtype=bool)
    cell_dict = {}
    for cell_idx, cell_name in enumerate(cell_names):
        exist_output = -1
        for cnt_id, connection in enumerate(ref_circuit.connections):
            if (connection.pre_cell is not None) and (connection.pre_cell.name == cell_name) and (
                    connection.post_segment is None):
                exist_output = cnt_id
        if exist_output >= 0:
            history_mask[exist_output] = True
            cell_dict[cell_name] = cell_idx
    correlation_data = []
    admission = []
    for cnt_id, connection in enumerate(ref_circuit.connections):
        if history_mask[cnt_id]:
            tmp_traces = ca_traces[cell_dict[connection.pre_cell.name]]
            shorten_traces = tmp_traces[:int(len(tmp_traces)*test_config['vis_ratio'])]
            connection.history = ((shorten_traces - np.mean(shorten_traces))/np.std(shorten_traces))*10-70
            correlation_data.append(tmp_traces)
            admission.append(connection.pre_cell.name)
        else:
            connection.history = None
    return admission, np.stack(correlation_data, axis=0), history_mask, ref_circuit


def correlation_test_factory(ca_data, test_config, data_config, model_config, sim_config, io_config):
    full_batch = int(test_config["batch_len"] / test_config["dt"])
    num_batch = int(ca_data.shape[-1] / full_batch)
    target_ca = []
    for batch_id in range(num_batch):
        target_ca.append(ca_data[:, batch_id * full_batch: (batch_id + 1) * full_batch])
    target_ca = torch.tensor(np.stack(target_ca, axis=0), dtype=torch.float32)
    input_data = eval("data_factory." + data_config['factory_name'])(
        num=len(io_config["input_cells"]), tstop=int(ca_data.shape[-1] * test_config["dt"] * 1000),
        dt=sim_config['dt'] * model_config['squeeze'], **data_config['args'])
    meta_input = nn.Parameter(torch.tensor(input2meta(input_data), dtype=torch.float32), requires_grad=True)
    return target_ca, meta_input


def prepare_batch(window_idx, target_ca, meta_input, train_config, params):
    # 199x58x10,0 15x1000,00
    batch_size = train_config['batch_size']
    half_window = params["half_window"]
    half_window_target = params["half_window_target"]
    batch_len = params["batch_len"]
    random_batch_indices = nr.choice(target_ca.shape[0], batch_size, replace=False)
    current_input, current_target = [], []
    for batch_idx in random_batch_indices:
        current_target.append(target_ca[batch_idx])
        current_input.append(meta_input[..., batch_idx * batch_len: (batch_idx + 1) * batch_len])
    return \
        meta2input(torch.stack(current_input, dim=0))[..., window_idx * half_window:(window_idx + 2) * half_window], \
        torch.stack(current_target, dim=0)[..., window_idx * half_window_target:(window_idx + 2) * half_window_target]


def oscillation_amp(data):
    return torch.pow(torch.sum(torch.abs(data[..., 1:] - data[..., :-1]), dim=-1), 2)


def zscore(data):
    return (data - torch.mean(data)) / torch.std(data)


def correlation_loss(connection_history, target_amp, ratio, criterion):
    return criterion(zscore(oscillation_amp(connection_history.reshape(*target_amp.shape, ratio))), zscore(target_amp))


def concat_1dim(data):
    return torch.cat([data[dim1_idx] for dim1_idx in range(data.shape[0])], dim=-1)


def assign_oscillation_amp(vis_ref_circuit, art_circuit, ratio):
    for cnt_id, connection in enumerate(art_circuit.circuit.connections):
        history_len = connection.history.shape[-1]
        reshape_history = (zscore(oscillation_amp(connection.history.reshape(int(history_len/ratio), ratio))))*10-70
        vis_ref_circuit.connections[cnt_id].history = reshape_history.detach().cpu().numpy()
    return vis_ref_circuit


def save_input_data(meta_input, save_dir):
    data2save = meta2input(meta_input).detach().cpu().numpy()
    vis.visualize_learned_input(data2save, path.join(save_dir, "input_data.jpg"))
    np.save(path.join(save_dir, "input_data.npy"), data2save)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    dataset_squeeze, addtional_squeeze = 20, 1
    inoise = 0
    circuit_name = "full109"
    config_name = "config_full109"

    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    correlation_config = {
        "ca_traces_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces.txt"),
        "cell_names_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces_cell_name.txt"),
        "time_series_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces_time.txt"),
        "clip_time": (0, 300)}
    test_config = {
        "dt": 0.1,
        "batch_len": 10,
        "vis_ratio": 1/4}
    data_config = {
        "factory_name": "random_data_factory",
        "args": {
            "window_range": (10, 1000),
            "volt_range": (-70, 20),
            "noise_settings": ((40, 5), (180, 60)),
            "reverse_noise_setting": None}}
    for vis_config in (circuit_name, config_name, task_name, dataset_name):
        print(vis_config)
    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, task_name, dataset_name, circuit_name)
    io_config = config['io']
    print("Loading Complete!")

    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "v_init": -65, "secondorder": 0}
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    fake_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.6, }
    test_name = f"correlation_test"
    print(test_name)

    admission, ca_data, history_mask, gt_ref_circuit = correlation_data_reader(correlation_config, test_config,
                                                                                abs_circuit)
    target_ca, meta_input = correlation_test_factory(ca_data, test_config, data_config, model_config, sim_config,
                                                     io_config)

    # load artificial network
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
    print("Training Start!")
    clip_time = correlation_config['clip_time']
    vis_sim_config = {
        "dt": 0.5, "v_init": -65, "secondorder": 0,
        "tstop": int((clip_time[1] - clip_time[0])*test_config['vis_ratio']*1000)}
    network_inception.visualize_circuit_history(gt_ref_circuit, vis_sim_config,
                                                path.join(working_directory, f'gt.jpg'))
    vis_ref_circuit = transform.detailed2abstract(gt_ref_circuit)

    # main part train process
    torch.backends.cudnn.benchmark = True
    new_art_circuit.mode_switch("train")
    train_config = {
        "loss": "MSE",
        "lr": 5e-3,
        "num_epoch": 1025,
        "window_len": 500,
        "batch_size": 16,
        "input_grad_amplifier": 10
    }
    optimizer = torch.optim.Adam(
        [{"params": meta_input, "lr": train_config['lr'] * train_config['input_grad_amplifier']},
         {"params": new_art_circuit.fetch_meta_weights(), "lr": train_config['lr']}])
    criterion = torch.nn.MSELoss() if train_config['loss'] == "MSE" else torch.nn.L1Loss()
    params = {
        "half_window": int(train_config['window_len'] // 2),
        "ratio_art2ca": int(test_config["dt"] * 1000 / (sim_config["dt"] * model_config["squeeze"]))  # 10
    }
    params["half_window_target"] = int(params["half_window"] / params["ratio_art2ca"])
    params["batch_len"] = int(test_config["batch_len"] / (test_config["dt"] / params["ratio_art2ca"]))

    loss_rec, loss_mean_rec, num_window = [], [], int(np.ceil(params["batch_len"] / params["half_window"])) - 1
    for epoch in range(train_config['num_epoch']):
        avg_loss = 0

        for window_idx in range(num_window):
            train_input, train_target = prepare_batch(window_idx, target_ca, meta_input, train_config, params)
            # if window_idx == 0:
            #     print(target_ca.shape, meta_input.shape, train_input.shape, train_target.shape)
            _ = new_art_circuit(train_input, io_config["input_cells"], io_config["output_cells"])
            epoch_loss = 0
            optimizer.zero_grad()
            for output_cnt_id, cell_name in enumerate(admission):
                for connection in new_art_circuit.circuit.cell(cell_name=cell_name).post_connections:
                    if connection.post_segment is None:
                        epoch_loss += correlation_loss(connection.history,
                                                       train_target[:, output_cnt_id].to(connection.history.device),
                                                       params["ratio_art2ca"], criterion).cpu()

            epoch_loss.backward()
            optimizer.step()
            avg_loss += epoch_loss.cpu()
        avg_loss /= num_window
        print(f"Epoch {epoch}/{train_config['num_epoch']} Loss {float(avg_loss)}")
        loss_rec.append(float(avg_loss))
        loss_mean_rec.append(np.mean(loss_rec[-100:]))
        if epoch & (epoch - 1) == 0:
            with torch.no_grad():
                test_squeeze_input = meta2input(meta_input[:, :int(meta_input.shape[-1]*test_config['vis_ratio'])]).unsqueeze(0)
                test_input = data_factory.squeeze_trace(test_squeeze_input[0].detach().cpu().numpy(),
                                                        1 / model_config['squeeze'])
                # print(test_input.shape, vis_sim_config)
                network_inception.artificial_circuit_inception(
                    new_art_circuit, test_squeeze_input, test_input, config_name, io_config, vis_sim_config, model_config,
                    path.join(working_directory, f"epoch #{epoch}"), 'ad', "circuit_ckp", prefix=f"test_")
                vis_ref_circuit = assign_oscillation_amp(vis_ref_circuit, new_art_circuit, params["ratio_art2ca"])
                network_inception.visualize_circuit_history(
                    vis_ref_circuit, vis_sim_config,
                    path.join(working_directory, f"epoch #{epoch}", f'oscillation_amp.jpg'), gt_circuit=gt_ref_circuit)
            save_input_data(meta_input, path.join(working_directory, f"epoch #{epoch}"))
            vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"), log=False)
