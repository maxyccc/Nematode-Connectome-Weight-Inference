import torch
import numpy as np
import _pickle as pickle
import numpy.random as nr
import matplotlib.pyplot as plt
import os
import os.path as path
import sys
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train import *


def muscle_test_factory(sim_config, data_config):
    data_config['args']['start_phase'] = -np.pi
    test_muscle_13, test_sign_13, input_volt = eval("data_factory." + data_config['factory_name'])(
        num=24, tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'], with_sign=True)

    data_config['args']['start_phase'] = 0
    test_muscle_02, test_sign_02, _ = eval("data_factory." + data_config['factory_name'])(
        num=24, tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'], with_sign=True)

    total_muscle = np.concatenate([test_muscle_02, test_muscle_13, test_muscle_02, test_muscle_13], axis=0)
    total_sign = torch.tensor(np.concatenate([test_sign_02, test_sign_13, test_sign_02, test_sign_13], axis=0), dtype=torch.float32)
    print(total_muscle.shape)
    return total_muscle, total_sign, input_volt


def readout2muscle(config):
    neuron_muscle, nrow, ncol = func.read_excel(
        file_name=path.join(path.dirname(__file__), '..', config['dir_info']['neuron_muscle_con']), sheet_name='prop')
    r2m_dic = {}
    for muscle_id in range(nrow-1):
        muscle_name = neuron_muscle.cell_value(muscle_id+1, 0)
        r2m_dic[muscle_name] = []
        for neuron_id in range(ncol-1):
            neuron_name = neuron_muscle.cell_value(0, neuron_id+1)
            if len(str(neuron_muscle.cell_value(muscle_id+1, neuron_id+1))) > 0:
                r2m_dic[muscle_name].append(neuron_name)
    return r2m_dic


def out2muscle(output_prediction, output_index, r2m_dic, muscles_name):
    train_muscle = []
    for muscle_name in muscles_name:
        current_muscle = torch.stack(
            [output_prediction[0, output_index[neuron_name]] for neuron_name in r2m_dic[muscle_name]], dim=0).mean(dim=0)
        train_muscle.append(current_muscle)
    return torch.stack(train_muscle, dim=0)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    dataset_squeeze, addtional_squeeze = 20, 1
    total_squeeze = dataset_squeeze * addtional_squeeze
    inoise = 0
    # circuit_name = "full109"
    # config_name = "config_full109"
    circuit_name = "body_undulation"
    config_name = "config_body_undulation"

    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    muscle_config = [
            "DR01", "DR02", "DR03", "DR04", "DR05", "DR06", "DR07", "DR08", "DR09", "DR10", "DR11", "DR12", "DR13",
            "DR14", "DR15", "DR16", "DR17", "DR18", "DR19", "DR20", "DR21", "DR22", "DR23", "DR24",
            "VR01", "VR02", "VR03", "VR04", "VR05", "VR06", "VR07", "VR08", "VR09", "VR10", "VR11", "VR12", "VR13",
            "VR14", "VR15", "VR16", "VR17", "VR18", "VR19", "VR20", "VR21", "VR22", "VR23", "VR24",
            "DL01", "DL02", "DL03", "DL04", "DL05", "DL06", "DL07", "DL08", "DL09", "DL10", "DL11", "DL12", "DL13",
            "DL14", "DL15", "DL16", "DL17", "DL18", "DL19", "DL20", "DL21", "DL22", "DL23", "DL24",
            "VL01", "VL02", "VL03", "VL04", "VL05", "VL06", "VL07", "VL08", "VL09", "VL10", "VL11", "VL12", "VL13",
            "VL14", "VL15", "VL16", "VL17", "VL18", "VL19", "VL20", "VL21", "VL22", "VL23", "VL24",]

    data_config = {
        "factory_name": "muscle_data_factory",
        "args": {
            "start_phase": -np.pi,
            "phase_shift": 4 * np.pi,
            "half_period_t": 600,
            "amp": 0.8,
            "input_volt": (-30, 40),
            "amp_decay": 0.1,
            "forward": True}}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    for vis_config in (circuit_name, config_name, task_name, dataset_name):
        print(vis_config)
    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, task_name, dataset_name, circuit_name)
    io_config = config["io"]
    r2m_dic = readout2muscle(config)

    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": total_squeeze}
    print("Loading Complete!")

    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    fake_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.6, }
    test_name = f"muscle_test_{int(data_config['args']['phase_shift']/np.pi)}pi_{data_config['args']['half_period_t']}"
    print(test_name)

    nr.seed(43)
    total_muscle, total_sign, input_volt = muscle_test_factory(sim_config, data_config)
    input_volt = np.array([input_volt for _ in range(len(io_config['input_cells']))])
    squeeze_input = torch.tensor(data_factory.squeeze_trace(input_volt, total_squeeze), dtype=torch.float32).unsqueeze(0)
    print("Dataset Generated!")

    # new start
    new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
    new_abs_circuit = transform.detailed2abstract(del_circuit)
    new_abs_circuit.update_connections(new_weights)
    if torch.cuda.is_available():
        devices = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    print(devices)
    total_sign = total_sign.to(devices[0])
    new_art_circuit = artificial_circuit.ArtificialCircuit(new_abs_circuit, weight_config, model_config, devices)

    working_directory = path.join(path.dirname(__file__), "log", task_name, test_name)
    os.makedirs(working_directory, exist_ok=True)
    vis.visualize_muscle_data(total_muscle, path.join(working_directory, 'gt.jpg'))
    print("Training Start!")

    # main part train process
    torch.backends.cudnn.benchmark = True
    new_art_circuit.mode_switch("train")
    train_config = {
        "loss": "MSE",
        "lr": 5e-3,
        "num_epoch": 1025,
        "window_len": 500,
    }
    optimizer = torch.optim.Adam(new_art_circuit.fetch_meta_weights(), lr=train_config['lr'])
    criterion = torch.nn.MSELoss() if train_config['loss'] == "MSE" else torch.nn.L1Loss()
    half_window = int(train_config['window_len'] // 2)
    output_index = dict(zip(io_config["output_cells"], range(len(io_config["output_cells"]))))

    loss_rec, loss_mean_rec, num_window = [], [], int(np.ceil(squeeze_input.shape[-1] / half_window)) - 1
    for epoch in range(train_config['num_epoch']):
        avg_loss = 0

        for window_idx in range(num_window):
            output_prediction = new_art_circuit(squeeze_input[..., window_idx*half_window:(window_idx+2)*half_window],
                                io_config["input_cells"], io_config["output_cells"])
            optimizer.zero_grad()

            train_muscle = out2muscle(output_prediction, output_index, r2m_dic, muscle_config)
            loss = -1 * torch.sum(total_sign[:, window_idx*half_window:(window_idx+2)*half_window]*train_muscle)

            loss.backward()
            optimizer.step()
            avg_loss += loss.cpu()
        avg_loss /= num_window
        print(f"Epoch {epoch}/{train_config['num_epoch']} Loss {float(avg_loss)}")
        loss_rec.append(float(avg_loss))
        loss_mean_rec.append(np.mean(loss_rec[-100:]))
        if epoch & (epoch - 1) == 0:
            with torch.no_grad():
                network_inception.artificial_circuit_inception(
                    new_art_circuit, squeeze_input, input_volt, config_name, io_config, sim_config, model_config,
                    path.join(working_directory, f"epoch #{epoch}"), 'ad', "circuit_ckp", prefix="")
                output_prediction = new_art_circuit(squeeze_input, io_config["input_cells"], io_config["output_cells"])
                test_muscle = out2muscle(output_prediction, output_index, r2m_dic, muscle_config)
                vis.visualize_muscle_data(test_muscle.cpu().numpy(),
                                          path.join(working_directory, f"epoch #{epoch}", 'muscle_ckp.jpg'))
            vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"), log=False)



