import os
import sys
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train import *
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import _pickle as pkl
import numpy.random as nr
import os.path as path
import matplotlib.pyplot as plt


class ReadOut(nn.Module):
    def __init__(self):
        super(ReadOut, self).__init__()
        self.pad_len = 1
        # self.layer = nn.Conv1d(80, 96, kernel_size=self.pad_len+1)
        self.layer = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=self.pad_len+1),
            nn.LeakyReLU(),
            nn.Conv1d(64, 96, kernel_size=1))
        # self.layer = nn.Conv1d(80, 96, kernel_size=1)

    def forward(self, x):
        x = F.pad(x, (self.pad_len, 0), value=-60.)
        return self.layer((x + 60.) / 80)


def compute_density(head_p, direction_v):
    direction_v = direction_v / np.linalg.norm(direction_v, axis=1, keepdims=True)
    direction_p = head_p + (0.3 / 36) * direction_v
    direction_density = np.linalg.norm(direction_p, axis=-1)
    return -direction_density


def forage_data_loader(cutoff=200, shift=0, ratio=6):
    gradient_dataset = []
    muscle_dataset = []
    vector_dataset = []
    density_dataset = []
    for root, dirs, files in os.walk(path.join(path.dirname(__file__), '..', "foraging_test", "recordings")):
        for file in files:
            data = pkl.load(open(path.join(root, file), "rb"))
            if len(data['Forward Vector']) < cutoff + shift + 1:
                continue
            forward_vector = np.array(data['Forward Vector'])[shift:]
            dorsal_vector = np.array(data['Dorsal Vector'])[shift:]
            head_vector = np.array(data['Head Vector'])[shift:]
            muscle = np.array(data['Muscle'])[shift:, :, 0].transpose()
            right_vector = np.cross(forward_vector, dorsal_vector, axis=-1)
            left_vector = np.cross(dorsal_vector, forward_vector, axis=-1)
            right_density = compute_density(head_vector, right_vector)
            left_density = compute_density(head_vector, left_vector)
            right_gradient = (right_density[1:] - right_density[:-1]) / 0.01
            left_gradient = (left_density[1:] - left_density[:-1]) / 0.01
            # vector_dataset.append(torch.tensor(
            #     np.concatenate([forward_vector[:cutoff], dorsal_vector[:cutoff], head_vector[:cutoff]],
            #                    axis=1).transpose(), dtype=torch.float32).cuda())
            #
            # density_dataset.append(
            #     torch.tensor(np.stack([left_density[:cutoff], right_density[:cutoff]], axis=0),
            #                  dtype=torch.float32).cuda())
            gradient_dataset.append(torch.tensor(data_factory.squeeze_trace(np.stack([
                -left_gradient[:cutoff], -right_gradient[:cutoff], left_gradient[:cutoff], right_gradient[:cutoff]],
                axis=0), 1/ratio), dtype=torch.float32).cuda())
            muscle_dataset.append(torch.tensor(data_factory.squeeze_trace(muscle[:, :cutoff], 1/ratio),
                                               dtype=torch.float32).cuda())
    input_dataset = torch.stack(gradient_dataset, dim=0)
    input_dataset = (input_dataset/torch.max(input_dataset))*70.-20.
    output_dataset = torch.stack(muscle_dataset, dim=0)
    print(f"input data: {input_dataset.shape}, output data: {output_dataset.shape}")
    return input_dataset, output_dataset


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    dataset_squeeze, addtional_squeeze = 20, 1
    inoise = 0
    circuit_name = "full109"
    config_name = "config_full109"
    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    test_name = f"forage_test_0"

    working_directory = path.join(path.dirname(__file__), "log", task_name, test_name)
    os.makedirs(working_directory, exist_ok=True)
    for vis_config in (circuit_name, config_name, task_name, dataset_name):
        print(vis_config)

    config, abs_circuit, del_circuit, pretrain_dir = network_inception.single_nrn_preparation(
        config_name, task_name, dataset_name, circuit_name)
    io_config = config['io']
    io_config["input_cells"] = ["AWCL", "AWCR", "ASKL", "ASKR"]

    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    fake_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.6, }
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "tstop": 12000, "v_init": -65, "secondorder": 0}

    # prepare models
    new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
    new_abs_circuit = transform.detailed2abstract(del_circuit)
    new_abs_circuit.update_connections(new_weights)
    # gpus = [torch.device(f"cuda:{cuda_index}") for cuda_index in range(torch.cuda.device_count())]
    gpus = [torch.device("cuda:1")]
    new_art_circuit = artificial_circuit.ArtificialCircuit(new_abs_circuit, weight_config, model_config, gpus)
    read_out = ReadOut().to(new_art_circuit.output_device)

    # prepare dataset
    dataset_config = {
        "cutoff": 200,
        "shift": 0,
        "ratio": 6
    }
    assert int(dataset_config['cutoff']*dataset_config['ratio']*model_config['squeeze']) == int(sim_config['tstop']/sim_config['dt'])
    input_dataset, output_dataset = forage_data_loader(**dataset_config)
    train_input_full, test_input = input_dataset[:-5], input_dataset[-5:]
    test_prolong_input = data_factory.squeeze_trace(np.array(test_input.cpu()), 1/(dataset_squeeze * addtional_squeeze))
    train_output_full, test_output = output_dataset[:-5].to(new_art_circuit.output_device), output_dataset[-5:]

    # training process
    print("Training Start!")
    print(io_config)
    print(torch.cuda.is_available(), ": ", torch.cuda.device_count())
    print(gpus)
    print(new_art_circuit.output_device)
    print(test_name)

    torch.backends.cudnn.benchmark = True

    train_config = {
        "loss": "MSE",
        "lr": 5e-3,
        "num_epoch": 1025,
        "window_len": 150,
        "batch_size": 32
    }
    params = [{"params": new_art_circuit.fetch_meta_weights(), "lr": train_config['lr']},
              {"params": read_out.parameters(), "lr": train_config['lr']}]
    optimizer = torch.optim.Adam(params)
    criterion = torch.nn.MSELoss() if train_config['loss'] == "MSE" else torch.nn.L1Loss()
    criterion = criterion.to(new_art_circuit.output_device)
    half_window = int(train_config['window_len'] // 2)
    loss_rec, loss_mean_rec, num_window = [], [], int(np.ceil(train_input_full.shape[-1] / half_window)) - 1

    for epoch in range(train_config['num_epoch']):
        new_art_circuit.mode_switch("train")
        avg_loss = 0
        tmp_indices = nr.choice(train_output_full.shape[0], train_config['batch_size'], replace=False)
        train_input = train_input_full[tmp_indices]
        train_output = train_output_full[tmp_indices]
        for window_idx in range(num_window):
            motor_signal = new_art_circuit(train_input[..., window_idx*half_window:(window_idx+2)*half_window],
                                           io_config["input_cells"], io_config["output_cells"])
            muscle_traces = read_out(motor_signal)
            epoch_loss = criterion(muscle_traces, train_output[..., window_idx*half_window:(window_idx+2)*half_window])
            loss_complete_time = time.time()

            optimizer.zero_grad()
            epoch_loss.backward()

            optimizer.step()
            avg_loss += epoch_loss.cpu()

            print(f"gradient assignment takes {time.time()-loss_complete_time} sec")
        avg_loss /= num_window
        print(f"Epoch {epoch}/{train_config['num_epoch']} Loss {float(avg_loss)}")
        loss_rec.append(float(avg_loss))
        loss_mean_rec.append(np.mean(loss_rec[-100:]))
        vis.visualize_loss(loss_rec, loss_mean_rec, path.join(working_directory, "loss.jpg"), log=False)
        if epoch & (epoch - 1) == 0:
            os.makedirs(path.join(working_directory, f"epoch #{epoch}"), exist_ok=True)
            with torch.no_grad():
                network_inception.artificial_circuit_inception(
                    new_art_circuit, test_input[0:1], test_prolong_input[0], config_name, io_config, sim_config, model_config,
                    path.join(working_directory, f"epoch #{epoch}"), 'ad', "circuit_ckp", prefix="test_")
                test_voltage = new_art_circuit(test_input[0:1], io_config["input_cells"], io_config["output_cells"])
                test_muscle = read_out(test_voltage)
                train_voltage = new_art_circuit(train_input[0:1], io_config["input_cells"], io_config["output_cells"])
                train_muscle = read_out(train_voltage)
                # network_inception.visualize_circuit_history(new_art_circuit, sim_config,
                #                                             path.join(working_directory, f"epoch #{epoch}", 'train_ckp.jpg'))
            vis.vis_muscle_io(test_input[0], test_voltage[0], io_config["output_cells"], test_muscle[0],
                              x_dt=model_config['squeeze']*sim_config['dt'], gt_muscle=test_output[0],
                              save_dir=path.join(working_directory, f"epoch #{epoch}", "muscle_test.jpg"))
            vis.vis_muscle_io(train_input[0], train_voltage[0], io_config["output_cells"], train_muscle[0],
                              x_dt=model_config['squeeze']*sim_config['dt'], gt_muscle=train_output[0],
                              save_dir=path.join(working_directory, f"epoch #{epoch}", "muscle_train.jpg"))
        # exit()









