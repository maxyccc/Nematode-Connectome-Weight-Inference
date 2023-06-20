import torch
import torch.nn as nn
import _pickle as pkl
import os
import os.path as path
import numpy as np
import matplotlib.pyplot as plt
import time


class SimpleCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleCNN, self).__init__()
        # self.blocks = nn.Sequential(
        #     nn.Conv1d(in_channels=2, out_channels=32, kernel_size=5, padding=2),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        #     nn.LeakyReLU(),
        #     nn.Conv1d(in_channels=64, out_channels=96, kernel_size=5, padding=2)
        # )
        self.input = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels=64, kernel_size=7, padding=3),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=7, padding=3),
        )
        self.rnn = nn.RNN(64, 64, num_layers=1)
        self.output = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=64, out_channels=out_dim, kernel_size=1))

    def forward(self, x):
        return self.output(self.rnn(self.input(x).permute(2, 0, 1))[0].permute(1, 2, 0))


def compute_density(head_p, direction_v):
    direction_v = direction_v / np.linalg.norm(direction_v, axis=1, keepdims=True)
    direction_p = head_p + (0.3 / 36) * direction_v
    direction_density = np.linalg.norm(direction_p, axis=-1)
    return -direction_density


def simple_muscle(muscle):
    assert muscle.shape[0] == 96
    return np.stack([np.mean(muscle[:24], axis=0), np.mean(muscle[24:48], axis=0),
                     np.mean(muscle[48:72], axis=0), np.mean(muscle[72:], axis=0)], axis=0)


def plot_io(input_data, output_data, gt_output=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(30, 15), nrows=2, ncols=2, sharex='all', gridspec_kw={'height_ratios': [5, 20]})
    input_traces = np.array(input_data.detach().cpu())
    output_traces = np.array(output_data.detach().cpu())
    for trace_id in range(input_traces.shape[0]):
        ax[0, 0].plot(input_traces[trace_id], alpha=0.7, lw=0.5)
    # ax[0, 0].legend()
    # assert output_traces.shape[0] == 96
    ax[1, 0].set_title('Output')
    shift_len = 24 if output_traces.shape[0] == 96 else 1
    shift_value = 0.
    for trace_id in range(output_traces.shape[0]):
        ax[1, 0].plot(output_traces[trace_id] + int(trace_id / shift_len) * shift_value, alpha=0.7, lw=0.7)

    if gt_output is not None:
        gt_traces = np.array(gt_output.detach().cpu())
        ax[1, 1].set_title('GroundTruth')
        for trace_id in range(output_traces.shape[0]):
            ax[1, 1].plot(gt_traces[trace_id] + int(trace_id / shift_len) * shift_value, alpha=0.7, lw=0.7)
            ax[0, 1].plot(gt_traces[trace_id] - output_traces[trace_id]+ int(trace_id / shift_len) * shift_value/2, alpha=0.7, lw=0.7)
    else:
        ax[1, 1].remove()
        ax[0, 1].remove()
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir, dpi=300)


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    cutoff = 200
    shift = 0
    gradient_dataset = []
    muscle_dataset = []
    vector_dataset = []
    density_dataset = []
    for root, dirs, files in os.walk("./recordings"):
        for file in files:
            data = pkl.load(open(path.join(root, file), "rb"))
            if len(data['Forward Vector']) < cutoff + shift + 1:
                continue
            forward_vector = np.array(data['Forward Vector'])[shift:]
            dorsal_vector = np.array(data['Dorsal Vector'])[shift:]
            head_vector = np.array(data['Head Vector'])[shift:]

            muscle = np.array(data['Muscle'])[shift:, :, 0].transpose()
            muscle4 = simple_muscle(muscle)
            right_vector = np.cross(forward_vector, dorsal_vector, axis=-1)
            left_vector = np.cross(dorsal_vector, forward_vector, axis=-1)
            right_density = compute_density(head_vector, right_vector)
            left_density = compute_density(head_vector, left_vector)
            right_gradient = (right_density[1:] - right_density[:-1])/0.01
            left_gradient = (left_density[1:] - left_density[:-1])/0.01
            vector_dataset.append(torch.tensor(
                np.concatenate([forward_vector[:cutoff], dorsal_vector[:cutoff], head_vector[:cutoff]],
                               axis=1).transpose(), dtype=torch.float32).cuda())

            density_dataset.append(
                torch.tensor(np.stack([left_density[:cutoff], right_density[:cutoff]], axis=0),
                             dtype=torch.float32).cuda())
            gradient_dataset.append(
                torch.tensor(np.stack([left_gradient[:cutoff], right_gradient[:cutoff]], axis=0),
                             dtype=torch.float32).cuda())
            muscle_dataset.append(torch.tensor(muscle4[:, :cutoff], dtype=torch.float32).cuda())
    input_dataset = gradient_dataset
    output_dataset = muscle_dataset
    train_input = torch.stack(input_dataset[:-5], dim=0)
    train_output = torch.stack(output_dataset[:-5], dim=0)
    print(train_input.shape, train_output.shape)

    model = SimpleCNN(train_input.shape[1], train_output.shape[1]).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.L1Loss()
    num_epoch = 4096*2+1
    for epoch in range(num_epoch):
        train_pred = model(train_input)
        loss = criterion(train_pred, train_output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"{epoch}/{num_epoch - 1}, Loss: {loss.detach().cpu()}")
        if (epoch & (epoch - 1) == 0) and (epoch > 32):
            test_index = 0
            plot_io(input_dataset[test_index], model(input_dataset[test_index].unsqueeze(0))[0],
                        gt_output=output_dataset[test_index], save_dir="./tmp.jpg")
            exit()

    for test_index in (-1, -2, -3, -4, -5):
        plot_io(input_dataset[test_index], model(input_dataset[test_index].unsqueeze(0))[0],
                    gt_output=output_dataset[test_index])
