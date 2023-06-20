import os
import os.path as path
import numpy as np
from eworm.utils import *
from eworm.network import *
from eworm.network_train import *
import matplotlib.pyplot as plt
import _pickle as pickle


def read_data():
    input_data = np.load(path.join(path.dirname(__file__), "ckp", "input_data.npy"))
    input_data = data_factory.squeeze_trace(input_data, 1 / 20)
    return input_data


def oscillation_amp(data):
    return np.sum(np.abs(data[..., 1:] - data[..., :-1]), axis=-1)**2


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


if __name__ == "__main__":
    working_directory = path.join(path.dirname(__file__), "figs")
    os.makedirs(working_directory, exist_ok=True)
    config = func.load_json(path.join(path.dirname(__file__), "ckp", "config_full109.json"))
    abs_circuit = pickle.load(open(path.join(path.dirname(__file__), "ckp", "circuit_ckp.pkl"), "rb"))
    del_circuit = transform.abstract2detailed(abs_circuit, config)
    io_config = config["io"]
    correlation_config = {
        "ca_traces_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces.txt"),
        "cell_names_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces_cell_name.txt"),
        "time_series_dir": path.join(path.dirname(__file__), "..", "tmp_data", "Ca_traces_time.txt"),
        "clip_time": (0, 300)}
    test_config = {
        "dt": 0.1,
        "batch_len": 10,
        "vis_ratio": 1/4}

    input_data = read_data()
    print(input_data.shape)
    admission, ca_data, history_mask, gt_ref_circuit = correlation_data_reader(correlation_config, test_config,
                                                                                abs_circuit)
    with open(path.join(path.dirname(__file__), "ckp", "Ca_corr_mat_cell_name.txt")) as f:
        output_names_target = f.read().split("\t")
    ca_corr = np.loadtxt(path.join(path.dirname(__file__), "ckp", "Ca_corr_mat.txt"))

    sim_config = {"dt": 0.5, "tstop": 75000, "v_init": -65, "secondorder": 0}
    select_len = int(sim_config['tstop']/sim_config['dt'])
    output_traces = del_circuit.simulation(sim_config, input_data[:, :select_len], io_config["input_cells"], output_names_target)
    print(output_traces.shape)
    _, len_sim = output_traces.shape
    assert len_sim == int(sim_config['tstop']/sim_config['dt'])
    sim_corr = np.corrcoef(output_traces)
    oscillation_traces = oscillation_amp(output_traces.reshape(len(output_names_target), int(len_sim/200), 200))
    print(oscillation_traces.shape)
    osc_corr = np.corrcoef(oscillation_traces)

    for connection in del_circuit.connections:
        connection.history = None
    for output_idx, cell_name in enumerate(output_names_target):
        tmp_cell = del_circuit.cell(cell_name=cell_name)
        for connection in tmp_cell.post_connections:
            if connection.post_segment is None:
                tmp_osc = oscillation_traces[output_idx]
                connection.history = ((tmp_osc - np.mean(tmp_osc))/np.std(tmp_osc))*10-70
                break
    network_inception.visualize_circuit_history(del_circuit, sim_config, path.join(working_directory, "osc.jpg"),
                                                gt_circuit=gt_ref_circuit)

    fig = plt.figure(figsize=(72, 24), dpi=200)
    ax = fig.add_subplot(131)
    ax.set_title("Simulation", fontsize=40)
    ax.set_yticks(range(len(io_config["output_cells"])))
    ax.set_yticklabels(io_config["output_cells"], fontsize=20)
    ax.set_xticks(range(len(io_config["output_cells"])))
    ax.set_xticklabels(io_config["output_cells"], fontsize=18)
    im = ax.imshow(sim_corr, cmap=plt.cm.cool)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    plt.tight_layout()

    ax = fig.add_subplot(132)
    ax.set_title("Oscillation", fontsize=40)
    ax.set_yticks(range(len(io_config["output_cells"])))
    ax.set_yticklabels(io_config["output_cells"], fontsize=20)
    ax.set_xticks(range(len(io_config["output_cells"])))
    ax.set_xticklabels(io_config["output_cells"], fontsize=18)
    im = ax.imshow(osc_corr, cmap=plt.cm.cool)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    plt.tight_layout()

    ax = fig.add_subplot(133)
    ax.set_title("Experiment", fontsize=40)
    ax.set_yticks(range(len(io_config["output_cells"])))
    ax.set_yticklabels(io_config["output_cells"], fontsize=20)
    ax.set_xticks(range(len(io_config["output_cells"])))
    ax.set_xticklabels(io_config["output_cells"], fontsize=18)
    im = ax.imshow(ca_corr, cmap=plt.cm.cool)
    plt.xticks(rotation=45)
    plt.rcParams['font.size'] = 30
    plt.colorbar(im, fraction=0.0452)
    plt.tight_layout()
    fig.savefig(path.join(working_directory, "corr_test.png"))
    plt.close()

