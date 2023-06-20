import os
import sys
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train.artificial_circuit import ArtificialCircuit
import argparse
import time
import torch
import numpy as np
import _pickle as pickle
import numpy.random as nr
import os.path as path
import matplotlib.pyplot as plt


def single_nrn_preparation(config_name, task_name, dataset_name, circuit_name, slurm=False):
    # prepare directory
    config_dir = path.join(path.dirname(__file__), "configs", config_name + ".json")
    circuit_dir = path.join(path.dirname(__file__), "circuits")
    single_nrn_train_dir = path.join(path.dirname(__file__), "..", "single_nrn_train")
    pretrain_dir = path.join(single_nrn_train_dir, "output", task_name)
    # make circuit
    config = func.load_json(config_dir)
    assert "io" in config.keys()
    io_config = config["io"]
    if path.exists(path.join(circuit_dir, circuit_name + ".pkl")):
        abs_circuit = pickle.load(open(path.join(circuit_dir, circuit_name + ".pkl"), "rb"))
        del_circuit = transform.abstract2detailed(abs_circuit, config)
    else:
        del_circuit = transform.config2detailed(config, io_config["input_cells"], io_config["output_cells"])
        abs_circuit = transform.detailed2abstract(del_circuit)
        pickle.dump(abs_circuit, open(path.join(circuit_dir, circuit_name + ".pkl"), "wb"))
        vis.visualize_circuit(abs_circuit, save_dir=path.join(circuit_dir, circuit_name + ".jpg"), layout="circular")
    print(f"Cell Num: {len(del_circuit.cells)}, Connection Num: {len(del_circuit.connections)}")
    func.network_feasibility_check(abs_circuit)
    # submit slurm task
    if not func.single_nrn_checker(abs_circuit, pretrain_dir):
        if slurm:
            func.slurm_script_generation(
                abs_circuit, run_script_dir=path.join(single_nrn_train_dir, "single_nrn_train.py"),
                save_dir=path.join(single_nrn_train_dir, "scripts"), split_num=2, dataset_name=dataset_name,
                task_name=task_name, config_dir=config_dir, circuit_dir=path.join(circuit_dir, circuit_name + ".pkl"))
        else:
            func.local_script_generation(
                abs_circuit, run_script_dir=path.join(single_nrn_train_dir, "single_nrn_train.py"),
                save_dir=path.join(single_nrn_train_dir, "scripts"), split_num=5, dataset_name=dataset_name,
                task_name=task_name, config_dir=config_dir, circuit_dir=path.join(circuit_dir, circuit_name + ".pkl"))
    return config, abs_circuit, del_circuit, pretrain_dir


def visualize_circuit_history(circuit, sim_config, save_dir, zoom_config=None, auxiliary_config=None, gt_circuit=None):
    # preparation
    assert isinstance(circuit, (ArtificialCircuit, detailed_circuit.DetailedCircuit, abstract_circuit.AbstractCircuit))
    art_flag, tstop = isinstance(circuit, ArtificialCircuit), sim_config["tstop"]
    zoom_config = (-0.01, 1.01) if zoom_config is None else zoom_config
    # copy connection history to vis_circuit
    if isinstance(circuit, ArtificialCircuit):
        vis_circuit = circuit.extract_connection_weights()
    elif isinstance(circuit, detailed_circuit.DetailedCircuit):
        vis_circuit = transform.detailed2abstract(circuit)
    else:
        vis_circuit = circuit
    for cnt_idx, connection in enumerate(vis_circuit.connections):
        if art_flag:
            if circuit.circuit.connections[cnt_idx].history is not None:
                connection.history = circuit.circuit.connections[cnt_idx].history[0].detach().cpu().numpy()
            else:
                connection.history = None
        else:
            connection.history = circuit.connections[cnt_idx].history
    # prepare auxiliary output
    if auxiliary_config is not None:
        auxiliary_circuit, squeeze_ratio = auxiliary_config["auxiliary_circuit"], auxiliary_config["squeeze_ratio"]
        assert isinstance(auxiliary_circuit, (ArtificialCircuit, detailed_circuit.DetailedCircuit))
        aux_art_flag = isinstance(auxiliary_circuit, ArtificialCircuit)
    # plot figure
    num_block = int(np.ceil(np.sqrt(len(vis_circuit.cells))))
    plt.figure(figsize=(num_block * 10, num_block * 5))
    for cell_idx, cell in enumerate(vis_circuit.cells):
        # plot cell input & output
        plt.subplot(num_block, num_block, cell_idx + 1)
        for pre_connection in cell.pre_connections:
            if pre_connection.history is not None:
                plt.plot(np.linspace(0, tstop, len(pre_connection.history)), pre_connection.history,
                         alpha=0.2, label=f"{pre_connection.category} {np.round(pre_connection.weight, 4)}", lw=4)
        output_label = "output_voltage"
        for post_segment_idx in np.unique([post_cnt.pre_segment.index for post_cnt in cell.post_connections]):
            for post_connection in cell.segment(post_segment_idx).post_connections:
                if post_connection.history is not None:
                    plt.plot(np.linspace(0, tstop, len(post_connection.history)), post_connection.history,
                             alpha=0.4, color='b', label=output_label, ls="--")
                    break
            output_label = None
        if auxiliary_config is not None:
            aux_input = np.stack([pre_cnt.history for pre_cnt in cell.pre_connections], axis=0)
            aux_input = data_factory.squeeze_trace(aux_input, squeeze_ratio)
            if aux_art_flag:
                aux_label = "aux_artificial"
                with torch.no_grad():
                    cell_device = auxiliary_circuit.get_device(cell)
                    aux_input = torch.FloatTensor(aux_input).unsqueeze(0).to(cell_device)
                    connection_categories = [connection.category for connection in cell.pre_connections]
                    connection_weights = torch.FloatTensor([connection.weight for connection in cell.pre_connections])
                    aux_weight = data_factory.sample2input(connection_weights, connection_categories,
                                                           auxiliary_circuit.weight_config, "torch")
                    aux_output, _ = auxiliary_circuit.circuit.cell(cell.index).model(
                        (aux_input + 25) / 75, aux_weight.unsqueeze(0).to(cell_device))
                    aux_output = (aux_output * 75 - 25)[0].cpu().numpy()
                    prefix_init = np.ones((*aux_output.shape[:-1], 1)) * sim_config["v_init"]
                    aux_output = np.concatenate([prefix_init, aux_output[..., :-1]], axis=-1)
            else:
                aux_label = "aux_detailed"
                aux_cell, _ = transform.select_cell(auxiliary_circuit, cell_index=cell.index, cell_name=cell.name)
                aux_output = aux_cell.simulation(sim_config, aux_input)[..., :-1]
            for trace in aux_output:
                plt.plot(np.linspace(0, tstop, len(trace)), trace, alpha=0.4, color='r',
                         label=aux_label, ls="--")
                aux_label = None
        if gt_circuit is None:
            # plt.legend()
            pass
        else:
            gt_cell = gt_circuit.cell(cell_index=cell.index)
            for post_segment_idx in np.unique([post_cnt.pre_segment.index for post_cnt in gt_cell.post_connections]):
                for post_connection in gt_cell.segment(post_segment_idx).post_connections:
                    if post_connection.history is not None:
                        plt.plot(np.linspace(0, tstop, len(post_connection.history)), post_connection.history,
                                 color='r', label="gt")
                        break
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.ylim(-100, 50)
        plt.xlim(zoom_config[0] * tstop, zoom_config[1] * tstop)
        title_name = f"Cell #{cell.index} {cell.name}: Input & Output(blue)"
        if auxiliary_config is not None:
            title_name += " & Auxiliary(Red)"
        plt.title(title_name)
    # plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


def artificial_circuit_inception(art_circuit, test_input_art, test_input_del,
                                 config_name, io_config, sim_config, model_config,
                                 save_dir, flag='ad', circuit_name="ref_circuit", prefix=""):
    # prepare circuit
    abs_circuit = art_circuit.extract_connection_weights()
    if '+' in flag:
        for cell in abs_circuit.cells:
            print(f"cell {cell.name} in: {len(cell.pre_connections)} out: {len(cell.post_connections)}")
    del_circuit = transform.abstract2detailed(
        abs_circuit, func.load_json(path.join(path.dirname(__file__), "configs", config_name + ".json")))

    # detailed circuit simulation
    _ = del_circuit.simulation(sim_config, test_input_del, io_config["input_cells"], io_config["output_cells"],
                               make_history=True)
    # for connection in del_circuit.connections:
    #     pre_name = connection.pre_cell.name if connection.pre_cell is not None else None
    #     print("Detailed", pre_name, connection.post_cell.name, connection.history.shape, type(connection.history))

    # artificial circuit prediction
    with torch.no_grad():
        _ = art_circuit(test_input_art, io_config["input_cells"], io_config["output_cells"])
    # for connection in art_circuit.circuit.connections:
    #     pre_name = connection.pre_cell.name if connection.pre_cell is not None else None
    #     print("Artificial", pre_name, connection.post_cell.name, connection.history.shape, type(connection.history))

    # visualization
    print(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    pickle.dump(abs_circuit, open(path.join(save_dir, circuit_name + ".pkl"), "wb"))
    vis.visualize_circuit(abs_circuit, save_dir=path.join(save_dir, prefix+"connectome.jpg"), layout="circular")
    if 'd' in flag:
        auxiliary_config = {"auxiliary_circuit": art_circuit, "squeeze_ratio": model_config["squeeze"]}
        visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_circuit.jpg"))
        if '+' in flag:
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_art.jpg"),
                                      auxiliary_config=auxiliary_config)
        if '++' in flag:
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_art_01.jpg"),
                                      zoom_config=(-0.01, 0.1), auxiliary_config=auxiliary_config)
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_art_001.jpg"),
                                      zoom_config=(-0.01, 0.01), auxiliary_config=auxiliary_config)
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_art_0005.jpg"),
                                      zoom_config=(-0.005, 0.005), auxiliary_config=auxiliary_config)
    if 's' in flag:
        auxiliary_config = {"auxiliary_circuit": del_circuit, "squeeze_ratio": 1}
        visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_del.jpg"),
                                  auxiliary_config=auxiliary_config)
        if '+' in flag:
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_del_01.jpg"),
                                      zoom_config=(-0.01, 0.1), auxiliary_config=auxiliary_config)
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_del_001.jpg"),
                                      zoom_config=(-0.01, 0.01), auxiliary_config=auxiliary_config)
            visualize_circuit_history(del_circuit, sim_config, save_dir=path.join(save_dir, prefix+"del_aux_del_0005.jpg"),
                                      zoom_config=(-0.005, 0.005), auxiliary_config=auxiliary_config)
    if 'a' in flag:
        auxiliary_config = {"auxiliary_circuit": del_circuit, "squeeze_ratio": 1 / model_config["squeeze"]}
        visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_circuit.jpg"))
        if '+' in flag:
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_del.jpg"),
                                      auxiliary_config=auxiliary_config)
        if '++' in flag:
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_del_01.jpg"),
                                      zoom_config=(-0.01, 0.1), auxiliary_config=auxiliary_config)
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_del_001.jpg"),
                                      zoom_config=(-0.01, 0.01), auxiliary_config=auxiliary_config)
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_del_0005.jpg"),
                                      zoom_config=(-0.005, 0.005), auxiliary_config=auxiliary_config)
    if 's' in flag:
        auxiliary_config = {"auxiliary_circuit": art_circuit, "squeeze_ratio": 1}
        visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_art.jpg"),
                                  auxiliary_config=auxiliary_config)
        if '+' in flag:
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_art_01.jpg"),
                                      zoom_config=(-0.01, 0.1), auxiliary_config=auxiliary_config)
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_art_001.jpg"),
                                      zoom_config=(-0.01, 0.01), auxiliary_config=auxiliary_config)
            visualize_circuit_history(art_circuit, sim_config, save_dir=path.join(save_dir, prefix+"art_aux_art_0005.jpg"),
                                      zoom_config=(-0.005, 0.005), auxiliary_config=auxiliary_config)
    # history_config = {"detailed_circuit": del_circuit, "squeeze_ratio": model_config["squeeze"]}
    # visualize_clamp_dynamic(history_config, art_circuit, sim_config, clamp_ratio=0.1,
    #                         save_dir=path.join(output_directory, "clamp_01.jpg"))


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    nr.seed(44)
    # single neuron preparation
    # dataset_squeeze, addtional_squeeze = 4, 5
    dataset_squeeze, addtional_squeeze = 20, 1
    inoise = 0

    circuit_name = "full109"
    config_name = "config_full109"
    # circuit_name = "head20"
    # config_name = "config_head20"
    # circuit_name = "command_circuit"
    # config_name = "config_small"
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
    task_name = f"{circuit_name}_{dataset_squeeze * addtional_squeeze}x_random_{inoise}inoise_GRUMega"
    dataset_name = f"{circuit_name}_{dataset_squeeze}x_random"
    # io_config = {
    #     "input_cells": ["AVBL", "AVBR", 'AVAL', 'AVAR'],
    #     "output_cells": ["PVCL", "PVCR"]}
    # io_config = {
    #     "input_cells": ["AWAL", "AWAR", 'AWCL', 'AWCR'],
    #     "output_cells": ["SAADL", "SAADR", "SAAVL", "SAAVR"]}

    for vis_config in (circuit_name, config_name, task_name, dataset_name, io_config):
        print(vis_config)
    config, abs_circuit, del_circuit, pretrain_dir = single_nrn_preparation(config_name, io_config, task_name,
                                                                            dataset_name, circuit_name, slurm=False)
    # for cell in abs_circuit.cells:
    #     print(vis.cell_info(cell))
    # exit()
    # inception test
    # data_config = {
    #     "factory_name": "random_data_factory",
    #     "args": {
    #         "window_range": (10, 1000),
    #         "volt_range": (-35, 35),
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
            "op_type": 'or'}}
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.4}
    model_config = {
        "model_name": "GRUMega",
        "pretrain_dir": pretrain_dir,
        "args": {"n_layers": 1},
        # "args": {"n_layers": 1},
        "device": torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        "squeeze": dataset_squeeze * addtional_squeeze}
    sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
    for vis_config in (data_config, weight_config, model_config, sim_config):
        print(vis_config)
    fake_weight_config = {
        "syn": (5e-2, 5e-1),
        "gj": (1e-5, 1e-4),
        "inh_prob": 0.7}

    # modify circuit
    nr_seeds = nr.randint(0, 1000, 100)
    for nr_seed in nr_seeds:
        print(f"----Start Inception Seed:{nr_seed}-----")
        nr.seed(nr_seed)
        new_weights = func.circuit_weight_sample(fake_weight_config, del_circuit)
        del_circuit.update_connections(new_weights)
        abs_circuit = transform.detailed2abstract(del_circuit)

        # make artificial circuit
        art_circuit = ArtificialCircuit(abs_circuit, weight_config, model_config)
        input_traces = eval("data_factory." + data_config['factory_name'])(
            num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
        squeeze_input = torch.tensor(data_factory.squeeze_trace(input_traces, model_config['squeeze']), dtype=torch.float32).unsqueeze(0)
        artificial_circuit_inception(art_circuit, squeeze_input, input_traces,
                                     config_name, io_config, sim_config, model_config,
                                     path.join(path.dirname(__file__), "output", f"{task_name}", f"{nr_seed}"), 'ad')
