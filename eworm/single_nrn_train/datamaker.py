"""
Functions work for:
    1. data_make: Give Cell instance, generate input-output voltage traces datasets

sim_config
    dt
    tstop
    v_init
dataset_config
    total_data_num
    file_data_num
    thread_num
data_config
    factory_name
    factory_args (except num, simulation_duration, dt)
        ...
        ...
        ...
weight_config
    syn
    gj
"""
import os
# import pymp
import numpy as np
import numpy.random as nr
import os.path as path
import _pickle as pickle
from tqdm import tqdm
from eworm.network import *
from eworm.utils import *


def data_make(ref_circuit, cell_config, sim_config, config,
              save_dir=None, generate_config=None, data_config=None, weight_config=None):
    """
    main function for single neuron training data generation
    Args:
        ref_circuit: reference DetailedCircuit class
        cell_config: cell information (name, index)
        config: general hoc/mechanism loading config
        sim_config: config dictionary
        generate_config: config dictionary
        data_config: config dictionary
        weight_config: config dictionary
        save_dir: save directory

    Returns:
        the last file data dictionary
    """
    generate_config = {
        "total_file": 100,
        "data_per_file": 1000,
        "thread_num": 32,
        'squeeze_ratio': 10,
    } if generate_config is None else generate_config
    data_config = {
        "factory_name": "c302_data_factory",
        "args": {
            'pause_prob': 0.7,
            'initial_pause': 500,
            'noise_amp': 80,
            'smooth_sigma': 60,
        }
    } if data_config is None else data_config
    weight_config = {
        "syn": (5e-2, 5),
        "gj": (1e-5, 1e-3),
        "inh_prob": 0.5,
    } if weight_config is None else weight_config
    # preparation
    try:
        assert isinstance(ref_circuit, abstract_circuit.AbstractCircuit)
    except AssertionError:
        if isinstance(ref_circuit, detailed_circuit.DetailedCircuit):
            new_circuit = transform.detailed2abstract(ref_circuit)
            ref_circuit.release_hoc()
            del ref_circuit
            ref_circuit = new_circuit
            print("reference circuit being detailed may cause low efficiency")
        else:
            raise ValueError("reference circuit is not compatible")

    ref_cell = ref_circuit.cell(**cell_config)
    cell_name = ref_cell.name
    connection_categories = [connect.category for connect in ref_cell.pre_connections]
    connection_pair_keys = [connect.pair_key for connect in ref_cell.pre_connections]
    total_file, data_per_file = generate_config['total_file'], generate_config['data_per_file']
    in_num, squeeze_ratio = len(ref_cell.pre_connections), generate_config['squeeze_ratio']
    out_num = len(np.unique([post_connection.pre_segment.index for post_connection in ref_cell.post_connections]))
    trace_len = int(sim_config['tstop'] / (sim_config['dt'] * squeeze_ratio))
    seeds = nr.randint(0, total_file * 1000, total_file)
    assert save_dir is not None
    os.makedirs(path.join(save_dir, cell_name), exist_ok=True)
    # generate data
    with pymp.Parallel(generate_config['thread_num']) as p:
        for file_index in p.range(total_file):
            input_traces = np.zeros([data_per_file, in_num, trace_len], dtype='float32')
            output_traces = np.zeros([data_per_file, out_num, trace_len], dtype='float32')
            connection_weights = np.zeros([data_per_file, in_num], dtype='float32')
            selected_cell, _ = transform.select_cell(ref_circuit, config, **cell_config, load_hoc=True)
            nr.seed(seeds[file_index])
            for data_index in tqdm(range(data_per_file), mininterval=10) if p.thread_num == 0 else range(data_per_file):
                input_trace = eval("data_factory."+data_config['factory_name'])(
                        num=in_num, tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
                new_weights = func.weights_sample(weight_config, connection_categories, connection_pair_keys)
                selected_cell.update_connections(new_weights)
                output_trace = selected_cell.simulation(sim_config, input_trace)
                input_traces[data_index] = data_factory.squeeze_trace(input_trace, squeeze_ratio)
                output_traces[data_index] = data_factory.squeeze_trace(output_trace, squeeze_ratio)
                connection_weights[data_index] = new_weights
            pickle.dump({"input_traces": input_traces,
                         "output_traces": output_traces,
                         "connection_weights": connection_weights},
                        open(path.join(save_dir, cell_name, f"{cell_name}_{file_index}.dat"), "wb"),
                        protocol=2)
            print(f"File #{file_index} dumped!")
            if file_index & (file_index-1) == 0:
                for visual_index in range(int(np.floor(np.log10(data_per_file)))+1):
                    data_index = (10 ** visual_index)-1
                    vis.visualize_io_data(
                        input_traces[data_index], output_traces[data_index],
                        save_dir=path.join(save_dir, cell_name, f"{file_index}_{data_index}_{cell_name}_sample.jpg"),
                        time_axis=np.linspace(sim_config['dt'], sim_config['tstop'], trace_len),
                        input_label=[f"{category}  {round(weight, 5)}" for weight, category in
                                     zip(connection_weights[data_index], connection_categories)],
                        output_label=None,
                        x_label="Time (ms)", y_label="Voltage (mV)", title=f"File#{file_index}_Data#{data_index}_io")
                print(f"File #{file_index} visualized")

#
# if __name__ == "__main__":
#     np.random.seed(42)
#     config_dir = path.join(path.dirname(__file__), "config.json")
#     config = func.load_json(config_dir)
#     input_cells = ["AWAL", "AWAR", "AWCL", "AWCR"]
#     sim_config = {"dt": 0.5, "tstop": 10000, "v_init": -65, "secondorder": 0}
#     generate_config = {
#         "total_file": 100,
#         "data_per_file": 2000,
#         "thread_num": 32,
#         'squeeze_ratio': 10,
#     }
#     cell_config = {
#         "cell_index": None,
#         "cell_name": "AWAL"
#     }
#     ref_circuit1 = pickle.load(open("./ref_circuit.pkl", "rb"))
#     # ref_circuit1 = transform.config2detailed(config, input_cells)
#     # selected_cell, selected_circuit = transform.select_cell(ref_circuit1, config, **cell_config)
#     # print(f"{selected_cell.name} {selected_cell.index} {len(selected_cell.pre_connections)} {len(selected_cell.post_connections)}")
#     data_make(ref_circuit1, cell_config, sim_config, config,
#               save_dir=path.join(path.dirname(__file__), 'output', "test"), generate_config=generate_config)
