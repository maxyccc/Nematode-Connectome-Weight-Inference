import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import sys
import os
import _pickle as pickle
from eworm.network import *
from eworm.utils import *
from eworm.single_nrn_train import *
from eworm.network_train import *
import numpy.random as nr

if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    nr.seed(42)

    circuit_name = "full109"
    config_name = "config_full109"

    config_dir = path.join(path.dirname(__file__), "configs", config_name + ".json")
    circuit_dir = path.join(path.dirname(__file__), "circuits")
    #
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
            "op_type": 'xor'}}
    # data_config = {
    #     "factory_name": "null_factory",
    #     "args": {
    #         "window_range": (10, 1000),
    #         "volt_range": (-75, 5),
    #         "noise_settings": ((80, 60), (20, 20), (180, 60), (40, 15)),
    #         "reverse_noise_setting": None}}
    sim_config = {"dt": 0.5, "tstop": 20000, "v_init": -65, "secondorder": 0}

    # make circuit
    config = func.load_json(config_dir)
    io_config = config["io"]
    abs_circuit = pickle.load(open(path.join(circuit_dir, circuit_name + ".pkl"), "rb"))

    if "null" in data_config['factory_name']:
        new_connections = []
        for connection in abs_circuit.connections:
            if connection.pre_segment is not None:
                new_connections.append(connection)
        abs_circuit.connections = new_connections

    del_circuit = transform.abstract2detailed(abs_circuit, config)
    # func.network_feasibility_check(abs_circuit)

    test_name = f"{circuit_name}_{data_config['factory_name']}"
    seeds = nr.randint(0, 1000, 100)
    working_directory = path.join(path.dirname(__file__), "inspect", test_name)
    os.makedirs(working_directory, exist_ok=True)
    vis.visualize_circuit(abs_circuit, save_dir=path.join(working_directory,
                                                          circuit_name + f"_connectome.jpg"), layout="circular")

    # optimal_input = np.load(path.join(path.dirname(__file__), "..", "tmp_data", "x_optimal_eworm_v4.npy"))
    # optimal_input = (data_factory.squeeze_trace(optimal_input, 0.5)/0.2)*25.-25
    # _ = del_circuit.simulation(sim_config, optimal_input, io_config["input_cells"], io_config["output_cells"],
    #                            make_history=True)
    # network_inception.visualize_circuit_history(
    #     del_circuit, sim_config,
    #     path.join(working_directory, f"{circuit_name}_input_optimal.jpg"))
    # exit()
    for input_seed in seeds:
        nr.seed(input_seed)
        if "null" in data_config['factory_name']:
            test_input = eval("data_factory.random_data_factory")(
                num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'],
                **data_config['args'])
            test_input = test_input*0-50
            print("null!")
        else:
            test_input = eval("data_factory." + data_config['factory_name'])(
                num=len(io_config["input_cells"]), tstop=sim_config['tstop'], dt=sim_config['dt'], **data_config['args'])
        _ = del_circuit.simulation(sim_config, test_input, io_config["input_cells"], io_config["output_cells"],
                                   make_history=True)
        network_inception.visualize_circuit_history(
            del_circuit, sim_config,
            path.join(working_directory, f"{circuit_name}_input_{input_seed}.jpg"))
