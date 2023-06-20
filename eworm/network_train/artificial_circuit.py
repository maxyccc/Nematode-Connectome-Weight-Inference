"""
Modules create a abstract circuit
    1. load pretrained ANN as cell
    2. run ANN prediction in "Circuit" way
    3. change between
        (fixed parameter, trainable weight connection)
        (trainable parameter, fixed weight connection)
"""
import torch
import glob
import time
import numpy as np
import torch
import os.path as path
import torch.nn as nn
from tqdm import tqdm
from eworm.network import *
from eworm.single_nrn_train import *
from eworm.utils import *
from concurrent.futures import ThreadPoolExecutor, wait


def mini_forward(cell, time_step, clamp):
    cell.prediction, cell.hidden = cell.model(
        (torch.stack([pre_cnt.history[time_step] for pre_cnt in cell.pre_connections], dim=1) + 25) / 75,
        cell.input_weights, init=cell.hidden)
    cell.prediction = cell.prediction * 75 - 25
    if clamp:
        cell.prediction = torch.clamp(cell.prediction, min=-100, max=50)


class ArtificialCircuit(nn.Module):
    def __init__(self, ref_circuit, weight_config, model_config, devices=None):
        assert isinstance(ref_circuit, abstract_circuit.AbstractCircuit)
        super(ArtificialCircuit, self).__init__()
        self.circuit, self.weight_config = ref_circuit, weight_config
        if devices is None:
            devices = [torch.device("cuda")] if torch.cuda.is_available() else torch.device("cpu")
        self.output_device = devices[0]
        self.mode = "train"
        self.devices = {}
        for cell_enu, cell in enumerate(self.circuit.cells):
            self.devices[cell.name] = devices[cell_enu % len(devices)]
        self.load_cells(model_config)
        self.prepare_weights(random_init=False)

    def get_device(self, cell=None):
        if cell is not None:
            return self.devices[cell.name]
        else:
            return self.output_device

    def load_cells(self, model_config, fixed_param=True):
        for cell in self.circuit.cells:
            cell.output_indices = np.unique([post_cnt.pre_segment.index for post_cnt in cell.post_connections])
            cell_args = {
                "in_channel": len(cell.pre_connections),
                "out_channel": len(cell.output_indices)}
            if model_config["args"] is not None:
                cell_args = {**cell_args, **model_config["args"]}
            cell.model = eval("model." + model_config["model_name"])(**cell_args).to(self.get_device(cell))
            pretrain_dir = model_config["pretrain_dir"]
            if pretrain_dir is not None:
                model_ckp_list = sorted(glob.glob(path.join(pretrain_dir, cell.name, 'train', 'model_*.ckp')),
                                        key=lambda x: int(x.split('model_epoch#')[-1].split('.ckp')[0]))
                print(f"Loading pretrained #{cell.name} checkpoint from {model_ckp_list[-1]} deployed to {self.get_device(cell)}")
                cell.model.load_state_dict(torch.load(model_ckp_list[-1], map_location=self.get_device(cell)))
            if fixed_param:
                for param in cell.model.parameters():
                    param.requires_grad = False

    def prepare_weights(self, random_init=True):
        connection_categories = [connection.category for connection in self.circuit.connections]
        connection_pair_keys = [connection.pair_key for connection in self.circuit.connections]
        if random_init:
            new_weights = func.circuit_weight_sample(self.weight_config, self.circuit)
            self.circuit.update_connections(new_weights)
        connection_weights = np.array([connect.weight for connect in self.circuit.connections])
        meta_weights = data_factory.sample2meta(connection_weights, connection_categories, self.weight_config, 'np')
        gj_param_buffer = {}
        for cnt_index, connection in enumerate(self.circuit.connections):
            if (connection.category == 'gj') and (gj_param_buffer.get(connection.pair_key, None) is not None):
                connection.meta_weight = gj_param_buffer.get(connection.pair_key).to(
                    self.get_device(connection.post_cell))
            else:
                connection.meta_weight = nn.Parameter(
                    torch.tensor([meta_weights[cnt_index]], dtype=torch.float32,
                                 device=self.get_device(connection.post_cell)),
                    requires_grad=True)
                if connection.category == 'gj':
                    gj_param_buffer[connection.pair_key] = connection.meta_weight

    def refresh_meta_weights(self):
        gj_param_buffer = {}
        for connection in self.circuit.connections:
            if (connection.category == 'gj') and (gj_param_buffer.get(connection.pair_key, None) is not None):
                connection.meta_weight = gj_param_buffer.get(connection.pair_key).to(
                    self.get_device(connection.post_cell))
            else:
                if connection.category == 'gj':
                    assert connection.meta_weight.is_leaf
                    gj_param_buffer[connection.pair_key] = connection.meta_weight

    def sim_init(self, batch_size, init_setting=None):
        if init_setting is None:
            init_v = -65.
            for connection in self.circuit.connections:
                connection.history = [torch.ones(batch_size, 1).to(self.get_device(connection.post_cell)) * init_v]
            for cell in self.circuit.cells:
                cell.hidden = None
        else:
            assert init_setting["history"].shape == (batch_size, len(self.circuit.connections), 1)
            for connection_index, connection in enumerate(self.circuit.connections):
                connection.history = [
                    init_setting["history"][:, connection_index].to(self.get_device(connection.post_cell))]
            for cell_index, cell in enumerate(self.circuit.cells):
                cell.hidden = init_setting["hidden"][cell_index].to(self.get_device(cell))

    def forward(self, input_traces, input_cell_names, output_cell_names, augment_config=None, init_setting=None):
        # preparation
        assert input_traces.ndim == 3 and input_traces.shape[1] == len(input_cell_names), \
            f"input traces has dimension {input_traces.ndim}; " \
            f"input traces shape[1] {input_traces.shape[1]} contradict to input cell names {input_cell_names}"
        if self.mode == "train":
            self.refresh_meta_weights()
        batch_size, _, trace_len = input_traces.shape
        input_index = dict(zip(input_cell_names, range(len(input_cell_names))))
        output_index = dict(zip(output_cell_names, range(len(output_cell_names))))
        augment_config = {"clamp": True, "noise": 0} if augment_config is None else augment_config
        self.sim_init(batch_size, init_setting)
        for input_cnt in self.circuit.input_connections:
            if input_cnt.post_cell.name in input_index.keys():
                input_cnt.history = [input_traces[:, input_index[input_cnt.post_cell.name], t_step:t_step + 1].to(
                    self.get_device(input_cnt.post_cell)) for t_step in range(trace_len)]
            else:
                input_cnt.history = [-60.*torch.ones(input_traces.shape[0], 1).to(self.get_device(input_cnt.post_cell))
                                     for _ in range(trace_len)]
        # simulation
        for cell in self.circuit.cells:
            meta_weights = torch.cat([pre_connection.meta_weight for pre_connection in cell.pre_connections])
            connection_categories = [pre_connection.category for pre_connection in cell.pre_connections]
            cell.input_weights = data_factory.meta2input(meta_weights, connection_categories).to(
                self.get_device(cell)).unsqueeze(0).expand(batch_size, -1)
        for time_step in tqdm(range(trace_len), mininterval=10):
            # with ThreadPoolExecutor() as executor:
            #     futures = [executor.submit(mini_forward, cell, time_step, augment_config["clamp"]) for cell in
            #                self.circuit.cells]
            # else:
            for cell in self.circuit.cells:
                mini_forward(cell, time_step, augment_config["clamp"])
            for cell in self.circuit.cells:
                cell.prediction += torch.randn(cell.prediction.shape).to(self.get_device(cell)) * augment_config[
                    "noise"]
                # assign output
                for output_idx, output_segment_idx in enumerate(cell.output_indices):
                    for post_connection in cell.segment(output_segment_idx).post_connections:
                        post_connection.history.append(
                            cell.prediction[:, output_idx].to(self.get_device(post_connection.post_cell)))
        # prepare history and output traces
        for connection in self.circuit.connections:
            connection.history = torch.cat(connection.history, dim=-1)
            if connection.pre_segment is not None:
                connection.history = connection.history[..., :-1]
        output_traces = torch.zeros((batch_size, len(output_cell_names), trace_len)).to(self.output_device)
        for output_cnt in self.circuit.output_connections:
            if output_cnt.pre_cell.name in output_index.keys():
                output_traces[:, output_index[output_cnt.pre_cell.name]] = output_cnt.history.to(self.output_device)
        return output_traces

    def extract_connection_weights(self):
        abs_circuit = abstract_circuit.AbstractCircuit()
        for cell in self.circuit.cells:
            abs_cell = abstract_circuit.AbsCell(index=cell.index, name=cell.name)
            for segment in cell.segments:
                abs_cell.add_segment(
                    abstract_circuit.AbsSegment(index=segment.index, cell=abs_cell, name=segment.name))
            abs_circuit.add_cell(abs_cell)
        meta_weights = np.array(torch.cat([connection.meta_weight.data.detach().cpu()
                                           for connection in self.circuit.connections]))
        connection_categories = [connection.category for connection in self.circuit.connections]
        sample_weights = data_factory.meta2sample(meta_weights, connection_categories, self.weight_config, "np")
        for connection_idx, connection in enumerate(self.circuit.connections):
            if connection.pre_segment is None:
                abs_pre_segment = None
            else:
                pre_cell = abs_circuit.cell(connection.pre_cell.index)
                abs_pre_segment = pre_cell.segment(connection.pre_segment.index)
            if connection.post_segment is None:
                abs_post_segment = None
            else:
                post_cell = abs_circuit.cell(connection.post_cell.index)
                abs_post_segment = post_cell.segment(connection.post_segment.index)
            abs_circuit.add_connection(abstract_circuit.AbsConnection(
                pre_segment=abs_pre_segment, post_segment=abs_post_segment, category=connection.category,
                weight=sample_weights[connection_idx], pair_key=connection.pair_key))
        return abs_circuit

    def mode_switch(self, mode):
        assert mode in ("train", "test")
        for cell in self.circuit.cells:
            for param in cell.model.parameters():
                param.requires_grad = False
        self.mode = mode
        for connection in self.circuit.connections:
            if connection.meta_weight.is_leaf:
                connection.meta_weight.requires_grad = True if mode == "train" else False

    def fetch_meta_weights(self):
        meta_weights = []
        gj_keys = set()
        for connection in self.circuit.connections:
            if (connection.pre_cell is not None) and (connection.post_cell is not None):
                if (connection.category == 'gj') and (connection.pair_key not in gj_keys):
                    if connection.meta_weight.is_leaf:
                        meta_weights.append(connection.meta_weight)
                        gj_keys.add(connection.pair_key)
                elif connection.category == 'syn':
                    if connection.meta_weight.is_leaf:
                        meta_weights.append(connection.meta_weight)
        return meta_weights
