import xlrd  # version 1.2.0
import numpy as np
import networkx as nx
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as ptc
import json
import os
import struct
import math
from eworm.network.detailed_circuit import DetailedCircuit
from eworm.network.point_circuit import PointCircuit


def visualize_io_data(input_traces, output_traces, save_dir, time_axis=None, input_label=None, output_label=None,
                      x_label=None, y_label=None, title=None, *args):
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    alpha = max(1 / len(input_traces), 0.4)
    for trace_index, trace in enumerate(input_traces):
        time_trace = np.arange(input_traces.shape[-1]) if time_axis is None else time_axis
        label = None if input_label is None else input_label[trace_index]
        plt.plot(time_trace, trace, alpha=alpha, label=label)
    if input_label is not None:
        plt.legend(ncol=4)
    plt.subplot(2, 1, 2)
    alpha = max(1 / len(output_traces), 0.4)
    for trace_index, trace in enumerate(output_traces):
        time_trace = np.arange(output_traces.shape[-1]) if time_axis is None else time_axis
        label = None if output_label is None else output_label[trace_index]
        plt.plot(time_trace, trace, alpha=alpha, label=label)
    if output_label is not None:
        plt.legend(ncol=4)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.tight_layout()
    plt.ylim(-80, 40)
    plt.savefig(save_dir)
    plt.close()


def visualize_neuron_heatmap(input_traces, output_traces, save_dir, time_axis=None, input_label=None, output_label=None,
                             x_label=None, y_label=None, title=None, *args):
    sim_dt = args[0]["dt"]
    step_unit = int(1000 / sim_dt)
    n_neuron, n_step = input_traces.shape
    n_step_del = np.mod(n_step, step_unit)
    time_trace = np.arange(input_traces.shape[-1]) if time_axis is None else time_axis
    input_traces = input_traces[:, :-n_step_del] if np.mod(n_step, step_unit) else input_traces
    input_traces = np.mean(input_traces.reshape((n_neuron, int(n_step / step_unit), step_unit)), axis=-1)
    time_trace = np.arange(0, int(n_step / step_unit), 1)
    n_neuron, n_step = output_traces.shape
    n_step_del = np.mod(n_step, step_unit)
    output_traces = np.mean(output_traces.reshape((n_neuron, int(n_step / step_unit), step_unit)), axis=-1)
    plt.figure(figsize=(int(len(time_trace) / 5), int(n_neuron / 5)))
    plt.subplot(2, 1, 1)
    plt.title("Input")
    seaborn.heatmap(input_traces, xticklabels=time_trace, yticklabels=input_label, cmap='jet')
    plt.subplot(2, 1, 2)
    plt.title("Output")
    seaborn.heatmap(output_traces, xticklabels=time_trace, yticklabels=output_label, cmap='jet')
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


def visualize_train_ckp(input_traces, output_traces, prediction, weights, categories, save_dir):
    plt.figure(figsize=(30, 15))
    plt.subplot(3, 1, 1)
    for trace_index, trace in enumerate(input_traces):
        plt.plot(trace, alpha=0.2, label=f"{categories[trace_index]} {round(weights[trace_index], 5)}")
    for trace_index, trace in enumerate(output_traces):
        plt.plot(trace, alpha=0.4, color='b')
    plt.legend()
    plt.title("Input & Output (blue color)")
    plt.subplot(3, 1, 2)
    for trace_index, trace in enumerate(output_traces):
        plt.plot(trace, alpha=0.4, color='b')
    for trace_index, trace in enumerate(prediction):
        plt.plot(trace, alpha=0.4, color='r')
    plt.title("Output (blue color) & Prediction (red color)")
    plt.subplot(3, 1, 3)
    for trace_index, trace in enumerate(input_traces):
        plt.plot(trace, alpha=0.2, label=f"{categories[trace_index]} {round(weights[trace_index], 5)}")
    for trace_index, trace in enumerate(prediction):
        plt.plot(trace, alpha=0.4, color='r')
    plt.legend(ncol=4)
    plt.title("Input & Prediction (red color)")
    plt.tight_layout()
    plt.savefig(save_dir)
    plt.close()


def export_for_neuronXcore(circuit, save_path, group_name='group1'):
    """export morphology data (.swc), synapse data (.snp), voltage data (.vtg) for neuronXcore

    Arguments:
       circuit: DetailedCircuit class, a neural circuit
       save_path: string, save three file to this path
       group_name: string, saved file name
    """
    assert isinstance(circuit, (DetailedCircuit, PointCircuit))
    assert circuit.cells[0].rec_v is True
    os.makedirs(save_path, exist_ok=True)
    morphology_path = os.path.join(save_path, group_name + '.swc')
    synapse_path = os.path.join(save_path, group_name + '.snp')
    voltage_path = os.path.join(save_path, group_name + '.vtg')

    ipoint = 1
    cnt = 0
    id_dict = {}  # key---(cell_id, point_id), value---global index
    # write swc file
    with open(morphology_path, "w", encoding="utf-8") as swc_file:
        for cell in circuit.cells:
            for point in cell.points:
                pid = (point.parent_id + ipoint) if (point.parent_id != -1) else point.parent_id
                swc_file.writelines(
                    f"{ipoint + point.index:d} {point.category:d} {point.location[0]:.4f} {point.location[1]:.4f} {point.location[2]:.4f} {point.diameter:.4f} {pid:d}\n")
                cnt += 1
                if (cell.index, point.index) not in id_dict:
                    id_dict[(cell.index, point.index)] = ipoint + point.index
                else:
                    assert id_dict[(cell.index, point.index)] == ipoint + point.index
            ipoint += len(cell.points)
    print(f"saved morphology to {morphology_path:s}")
    # write snp file
    with open(synapse_path, 'w', encoding="utf-8") as snp_file:
        cate_dic = {'gj': 0, 'syn': 1}
        for conn in circuit.connections:
            if not conn.pre_segment or not conn.post_segment:
                continue
            pre_id = id_dict[(conn.pre_cell.index, conn.pre_segment.point2.index)]
            post_id = id_dict[(conn.post_cell.index, conn.post_segment.point2.index)]
            # cate: gj-0, exc_syn-1, inh_syn-2
            cate = cate_dic[conn.category] if conn.weight > 0 else cate_dic[conn.category] + 1
            snp_file.write(f"{pre_id:d} {post_id:d} {cate:d} 1\n")
    print(f"saved synapse info to {synapse_path:s}")
    # write vtg file
    with open(voltage_path, 'wb') as voltage_file:
        time_step = len(circuit.cells[0].segments[0].voltage.as_numpy())
        voltage_file.write(struct.pack("i", time_step))
    cnt = 0
    with open(voltage_path, 'ab') as voltage_file:
        for cell in circuit.cells:
            for seg in cell.segments:
                volt = seg.voltage.as_numpy().astype('float32')
                voltage_file.write(volt.tobytes())
                cnt += 1
                if seg.index == 1:
                    voltage_file.write(volt.tobytes())
                    cnt += 1
    print(f"saved voltage to {voltage_path:s}")


def export_volt_figure(circuit, sim_config, save_path):
    """save figures of all neurons (red: soma trace, gray: other segments)

    Arguments:
       circuit: DetailedCircuit class, a neural circuit
       sim_config: dictionary, simulation configuration
       save_path: string, save figure to this path
    """
    assert isinstance(circuit, DetailedCircuit)
    assert circuit.cells[0].segments[0].voltage is not None
    os.makedirs(save_path, exist_ok=True)
    trace_len = int(sim_config['tstop'] / sim_config['dt']) + 1
    time_vector = np.linspace(0, sim_config['tstop'], trace_len)
    for cell in circuit.cells:
        plt.figure(figsize=(20, 2))
        for seg in cell.segments:
            plt.plot(time_vector, seg.voltage, linewidth=1, color=[0.8, 0.8, 0.8])
        plt.plot(time_vector, cell.segments[0].voltage, linewidth=1, color=[1, 0, 0])
        plt.ylabel("voltage (mV)")
        plt.xlabel("time (ms)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, str(cell.index) + "_" + cell.name + '.png'))
        plt.close()
    print(f"saved figures to {save_path:s}")


def watch_neuron_io(circuit, input_traces, input_cell_names, sim_config, save_path, watch_neuron_name=None):
    """save figure of neuron's input and its output voltage (blue: input traces, red: ouput traces)

    Arguments:
       circuit: DetailedCircuit class, a neural circuit
       input_traces: 2d array, input traces of this neural circuit
       input_cell_names: list of string, input cell names
       sim_config: dictionary, simulation configuration
       save_path: string, save figure to this path
       watch_neuron_name: list of string, name of interested neurons
    """
    assert isinstance(circuit, DetailedCircuit)
    assert circuit.cells[0].segments[0].voltage is not None

    if not watch_neuron_name:
        watch_neuron = circuit.cells
    else:
        watch_neuron = [circuit.cell(cell_name=x) for x in watch_neuron_name]
    input_index = dict(zip(input_cell_names, range(len(input_cell_names))))
    trace_length = int(sim_config['tstop'] / sim_config['dt'])
    subplot_nrow = int(np.ceil(math.sqrt(len(watch_neuron))))
    subplot_ncol = int(np.round(math.sqrt(len(watch_neuron))))
    plt.figure(figsize=(subplot_nrow * 20, subplot_ncol * 16))
    for i, cell in enumerate(watch_neuron):
        plt.subplot(subplot_nrow, subplot_ncol, i + 1)
        for connection in cell.pre_connections:
            if not connection.pre_segment:
                plt.plot(np.linspace(0, sim_config['tstop'], trace_length),
                         input_traces[input_index[connection.post_cell.name]], 'b', alpha=0.1)
            else:
                plt.plot(np.linspace(0, sim_config['tstop'], trace_length + 1), connection.pre_segment.voltage, 'b',
                         alpha=0.1)
        for connection in cell.post_connections:
            plt.plot(np.linspace(0, sim_config['tstop'], trace_length + 1), connection.pre_segment.voltage, 'r',
                     alpha=0.1)
            plt.title(cell.name)
    plt.savefig(save_path)


def visualize_loss(loss_rec, loss_mean_rec, save_dir, log=True):
    plt.figure()
    if log:
        plt.plot([np.log(loss) for loss in loss_rec])
        plt.plot([np.log(loss) for loss in loss_mean_rec])
        plt.ylabel('ln(loss)')
    else:
        plt.plot(loss_rec)
        plt.plot(loss_mean_rec)
        plt.ylabel('loss')
    plt.xlabel('iterations')
    plt.savefig(save_dir)
    plt.close()


def cell_info(cell):
    info_str = f"No. {cell.index} Cell {cell.name} In: {len(cell.pre_connections)} Out: {len(cell.post_connections)}"
    return info_str


def connection_info(connection):
    if connection.pre_cell is None:
        info_str = f"Input {connection.category} Towards {connection.post_cell.name}, Weight {connection.weight}"
    elif connection.post_cell is None:
        info_str = f"Output {connection.category} From {connection.pre_cell.name}, Weight {connection.weight}"
    else:
        info_str = f"Intermediate {connection.category} From {connection.pre_cell.name} towards {connection.post_cell.name}, Weight {connection.weight}"
    return info_str


def visualize_circuit(circuit, save_dir, layout="planar"):
    # preparation
    graph = nx.MultiGraph()
    graph.add_nodes_from([(cell.index, {"name": cell.name}) for cell in circuit.cells])
    graph.add_nodes_from([(1001, {"name": "Input"}), (1002, {"name": "Output"})])
    gj_weights, syn_weights = [], []
    for connection in circuit.connections:
        if connection.category == "gj":
            gj_weights.append(connection.weight)
        elif connection.category == 'syn':
            syn_weights.append(abs(connection.weight))
            if np.isnan(syn_weights[-1]):
                print(f"Nan warning!: {connection_info(connection)}")
        else:
            raise ValueError
    gj_range = (np.max(gj_weights), np.min(gj_weights) * 0.99) if len(gj_weights) > 0 else None
    syn_range = (np.max(syn_weights), np.min(syn_weights) * 0.99) if len(syn_weights) > 0 else None

    for connection in circuit.connections:
        pre_cell = 1001 if connection.pre_cell is None else connection.pre_cell.index
        post_cell = 1002 if connection.post_cell is None else connection.post_cell.index
        if connection.category == 'gj':
            lw = (connection.weight - gj_range[1]) / (gj_range[0] - gj_range[1])
        elif connection.category == 'syn':
            lw = (abs(connection.weight) - syn_range[1]) / (syn_range[0] - syn_range[1])
        else:
            raise ValueError
        graph.add_edge(pre_cell, post_cell, category=connection.category, lw=lw + 0.1, weight=connection.weight,
                       direction=int(pre_cell) < int(post_cell))
    # Plot
    plt.figure(figsize=(100, 100))
    fig, ax = plt.subplots()
    pos = eval(f"nx.{layout}_layout")(graph)
    node_labels = nx.get_node_attributes(graph, 'name')
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    # nx.draw_networkx_labels(graph, pos, ax=ax, labels=node_labels)
    ax = plt.gca()
    category = nx.get_edge_attributes(graph, 'category')
    weight = nx.get_edge_attributes(graph, 'weight')
    lw = nx.get_edge_attributes(graph, 'lw')
    direction = nx.get_edge_attributes(graph, 'direction')
    for e in graph.edges:
        if category[e] == 'gj':
            arrow_sty, arrow_lw, arrow_fc, arrow_ls = ptc.ArrowStyle("-"), lw[e] * 0.6, (0, 0, 1, 0.5), "--"
        elif weight[e] > 0:
            arrow_sty, arrow_lw, arrow_fc, arrow_ls = ptc.ArrowStyle("wedge", tail_width=lw[e] * 0.4), 0., (
                1, 0, 0, 0.5), "-"
        else:
            arrow_sty, arrow_lw, arrow_fc, arrow_ls = ptc.ArrowStyle("wedge", tail_width=lw[e] * 0.4), 0., (
                0, 0, 1, 0.5), "-"
        (xy, xytext) = (pos[e[1]], pos[e[0]]) if direction[e] else (pos[e[0]], pos[e[1]])
        rad = 0.05 * e[2] * np.sign((e[2] % 2) - 0.5) * np.sign(int(direction[e]) - 0.5)
        ax.annotate("", xy=xy, xycoords='data', xytext=xytext, textcoords='data',
                    arrowprops=dict(arrowstyle=arrow_sty, shrinkA=5, shrinkB=10, lw=arrow_lw, fc=arrow_fc, ls=arrow_ls,
                                    connectionstyle=f"arc3,rad={rad}"))
    nx.draw_networkx_labels(graph, pos, ax=ax, labels=node_labels, font_size=5)
    plt.axis('off')
    plt.savefig(save_dir, bbox_inches='tight', dpi=500)
    plt.close()


def vis_muscle_io(input_data, voltage_data, cell_names, muscle_data, x_dt=10, gt_muscle=None, save_dir=None):
    fig, ax = plt.subplots(figsize=(30, 15), nrows=7, ncols=2, sharex='all')
    # fig, ax = plt.subplots(figsize=(30, 15), nrows=2, ncols=2, sharex='all', gridspec_kw={'height_ratios': [5, 20]})
    voltage_traces = np.array(voltage_data.detach().cpu()+0.)
    input_traces = np.array(input_data.detach().cpu())
    muscle_traces = np.array(muscle_data.detach().cpu())
    ax1, ax2, ax3, ax4, ax5 = plt.subplot2grid((7, 2), (0, 0)), plt.subplot2grid((7, 2), (1, 0), rowspan=6),\
                              plt.subplot2grid((7, 2), (0, 1), rowspan=3), plt.subplot2grid((7, 2), (3, 1), rowspan=3),\
                              plt.subplot2grid((7, 2), (6, 1))
    for trace_id in range(input_traces.shape[0]):
        ax1.plot(input_traces[trace_id], alpha=0.7, lw=0.5)
    ax3.set_title('MusclePrediction')
    shift_len = 24 if muscle_traces.shape[0] == 96 else 1
    shift_value = 1.
    for trace_id in range(muscle_traces.shape[0]):
        ax3.plot(muscle_traces[trace_id] + int(trace_id / shift_len) * shift_value, alpha=0.7, lw=0.7)
    if gt_muscle is not None:
        gt_traces = np.array(gt_muscle.detach().cpu())
        ax4.set_title('GroundTruth')
        for trace_id in range(muscle_traces.shape[0]):
            ax4.plot(gt_traces[trace_id] + int(trace_id / shift_len) * shift_value, alpha=0.7, lw=0.7)
            ax5.plot(gt_traces[trace_id] - muscle_traces[trace_id] + int(trace_id / shift_len) * shift_value/2, alpha=0.7, lw=0.7)
    else:
        ax4.remove()
        ax5.remove()
    threads_plot(data=voltage_traces, y_tick_labels=cell_names, x_dt=x_dt, ax=ax2)
    if save_dir is None:
        plt.show()
    else:
        plt.savefig(save_dir, dpi=300)


def threads_plot(data, y_tick_labels, x_dt, ax, overlap_num=2., xlabel="Time (s)", ylabel="Neurons"):
    """
    data: np.array with shape (n_traces, num_step)
    y_tick_labels: list of string, label for each trace
    x_dt: time for each step
    ax: axes to plot
    overlap_num: density of traces overlapping with each other, 1 for no overlapping, 2 for half overlapping
    """
    max_char_len = np.max([len(char_tmp) for char_tmp in y_tick_labels])
    parsed_cell_names = []
    for cell_id in range(len(y_tick_labels)):
        if cell_id % 2 == 0:
            parsed_cell_names.append(y_tick_labels[cell_id].ljust(2 * max_char_len, " ").rjust(4 * max_char_len, " "))
        else:
            parsed_cell_names.append(y_tick_labels[cell_id].ljust(2 * max_char_len, " ").ljust(4 * max_char_len, " "))
    data = np.nan_to_num(data, nan=0., posinf=100., neginf=-100.)
    max_shift = max(np.max(data) - np.min(data), 1) / overlap_num
    xtick_len = int(np.ceil(int(data.shape[-1] / 5) / 500) * 500)
    xtick_len_minor = int(xtick_len / 10)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for trace_id, trace in enumerate(data):
        ax.plot(trace / max_shift + trace_id, lw=0.6, alpha=0.8)
        ax.hlines(trace_id, xmin=0, xmax=data.shape[-1], colors=colors[trace_id % len(colors)], ls='dashed',
                  lw=0.4, alpha=0.6)
    ax.set_xticks(ticks=np.arange(0, data.shape[-1], xtick_len),
                  labels=np.array(np.arange(0, data.shape[-1], xtick_len) * x_dt, dtype=int))
    ax.set_xticks(ticks=np.arange(0, data.shape[-1], xtick_len_minor),
                  labels=np.array(np.arange(0, data.shape[-1], xtick_len_minor) * x_dt, dtype=int), minor=True)
    ax.set_yticks(ticks=np.arange(data.shape[0]), labels=parsed_cell_names, fontsize=6)
    ax.tick_params(axis='y', length=2., direction='in')
    for ytick_id, ytick in enumerate(ax.get_yticklabels()):
        ytick.set_color(colors[ytick_id % len(colors)])
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', which='minor', alpha=0.2)
    ax.grid(axis='x', which='major', alpha=0.5)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # ax.set_xlim(-xtick_len_minor, data.shape[-1] + xtick_len_minor)
    # ax.set_ylim(-2, data.shape[0])