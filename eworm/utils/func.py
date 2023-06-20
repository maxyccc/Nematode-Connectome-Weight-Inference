import xlrd  # version 1.2.0
import numpy as np
import networkx as nx
import numpy.random as nr
import matplotlib.pyplot as plt
import seaborn
import matplotlib.patches as ptc
import json
import os
import torch
import struct
import math
import os.path as path
from eworm.network.detailed_circuit import DetailedCircuit
from eworm.network.point_circuit import PointCircuit


def load_json(file_name):
    with open(file_name, 'r+') as f:
        data_dic = json.load(f)
    return data_dic


def read_excel(file_name, sheet_name=None):
    wb = xlrd.open_workbook(filename=file_name)
    if not sheet_name:
        sheet = wb.sheet_by_index(0)
    else:
        sheet = wb.sheet_by_name(sheet_name)
    nrow = sheet.nrows
    ncol = sheet.ncols
    return sheet, nrow, ncol


def weights_sample(weight_config, weight_categories, weight_pair_keys):
    """sample weight and make sure that weights of two gapjunction connections
    belongs to one pair are equal"""
    sample_result = []
    gj_weight_buffer = {}  # key--pair_key, value--weight
    active_prob = np.exp2(nr.uniform(2, np.log2(len(weight_categories)))) / len(weight_categories)
    sparse_active = True if (weight_config.get('sparse_active', None) is not None) else False
    for category, pair_key in zip(weight_categories, weight_pair_keys):
        if (weight_config.get('sample_distribution', None) is None) or \
                (weight_config.get('sample_distribution', None).get(category) == 'log'):
            if sparse_active:
                mid_range = (weight_config[category][0] ** 0.5) * (weight_config[category][1] ** 0.5)
                if nr.rand() < active_prob:
                    new_weight = np.exp(nr.uniform(np.log(mid_range), np.log(weight_config[category][1])))
                else:
                    new_weight = np.exp(nr.uniform(np.log(weight_config[category][0]), np.log(mid_range)))
            else:
                new_weight = np.exp(nr.uniform(*np.log(weight_config[category])))
        elif weight_config.get('sample_distribution', None).get(category) == 'uniform':
            new_weight = nr.uniform(*weight_config[category])
        else:
            raise ValueError
        if category == 'gj':
            if gj_weight_buffer.get(pair_key, None) is not None:
                sampled_weight = gj_weight_buffer.get(pair_key)
            else:
                sampled_weight = new_weight
                gj_weight_buffer[pair_key] = sampled_weight
        elif category == 'syn':
            sampled_weight = new_weight
            sampled_weight *= np.sign(nr.rand() - weight_config['inh_prob'])
        else:
            raise ValueError
        sample_result.append(sampled_weight)
    return np.array(sample_result)


def circuit_weight_sample(weight_config, circuit, strength_io=True):
    """weight sampling compatible with Circuit class"""
    connection_categories = [connect.category for connect in circuit.connections]
    connection_pair_keys = [connect.pair_key for connect in circuit.connections]
    new_weights = weights_sample(weight_config, connection_categories, connection_pair_keys)
    if strength_io:
        for connection_idx, connection in enumerate(circuit.connections):
            if connection.pre_segment is None:
                new_weights[connection_idx] = weight_config['syn'][1] * .95
            if connection.post_segment is None:
                new_weights[connection_idx] = weight_config['syn'][1] * .95
    return new_weights


def single_nrn_checker(circuit, pretrain_dir):
    for cell in circuit.cells:
        if not path.exists(path.join(pretrain_dir, cell.name, "valid")):
            return False
    return True


def network_feasibility_check(circuit):
    for cell in circuit.cells:
        print(f"Cell {cell.name} In: {len(cell.pre_connections)} Out:{len(cell.post_connections)}")
        assert len(cell.pre_connections) < 128, \
            f"cell {cell.name} with {len(cell.pre_connections)} in_connect that's too much! (should less than 128)"


def slurm_script_generation(circuit, run_script_dir, save_dir, split_num, **kwargs):
    task_id_line = ",".join([str(task_cell.index) for task_cell in circuit.cells])
    task_id_line = "#SBATCH --array " + task_id_line[:-1] + f"%{split_num}\n"
    run_scripts_line = f"python3 {run_script_dir}" + " --cell_index ${SLURM_ARRAY_TASK_ID}"
    for key in kwargs.keys():
        if kwargs[key] is not None:
            run_scripts_line += f" --{key} {kwargs[key]}"
    run_scripts_line += "\n"
    output_report_line = f"#SBATCH -o {path.join(save_dir, 'reports', 'slurm_%j.out')}\n"
    with open(path.join(save_dir, f"{kwargs['task_name']}.sh"), 'w') as slurm_script:
        slurm_script.writelines([
            "#!/bin/bash\n",
            "##SBATCH --time=400:00:00   # walltime\n",
            "#SBATCH --ntasks=1   # number of processes (i.e. tasks)\n",
            "#SBATCH --nodes=1   # number of nodes\n",
            "#SBATCH --cpus-per-task=32 #of cpu cores per task\n",
            "#SBATCH --gres=gpu:1 #of gpu you need\n",
            "##SBATCH --mem-per-cpu=4096M   # memory per CPU core\n",
            "##SBATCH -w node4 #to submit to a specific node\n",
            "#SBATCH -J \"SingleNrnTrain\"   # job name\n",
            "#SBATCH -p internal\n",
            output_report_line,
            task_id_line,
            "startTime=`date +\"%Y-%m-%d %H:%M:%S\"`\n",
            "echo \"Start time: `date`\"\n",
            "echo \"SLURM_JOB_ID: $SLURM_JOB_ID\"\n",
            "echo \"SLURM_NNODES: $SLURM_NNODES\"\n",
            "echo \"SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE\"\n",
            "echo \"SLURM_NTASKS: $SLURM_NTASKS\"\n",
            "echo \"SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION\"\n",
            "source ~/.bashrc\n",
            "source activate Celegans\n",
            run_scripts_line,
            "endTime=`date +\"%Y-%m-%d %H:%M:%S\"`\n",
            "st=`date -d  \"$startTime\" +%s`\n",
            "et=`date -d  \"$endTime\" +%s`\n",
            "sumTime=$(($et-$st))\n",
            "echo \"Total time is : $sumTime second.\"\n", ])
    os.system(f"sbatch {save_dir}")


def local_script_generation(circuit, run_script_dir, save_dir, split_num, **kwargs):
    cell_idx_list = [int(task_cell.index) for task_cell in circuit.cells]
    nr.shuffle(cell_idx_list)
    device_num = 1
    for file_idx, cell_idx in enumerate(cell_idx_list):
        run_scripts_line = f"python3 {run_script_dir}" + f" --cell_index {cell_idx} --device {file_idx % device_num}"
        for key in kwargs.keys():
            if kwargs[key] is not None:
                run_scripts_line += f" --{key} {kwargs[key]}"
        if file_idx < split_num:
            file_flag = 'w'
        else:
            file_flag = 'a'
            run_scripts_line = " && " + run_scripts_line
        with open(path.join(save_dir, f"{kwargs['task_name']}_file{file_idx % split_num}.sh"),
                  file_flag) as local_script:
            local_script.write(run_scripts_line)
    scripts_dir_list = ["bash " + path.join(save_dir, f"{kwargs['task_name']}_file{file_idx}.sh") for file_idx in
                        range(split_num)]
    # exit()
    # os.system("&".join(scripts_dir_list))
