import os
import time
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import timeit


def check_availability():
    print(cuda.is_available())
    print(cuda.device_count())
    for device_id in range(cuda.device_count()):
        print(device_id, cuda.get_device_name(device_id), cuda.get_device_capability(device_id))


class ToyModel(nn.Module):
    def __init__(self, device1, device2, device3, sequence, sep=False):
        super(ToyModel, self).__init__()
        self.sequence, self.sep = sequence, sep
        self.device1, self.device2, self.device3 = device1, device2, device3
        self.model1 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1).to(device1)
        self.model2 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1).to(device2)
        self.model3 = nn.Conv2d(16, 16, kernel_size=(3, 3), padding=1).to(device3)

    def forward(self, x, y, z, mp_):
        if not self.sequence:
            if mp_:
                s1, s2, s3 = cuda.Stream(self.device1), cuda.Stream(self.device2), cuda.Stream(self.device3)
                with cuda.stream(s1):
                    u = self.model1(x)
                with cuda.stream(s2):
                    v = self.model2(y)
                with cuda.stream(s3):
                    w = self.model3(z)
            else:
                if self.sep:
                    e1, e2, e3, e4 = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True),\
                                     cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
                    cuda.synchronize()
                    e1.record()
                    u = self.model1(x)
                    e2.record()
                    v = self.model2(y)
                    e3.record()
                    w = self.model3(z)
                    e4.record()
                    cuda.synchronize()
                    print(e1.elapsed_time(e2), e2.elapsed_time(e3), e3.elapsed_time(e4))
                else:
                    u = self.model1(x)
                    v = self.model2(y)
                    w = self.model3(z)
        else:
            if self.sep:
                e1, e2, e3, e4 = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True), \
                                 cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
                cuda.synchronize()
                e1.record()
                u = self.model1(x).to(self.device2)
                e2.record()
                v = self.model2(u).to(self.device3)
                e3.record()
                w = self.model3(v)
                e4.record()
                cuda.synchronize()
                print(e1.elapsed_time(e2), e2.elapsed_time(e3), e3.elapsed_time(e4))
            else:
                u = self.model1(x).to(self.device2)
                v = self.model2(u).to(self.device3)
                w = self.model3(v)


num_batches = 3
batch_size = 120
image_w = 128
image_h = 128


def train(model, inputs, mp_):
    print(time.time())
    for batch_id in range(num_batches):
        _ = model(*inputs[batch_id], mp_=mp_)


def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)


if __name__ == "__main__":
    check_availability()
    cpu, gpu1, gpu2, gpu3 = torch.device("cpu"), torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:2")
    num_repeat = 30

    inputs111 = [[torch.randn(batch_size, 16, image_w, image_h).to(gpu1),
                  torch.randn(batch_size, 16, image_w, image_h).to(gpu1),
                  torch.randn(batch_size, 16, image_w, image_h).to(gpu1)] for _ in range(3)]

    inputs123 = [[torch.randn(batch_size, 16, image_w, image_h).to(gpu1),
                  torch.randn(batch_size, 16, image_w, image_h).to(gpu2),
                  torch.randn(batch_size, 16, image_w, image_h).to(gpu3)] for _ in range(3)]
    # warm up
    print("warmup")
    stmt = "train(model, inputs123, True)"
    setup = "model=ToyModel(gpu1, gpu2, gpu3, sequence=False)"
    _ = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())

    # experiments
    print("psg")
    stmt = "train(model, inputs111, False)"
    setup = "model=ToyModel(gpu1, gpu1, gpu1, sequence=False)"
    psg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    psg_mean, psg_std = np.mean(psg_run_times), np.std(psg_run_times)

    print("real_psg")
    stmt = "train(model, inputs111, False)"
    setup = "model=ToyModel(gpu1, gpu1, gpu1, sequence=False, sep=True)"
    r_psg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    r_psg_mean, r_psg_std = np.mean(r_psg_run_times), np.std(r_psg_run_times)

    # print("stg")
    # stmt = "train(model, inputs123, False)"
    # setup = "model=ToyModel(gpu1, gpu2, gpu3, sequence=True)"
    # stg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    # stg_mean, stg_std = np.mean(stg_run_times), np.std(stg_run_times)

    print("ssg")
    stmt = "train(model, inputs111, False)"
    setup = "model=ToyModel(gpu1, gpu1, gpu1, sequence=True)"
    ssg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    ssg_mean, ssg_std = np.mean(ssg_run_times), np.std(ssg_run_times)

    print("real_ssg")
    stmt = "train(model, inputs111, False)"
    setup = "model=ToyModel(gpu1, gpu1, gpu1, sequence=True, sep=True)"
    r_ssg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    r_ssg_mean, r_ssg_std = np.mean(r_ssg_run_times), np.std(r_ssg_run_times)

    print("ptg")
    stmt = "train(model, inputs123, False)"
    setup = "model=ToyModel(gpu1, gpu2, gpu3, sequence=False)"
    ptg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    ptg_mean, ptg_std = np.mean(ptg_run_times), np.std(ptg_run_times)

    print("real_ptg")
    stmt = "train(model, inputs123, False)"
    setup = "model=ToyModel(gpu1, gpu2, gpu3, sequence=False, sep=True)"
    r_ptg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    r_ptg_mean, r_ptg_std = np.mean(r_ptg_run_times), np.std(r_ptg_run_times)

    print("mp_ptg")
    stmt = "train(model, inputs123, True)"
    setup = "model=ToyModel(gpu1, gpu2, gpu3, sequence=False)"
    mp_ptg_run_times = timeit.repeat(stmt, setup, number=1, repeat=num_repeat, globals=globals())
    mp_ptg_mean, mp_ptg_std = np.mean(mp_ptg_run_times), np.std(mp_ptg_run_times)

    plot([psg_mean, r_psg_mean, ptg_mean, r_ptg_mean, ssg_mean, r_ssg_mean, mp_ptg_mean],
         [psg_std, r_psg_std, ptg_std, r_ptg_std, ssg_std, r_ssg_std, mp_ptg_std],
         ['par_single_gpus', 'real_par_single_gpus', 'par_triple_gpus', 'real_par_triple_gpus',
          'seq_single_gpus', 'real_seq_single_gpus', "mp_par_triple_gpus"],
         'pipeline_test.png')