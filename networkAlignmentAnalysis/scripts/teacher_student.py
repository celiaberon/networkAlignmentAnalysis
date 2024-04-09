from tqdm import tqdm
import numpy as np
import torch

from matplotlib import pyplot as plt

import os
import sys

mainPath = os.path.dirname(os.path.abspath(__file__)) + "/.."
sys.path.append(mainPath)

from networkAlignmentAnalysis.models.registry import get_model
from networkAlignmentAnalysis import utils
from networkAlignmentAnalysis import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_data(modes, signal_dist, noise_amplitude, batch_size):
    signal = signal_dist.sample((batch_size,))
    data = signal @ modes + torch.normal(0, noise_amplitude, (batch_size, modes.size(1)))
    return signal, data


def train(teacher, student, modes, signal_dist, noise_amplitude, batch_size=256, num_epochs=1000):
    optimizer = torch.optim.SGD(student.parameters(), lr=1e-2)
    lossfn = torch.nn.MSELoss()
    track_loss = []
    alignment = []
    for epoch in tqdm(range(num_epochs)):
        signal, data = get_data(modes, signal_dist, noise_amplitude, batch_size)
        signal, data = signal.to(DEVICE), data.to(DEVICE)

        optimizer.zero_grad()
        target = teacher(signal)
        output = student(data, store_hidden=True)
        loss = lossfn(output, target)
        loss.backward()
        optimizer.step()
        track_loss.append(loss.item())
        alignment.append(student.measure_alignment(data, precomputed=True))

    alignment = [torch.stack(a) for a in utils.transpose_list(alignment)]  # list across layers, (epochs x dim)
    return student, track_loss, alignment


if __name__ == "__main__":
    print("using device: ", DEVICE)

    N = 100
    num_modes = 50
    shape = 0.7 * torch.ones((num_modes,))
    modes = torch.normal(0, 1, (num_modes, N))
    amplitude = torch.rand((num_modes,)) * 0.9 + 0.1
    signal_dist = torch.distributions.gamma.Gamma(shape, amplitude)
    noise_amplitude = 0.2

    model_name = "MLP"
    teacher = get_model(model_name, build=True, input_dim=num_modes, dropout=0.0, ignore_flag=False, linear=True).to(DEVICE).eval()
    student = get_model(model_name, build=True, input_dim=N, dropout=0.0, ignore_flag=False).to(DEVICE)

    student, track_loss, alignment = train(teacher, student, modes, signal_dist, noise_amplitude)
    mean_alignment = torch.stack([torch.mean(align, dim=1) for align in alignment])

    fig, ax = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
    ax[0].plot(track_loss)
    ax[0].set_ylim(0, 0.5)
    for layer in range(len(alignment)):
        ax[1].plot(mean_alignment[layer], label=f"layer{layer}")
    ax[1].legend()

    signal, data = get_data(modes, signal_dist, noise_amplitude, 1)
    ax[2].scatter(teacher(signal).detach(), student(data).detach())
    plt.show()

    N, S = 2, 10000
    num_modes = 2
    nonnormality = 1.5
    a = b = 1 / nonnormality
    modes = np.random.normal(0, 1, (num_modes, N)) * np.random.normal(0, 1, num_modes).reshape(-1, 1)
    signal = np.random.gamma(0.5, 1, (S, num_modes)) * np.sign(np.random.random((S, num_modes)) - 0.5)
    data = signal @ modes
    w, v = [np.array(o) for o in utils.smart_pca(torch.tensor(data).T)]
    print(w.shape, v.shape)

    plt.scatter(data[:, 0], data[:, 1], s=1, c=("b", 0.4))
    for mode in modes:
        plt.plot([0, mode[0]], [0, mode[1]], c="r")
    for evec in v.T:
        plt.plot([0, evec[0]], [0, evec[1]], c="k")

    plt.show()
