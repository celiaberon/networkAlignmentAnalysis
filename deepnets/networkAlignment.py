
import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torchvision import transforms
from tqdm import tqdm

sys.path.append("../")
sys.path.append('/n/home00/cberon/code/networkAlignmentAnalysis/')
from deepnets import nnModels as models
from deepnets import nnUtilities as nnutils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

dataPath = None


def initialize_model(model_name='MLP4', actFunc=F.relu, pDropout=0):

    if model_name == 'CNN2P2':
        convActivation = actFunc
        linearActivation = actFunc
        net = models.CNN2P2(convActivation=convActivation,
                            linearActivation=linearActivation)
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
        ])
    elif model_name.startswith('MLP'):
        if model_name == 'MLP3':
            net = models.MLP3(actFunc=actFunc)
        elif model_name == 'MLP4':
            net = models.MLP4(actFunc=actFunc, pDropout=pDropout)
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
            transforms.Lambda(torch.flatten),  # convert to vectors
        ])
    elif model_name == 'AlexNet':
        # Initialize AlexNet with no pretrained weights.
        net = models.AlexNet(weights=None, pDropout=pDropout)
        preprocess = None
    else:
        raise ValueError('useNet not recognized')

    return net, preprocess


def prepare_loaders(dataset, **kwargs):

    if dataset == 'MNIST':
        train_loader, test_loader, n_classes = nnutils.downloadMNIST(**kwargs)
    elif dataset == 'ImageNet-Tiny':
        train_loader, test_loader, n_classes = nnutils.downloadImageNetTiny(**kwargs)
    elif dataset == 'ImageNet':
        train_loader, test_loader, n_classes = nnutils.downloadImageNet(**kwargs)

    return train_loader, test_loader, n_classes


@torch.no_grad()
def eval_test_performance(test_loader, trained_net, loss_function,
                          DEVICE=None):

    '''Measure performance on test set'''

    if DEVICE is None:
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    total_loss = 0
    n_correct = 0
    n_attempted = 0
    for images, label in test_loader:
        images, label = images.to(DEVICE), label.to(DEVICE)
        outputs = trained_net(images)
        total_loss += loss_function(outputs, label).item()
        predictions = torch.argmax(outputs, axis=1)
        n_correct += sum(predictions == label)
        n_attempted += images.shape[0]

    print(f'Average loss over test set: {(total_loss / len(test_loader)):.2f}.')
    print(f'Accuracy over test set: {(100 * n_correct / n_attempted):.2f}%.')


def load_config(config_path):
    with open(config_path, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def run_training_loop(config_path):

    config = load_config(config_path)

    useNet = config['dataloader_params']['useNet']
    dataset = config['dataloader_params']['dataset']
    batch_size = config['dataloader_params']['batch_size']
    pDropout = config['training_params']['pDropout']
    learningRate = config['training_params']['learningRate']
    iterations = config['training_params']['iterations']
    verbose = config['training_params']['verbose']
    root = config['saving_params']['root']
    run_id = config['saving_params']['run_id']
    save_path = os.path.join(root, 'runs', run_id)

    print(f'Run {run_id}: Training {useNet} on {dataset}')

    if isinstance(learningRate, str):
        learningRate = eval(learningRate)

    # Prepare network and preprocessing transforms for data.
    net, preprocess = initialize_model(model_name=useNet,
                                       actFunc=F.relu,
                                       pDropout=pDropout)
    net.to(DEVICE)
    if verbose:
        print(f'Initialized {useNet} on {DEVICE}')

    t = time.time()
    # Prepare Dataloaders
    train_loader, test_loader, _ = prepare_loaders(dataset=dataset,
                                                   batchSize=batch_size,
                                                   preprocess=preprocess,
                                                   n_workers=2)

    # Prepare Training Functions
    loss_function = nn.CrossEntropyLoss()  # Note: automatically applies softmax
    optimizer = torch.optim.SGD(net.parameters(), lr=learningRate)

    # Preallocate summary variables
    n_batches = len(train_loader)
    n_train_steps = n_batches * iterations
    track_loss = torch.zeros(n_train_steps)
    track_accuracy = torch.zeros(n_train_steps)
    alignFull = []
    alignFull_nodo = []
    deltaWeights = []

    if verbose:
        print(f'DataLoaders prepared in {(time.time() - t):.3f} seconds.')
        print(f'DataLoaders ready with {n_batches=} for {n_train_steps=}')

    # Store initial network weights to view how things change over training
    initWeights = net.getNetworkWeights()

    # Train Network & Measure Integration
    t = time.time()
    for epoch in range(iterations):

        print(f'Starting {epoch=}')
        for idx, (images, labels) in enumerate(train_loader):
            cidx = epoch * n_batches + idx  # stores idx of each "miniepoch"

            # move batch to GPU (if available)
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = net(images)

            # Perform backward pass & optimization
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # Measure alignment of weights using current images in batch as
            # sample. Store measurements for both (with) and without dropout.
            alignFull.append(net.measureAlignment(images, usePrevious=True))
            with torch.no_grad():
                net.setDropout(0)
                alignFull_nodo.append(net.measureAlignment(images, usePrevious=False))
                net.setDropout(pDropout)

            # Measure Change in Weights
            deltaWeights.append(net.compareNetworkWeights(initWeights))

            # Measure alignment of change in weights

            # Track Loss and Accuracy
            track_loss[cidx] = loss.item()
            track_accuracy[cidx] = (100 * torch.sum(torch.argmax(outputs, axis=1) == labels)
                                    / images.shape[0])

        if verbose:   # print statistics for each epoch
            print(f'Loss in epoch {epoch:3d}: {loss.item():.3f}, Accuracy: '
                  f'{track_accuracy[cidx]:.2f}%')
            print(f'Epoch ran in {(time.time() - t):.3f} seconds.')

    # Measure performance on test set
    eval_test_performance(test_loader, trained_net=net,
                          loss_function=loss_function, DEVICE=DEVICE)

    print(f'Training process has finished in {(time.time() - t):.3f} seconds.')

    results = {
        'net': net,
        'initWeights': initWeights,
        'alignFull': alignFull,
        'alignFull_nodo': alignFull_nodo,
        'deltaWeights': deltaWeights,
        'trackLoss': track_loss,
        'trackAccuracy': track_accuracy,
        'trainloader': train_loader,
        'testloader': test_loader,
        'learningRate': learningRate,
    }

    with open(os.path.join(save_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    return results


def plot_training_alignment(config_path, results=None):

    config = load_config(config_path)

    verbose = config['training_params']['verbose']
    root = config['saving_params']['root']
    run_id = config['saving_params']['run_id']
    ub = config['plotting_params']['ub']
    lb = config['plotting_params']['lb']

    save_path = os.path.join(root, 'runs', run_id)
    os.makedirs(save_path) if not os.path.isdir(save_path) else None
    if results is None:
        results = run_training_loop(config_path)

    net = results.get('net')
    alignFull = results.get('alignFull')
    alignFull_nodo = results.get('alignFull_nodo')
    trainloader = results.get('trainloader')
    totalEpochs = config['training_params']['iterations'] * len(trainloader)

    # Store config params alongside saved data.
    with open(os.path.join(save_path, 'config.yaml'), 'w') as yaml_file:
        yaml.dump(config, yaml_file, default_flow_style=False)

    # returns average alignment for each layer for each training step
    # (aka mini-epoch = n_batches * iterations).
    alignMean = net.avg_within_layer(alignFull)
    alignMean_nodo = net.avg_within_layer(alignFull_nodo)

    # [(numNodes, numMiniEpochs) for layer in numLayers] for each layer,
    # return the alignment of each network node for each mini-epoch.
    alignLayer = [net.layerFromFull(alignFull, layer)
                  for layer in range(net.numLayers)]
    alignLayer_nodo = [net.layerFromFull(alignFull_nodo, layer)
                       for layer in range(net.numLayers)]

    fig, axs = plt.subplots(1, net.numLayers, figsize=(16, 4))
    for ax, layer in zip(axs, range(net.numLayers)):
    
        # get upper and lower quantile of alignment for each layer
        uq = torch.quantile(alignLayer[layer], q=ub, dim=0)
        lq = torch.quantile(alignLayer[layer], q=lb, dim=0)

        uqnd = torch.quantile(alignLayer_nodo[layer], q=ub, dim=0)
        lqnd = torch.quantile(alignLayer_nodo[layer], q=lb, dim=0)

        # plot average alignment for each layer (scale to % of variance)
        ax.plot(range(totalEpochs), 100 * alignMean[layer],
                color='k', linewidth=1.5, label='alignment')
        ax.plot(range(totalEpochs), 100 * alignMean_nodo[layer],
                color='b', linewidth=1.5, label='alignment_nodo')

        # plot quantile range of alignment for each layer (scale to % of variance)
        ax.fill_between(range(totalEpochs), 100 * uq, 100 * lq, color='k',
                        alpha=0.4)
        ax.fill_between(range(totalEpochs), 100 * uqnd, 100 * lqnd, color='b',
                        alpha=0.4)

        ax.plot(range(totalEpochs), results.get('trackLoss'), color='r',
                linewidth=1.5, label='trainingLoss')

        ax.set_ylim(0)
        ax.set_xlabel('Training Epoch')
        ax.set_ylabel('Alignment (%VarExpl)')
        ax.set_title(f'Average Alignment Layer {layer}')
        if layer == net.numLayers - 1:
            ax.legend(loc='lower right')
        else:
            ax.legend().remove()
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'training_alignment.png'), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    ax.plot([0, 1], [0, 1], ls='--', lw=1, color='k')
    for i in range(net.numLayers):
        plt.scatter(alignMean[i], alignMean_nodo[i], label=f"layer {i}", s=8,
                    alpha=0.3)
        plt.scatter(alignMean[i][0], alignMean_nodo[i][0], c='k')
    plt.legend()
    ax.set(xlim=(0, max(alignMean.max(), alignMean_nodo.max() + 0.03)),
           ylim=(0, max(alignMean.max(), alignMean_nodo.max() + 0.03)),
           xlabel='alignment (dropout)', ylabel='alignment (no dropout)',
           title='Layer-wise alignment\nwith/without dropout')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'dropout_alignment.png'),
                bbox_inches='tight')

    # Measure Activations (without dropout) for all images
    storeDropout = net.getDropout()
    net.setDropout(0)  # no dropout for measuring eigenfeatures

    allimages = []
    activations = []
    for images, label in tqdm(trainloader):    
        allimages.append(images)
        images, label = images.to(DEVICE), label.to(DEVICE)
        # this is a huge list of the activations across all layers for each
        # image in the data.
        activations.append(net.getActivations(images))
    net.setDropout(storeDropout)

    if verbose:
        print('n batches = ', len(activations))  # num batches in trainLoader
        print('n layers = ', len(activations[0]))  # num layers in network
        print('batchSize x nodesPerLayer) for each layer\n',
              [a.shape for a in activations[0]])

    # Make list containing the input to each layer for every image. The first
    # element (allinputs[0]) is just the images. Every next element is the
    # post-activations of each successive layer in the network.
    allinputs = []
    allinputs.append(torch.cat(allimages, dim=0).detach().cpu())
    for layer in range(net.numLayers - 1):
        allinputs.append(torch.cat([cact[layer] for cact in activations], dim=0).detach().cpu())
    if verbose:
        for ai in allinputs:
            print(ai.shape)

    # Measure eigenfeatures for each layer so we can measure the eigenvalues
    # of the input to each layer.
    eigenvalues = []
    eigenvectors = []
    for ai in allinputs:
        w, v = sp.linalg.eigh(torch.cov(ai.T))
        widx = np.argsort(w)[::-1]
        w = w[widx]
        v = v[:, widx]
        eigenvalues.append(w)
        eigenvectors.append(v)
    if verbose:
        for evl, evc in zip(eigenvalues, eigenvectors):
            print(f'EVal Shape: {evl.shape}, EVec Shape: {evc.shape}')


    # Measure dot product of weights for each layer
    # The dot product of the weight for each layer with the eigenvectors of the input to each layer
    # tells us what fraction of the weight for each node comes from each eigenvector!
    beta = []
    netweights = net.getNetworkWeights()
    for evc, nw in zip(eigenvectors, netweights):
        nw = nw / torch.norm(nw, dim=1, keepdim=True)
        beta.append(torch.abs(nw.cpu() @ evc))
    if verbose:
        for b in beta:
            print(b.shape)

    # Compare the eigenvalues of the input to each layer with the betas of the weight vectors for each eigenvector
    # The black line is the eigenvalues. It tells us the variance structure of the input to each layer --
    # -- specifically, it tells us which dimensions of the input contain significant variance
    # The blue lines are the average +/- std dot product ("beta") between the weights of each layer and the eigenvectors of the input to each layer --

    fig, axs = plt.subplots(1, net.numLayers, figsize=(10, 3))
    for ax, layer in zip(axs, range(net.numLayers)):
        cNEV = len(eigenvalues[layer])
        mnbeta = torch.mean(beta[layer], dim=0)
        sebeta = torch.std(beta[layer], dim=0)
        ax.fill_between(range(cNEV), mnbeta + sebeta, mnbeta - sebeta, color='b', alpha=0.2)
        ax.plot(range(cNEV), mnbeta, c='b', label='AverageProjection')
        ax.plot(range(cNEV), eigenvalues[layer] / np.sum(eigenvalues[layer]), c='k', label='Eigenvalues')
    plt.tight_layout()
    fig.savefig(os.path.join(save_path, 'eigenvalues_vs_weights.png'),
                bbox_inches='tight')

config_path = sys.argv[1]
# config_path = 'config.yaml'
plot_training_alignment(config_path)
