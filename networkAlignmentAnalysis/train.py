import time
from copy import copy, deepcopy
from pathlib import Path

import torch
import torch.distributed as dist
from tqdm import tqdm

from networkAlignmentAnalysis.utils import (condense_values, gather_by_layer,
                                            get_alignment_dims, get_list_dims,
                                            permute_distributed_metric,
                                            save_checkpoint, test_nets,
                                            train_nets, transpose_list,
			     		    smart_pca,
   					    expected_alignment_distribution)

@train_nets
def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # input argument checks
    if not (isinstance(nets, list)):
        nets = [nets]
    if not (isinstance(optimizers, list)):
        optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    # check if we should print progress bars
    verbose = parameters.get("verbose", True)

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    use_train = parameters.get("train_set", True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader) * parameters["num_epochs"]

    # --- optional W&B logging ---
    run = parameters.get("run")

    # --- optional analyses ---
    measure_alignment = parameters.get("alignment", True)
    measure_delta_weights = parameters.get("delta_weights", False)
    measure_delta_alignment = parameters.get("delta_alignment", False)
    measure_frequency = parameters.get("frequency", 1)
    compare_expected = parameters.get("compare_expected", False)

    # --- optional training method: manual shaping with eigenvectors ---
    manual_shape = parameters.get("manual_shape", False)  # true or False, whether to do this
    # frequency of manual shape (similar to measure_frequency)
    # if positive, by minibatch, if negative, by epoch
    manual_frequency = parameters.get("manual_frequency", -1)
    manual_transforms = parameters.get("manual_transforms", None)  # len()==len(nets) callable methods
    manual_layers = parameters.get("manual_layers", None)  # index to which layers

    # --- create results dictionary if not provided and handle checkpoint info ---
    results = parameters.get("results", False)
    num_complete = parameters.get("num_complete", 0)
    save_ckpt = parameters.get("save_ckpt", False)
    freq_ckpt = parameters.get("freq_ckpt", 1)
    path_ckpt, unique_ckpt = parameters.get("path_ckpt", ("", True))
    
    # Store metrics on GPU if using DDP, otherwise store on cpu.
    internal_device = dataset.device if dataset.distributed else 'cpu'
    
    if not results:
        # initialize dictionary for storing performance across epochs
        results = {
            "loss": torch.zeros((num_steps, num_nets), device=internal_device),
            "accuracy": torch.zeros((num_steps, num_nets), device=internal_device),
        }

        # measure alignment throughout training
        if measure_alignment:
            results["alignment"] = []
            if dataset.distributed:
                alignment_dims = get_alignment_dims(nets, dataset,
                                    num_epochs=1,  # can set to num epochs if don't want to agg every epoch
                                    use_train=use_train)
                full_alignment = [[torch.zeros(layer_dims, dtype=torch.float, device=dataset.device)
                                  for _ in range(dist.get_world_size())]
                                  for layer_dims in alignment_dims]
                alignment_reference = [[torch.clone(proc) for proc in layer_dims] for layer_dims in full_alignment]
                print(f'full alignment dims should be {dist.get_world_size()} x {alignment_dims}')

        # measure weight norm throughout training
        if measure_delta_weights:
            results["delta_weights"] = []
            results["init_weights"] = [net.module.get_alignment_weights() for net in nets]

        # measure alignment of weight updates throughout training
        if measure_delta_alignment:
            if not "init_weights" in results:
                results["init_weights"] = [net.get_alignment_weights() for net in nets]
            results["delta_alignment"] = []

        # compare true alignment distribution to expected distribution (according to Fiete alignment definition)
        if compare_expected:
            calign_bins = torch.linspace(0, 1, 301)
            results["compare_alignment_bins"] = calign_bins
            results["compare_alignment_expected"] = []
            results["compare_alignment_observed"] = []
            if measure_delta_alignment:
                results["compare_delta_alignment_observed"] = []

    # If loaded from checkpoint but running more epochs than initialized for.
    elif results["loss"].shape[0] < num_steps:
        add_steps = num_steps - results["loss"].shape[0]
        assert (add_steps / (parameters["num_epochs"] - num_complete)) == len(
            dataset.train_loader
        ), "Number of new steps needs to multiple of epochs and num minibatches"
        results["loss"] = torch.vstack((results["loss"], torch.zeros((add_steps, num_nets))))
        results["accuracy"] = torch.vstack((results["accuracy"], torch.zeros((add_steps, num_nets))))

    if num_complete > 0:
        print("resuming training from checkpoint on epoch", num_complete)

    # --- training loop ---
    epoch_loop = range(num_complete, parameters["num_epochs"])
    if verbose:
        epoch_loop = tqdm(epoch_loop, desc="training epoch")

    for epoch in epoch_loop:

	if dataset.distributed:
            if use_train:
                dataset.train_sampler.set_epoch(epoch)
            else:
                dataset.test_sampler.set_epoch(epoch)

            # Reset local and group metrics every epoch.
            local_alignment=[]
            full_alignment = [[torch.clone(proc) for proc in layer_dims] for layer_dims in alignment_reference]

        # Create batch loop with optional progress updates
        batch_loop = dataloader
        if verbose:
            batch_loop = tqdm(batch_loop, desc="minibatch", leave=False)

        for idx, batch in enumerate(batch_loop):

            cidx = epoch * len(dataloader) + idx
            images, labels = dataset.unwrap_batch(batch, device=dataset.device)

            # Zero the gradients
            for opt in optimizers:
                opt.zero_grad()

            # Perform forward pass
            outputs = [net(images, store_hidden=True) for net in nets]

            # Perform backward pass & optimization
            loss = [dataset.measure_loss(output, labels) for output in outputs]
            for l, opt in zip(loss, optimizers):
                l.backward()
                opt.step()

            results["loss"][cidx] = torch.tensor([l.item() for l in loss], device=internal_device)
            results["accuracy"][cidx] = torch.tensor(
                [dataset.measure_accuracy(output, labels) for output in outputs], device=internal_device
            )

            # Put accuracy reduce here. Assume loss done automatically?
            if dataset.distributed:
                # print('loss', dist.get_rank(), results["loss"][cidx])
                # Seems that loss isn't automatically all reduced as expected?
                if dataset.loss_function.reduction == 'mean':
                    dist.all_reduce(results["loss"][cidx], op=dist.ReduceOp.AVG)
                    dist.all_reduce(results["loss"][cidx], op=dist.ReduceOp.AVG)
                    # print('loss', dist.get_rank(), results["loss"][cidx])
                else:
                    raise NotImplementedError
                
            if idx % measure_frequency == 0:
                if measure_alignment:
                    # Measure alignment if requested
	            alignment = [net.module.measure_alignment(images, precomputed=True, method="alignment")
                            for net in nets]

                    if not dataset.distributed:
                        results["alignment"].append(alignment)
                    else:
                        local_alignment.append(alignment)

                if measure_delta_weights or measure_delta_alignment:
                    c_delta_weights = [net.compare_weights(init_weight) for net, init_weight in zip(nets, results["init_weights"])]
                    if measure_delta_weights:
                        # Save change in weights if requested
                        results["delta_weights"].append(c_delta_weights)
                    if measure_delta_alignment:
                        # Save delta weight alignment if requested
                        c_delta_alignment = [
                            net.measure_alignment_weights(images, weights, precomputed=True, method="alignment")
                            for net, weights in zip(nets, c_delta_weights)
                        ]
                        results["delta_alignment"].append(c_delta_alignment)

                if compare_expected:
                    # Measure distribution of alignment, compare with expected given "Alignment" from Fiete definition
                    if measure_alignment:
                        c_alignment = results["alignment"][-1]
                    else:
                        c_alignment = [net.measure_alignment(images, precomputed=True, method="alignment") for net in nets]
                    c_inputs = [net.get_layer_inputs(images, precomputed=True) for net in nets]
                    c_inputs = [net._preprocess_inputs(cin) for net, cin in zip(nets, c_inputs)]
                    c_evals = [[smart_pca(c.T)[0] for c in cin] for cin in c_inputs]
                    c_dist = [[expected_alignment_distribution(ev, valid_rotation=False, bins=calign_bins)[0] for ev in c_eval] for c_eval in c_evals]
                    t_dist = [[torch.histogram(align.cpu(), bins=calign_bins, density=True)[0] for align in c_align] for c_align in c_alignment]
                    results["compare_alignment_expected"].append(c_dist)
                    results["compare_alignment_observed"].append(t_dist)
                    if measure_delta_alignment:
                        d_alignment = results["delta_alignment"][-1]
                        d_dist = [[torch.histogram(dalign.cpu(), bins=calign_bins, density=True)[0] for dalign in d_align] for d_align in d_alignment]
                        results["compare_delta_alignment_observed"].append(d_dist)

            # Log run with WandB, but only from main process if using distributed training.
            if (run is not None) and ((not dataset.distributed) or dist.get_rank() == 0):
                run.log(
                    {f"losses/loss-{ii}": l.item() for ii, l in enumerate(loss)}
                    | {f"accuracies/accuracy-{ii}": dataset.measure_accuracy(output, labels) for ii, output in enumerate(outputs)}
                    | {"batch": cidx}
                )

        if manual_shape:
            # only do it at the end of #=manual_frequency epochs (but not last)
            if ((epoch + 1) % manual_frequency == 0) and (epoch < parameters["num_epochs"] - 1):
                for net, transform in tqdm(zip(nets, manual_transforms), desc="manual shaping", leave=False):
                    # just use this minibatch for computing eigenfeatures
                    inputs, _ = net._process_collect_activity(dataset, train_set=False, with_updates=False, use_training_mode=False)
                    _, eigenvalues, eigenvectors = net.measure_eigenfeatures(inputs, with_updates=False)
                    idx_to_layer_lookup = {layer: idx for idx, layer in enumerate(net.get_alignment_layer_indices())}
                    eigenvalues = [eigenvalues[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    eigenvectors = [eigenvectors[idx_to_layer_lookup[ml]] for ml in manual_layers]
                    net.shape_eigenfeatures(manual_layers, eigenvalues, eigenvectors, transform)

        if dataset.distributed:
            # Transpose first dim from steps to layers.
            local_alignment = condense_values(transpose_list(local_alignment))
            local_alignment = [layer.to(dataset.device) for layer in local_alignment]
            gather_by_layer(local_alignment, full_alignment)
            full_alignment = [permute_distributed_metric(torch.cat(layer, dim=1).cpu()) for layer in full_alignment]
            results['alignment'].append(full_alignment)
            dist.barrier()

        if save_ckpt and ((epoch % freq_ckpt == 0) or (epoch == (parameters["num_epochs"] - 1))):
            if (not dataset.distributed) or (dist.get_rank() == 0):
                if unique_ckpt:
                    prefix, suffix = str(path_ckpt).split('.')
                    prefix = prefix.split('checkpoint')[0]  # for subsequent epoch checkpoints
                    path_ckpt = Path(f'{prefix}checkpoint_{epoch}.{suffix}')
                save_checkpoint(
                    nets,
                    optimizers,
                    results | {"prms": parameters, "epoch": epoch, "device": dataset.device},
                    path_ckpt,
                )
            if dataset.distributed:
                dist.barrier()

    # condense optional analyses
    for k in [
        "alignment",
        "delta_weights",
        "delta_alignment",
        "avgcorr",
        "fullcorr",
        "compare_alignment_expected",
        "compare_alignment_observed",
        "compare_delta_alignment_observed",
    ]:
        if k not in results.keys():
            continue
        if (k == 'alignment') and dataset.distributed:
            continue
        results[k] = condense_values(transpose_list(results[k]))

    if measure_alignment and dataset.distributed:
        results["loss"] = results["loss"].cpu()
        results["accuracy"] = results["accuracy"].cpu()
        
        # Concatenate along step axis for each layer?
        results['alignment'] = [torch.cat(ilayer, axis=1) for ilayer in zip(*results['alignment'])]
        print('post', get_list_dims(results["alignment"]))
        dist.barrier()

    return results


@torch.no_grad()
@test_nets
def test(nets, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    # --- optional W&B logging ---
    run = parameters.get("run")

    # input argument checks
    if not (isinstance(nets, list)):
        nets = [nets]

    # preallocate variables and define metaparameters
    verbose = parameters.get("verbose", True)
    num_nets = len(nets)

    # retrieve requested dataloader from dataset
    use_test = not parameters.get("train_set", False)  # if train_set=True, use_test=False
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # Performance Measurements
    total_loss = [0 for _ in range(num_nets)]
    num_correct = [0 for _ in range(num_nets)]
    num_batches = 0

    # --- optional analyses ---
    measure_alignment = parameters.get("alignment", True)

    # measure alignment throughout training
    if measure_alignment:
        alignment = []
        if dataset.distributed:
            alignment_dims = get_alignment_dims(nets, dataset, 1, use_train=False)
            full_alignment = [[torch.zeros(layer_dims, dtype=torch.float, device=dataset.device)
                                for _ in range(dist.get_world_size())]
                                for layer_dims in alignment_dims]
            print(f'full alignment dims should be {len(full_alignment)} x {alignment_dims}')

    batch_loop = tqdm(dataloader) if verbose else dataloader
    for batch in batch_loop:
        images, labels = dataset.unwrap_batch(batch)

        # Perform forward pass
        outputs = [net(images, store_hidden=True) for net in nets]

        # Performance Measurements
        for net_idx, output in enumerate(outputs):
            total_loss[net_idx] += dataset.measure_loss(output, labels).item()
            num_correct[net_idx] += dataset.measure_accuracy(output, labels)

        # Keep track of number of batches
        num_batches += 1

        # Measure Alignment
        if measure_alignment:
            alignment.append([net.module.measure_alignment(images, precomputed=True, method="alignment")
                    for net in nets])

    if dataset.distributed:
        # Seems that loss isn't automatically all reduced as expected?
        total_loss = torch.tensor(total_loss, device=dataset.device)
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        num_correct = torch.tensor(num_correct, device=dataset.device)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        num_batches = torch.tensor(num_batches, device=dataset.device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        num_batches = num_batches.cpu()
        dist.barrier()

    results = {
        "loss": [loss / num_batches for loss in total_loss],
        "accuracy": [correct / num_batches for correct in num_correct],
    }

    if measure_alignment:
        results["alignment"] = condense_values(transpose_list(alignment))

    if measure_alignment and dataset.distributed:
        alignment_local = [layer.to(dataset.device) for layer in results['alignment']]
        gather_by_layer(alignment_local, full_alignment)
        # Overwrite local alignment for main process with aggregated. Stack onto dimension for test batches.
        # Order shouldn't matter for inference except for traceback to eigenfeatures?
        results['alignment'] = [torch.cat(layer, dim=1).cpu() for layer in full_alignment]

    return results


@torch.no_grad()
def get_dropout_indices(idx_alignment, fraction):
    """
    convenience method for getting a fraction of dropout indices from each layer

    idx_alignment should be a list of the indices of alignment (sorted from lowest to highest)
    where len(idx_alignment)=num_layers_per_network and each element is a tensor such that
    idx_alignment[0].shape=(num_nodes_per_layer, num_networks)

    returns a fraction of indices to drop of highest, lowest, and random alignment
    """
    num_nets = idx_alignment[0].size(0)
    num_nodes = [idx.size(1) for idx in idx_alignment]
    num_drop = [int(nodes * fraction) for nodes in num_nodes]
    idx_high = [idx[:, -drop:] for idx, drop in zip(idx_alignment, num_drop)]
    idx_low = [idx[:, :drop] for idx, drop in zip(idx_alignment, num_drop)]
    idx_rand = [torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0) for nodes, drop in zip(num_nodes, num_drop)]
    return idx_high, idx_low, idx_rand


@torch.no_grad()
@test_nets
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout

    takes as input a list of networks (usually trained) and a dataset, along with some
    experiment parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    alignment, when provided is a list of the alignment measurement for each layer of
    each network. It is expected that each alignment list has the structure:
    len(alignment)=num_alignment_layers
    alignment[i].shape = (num_networks, num_batches, num_dimensions_per_layer)
    : see how the outputs of ``measure_alignment`` is handled in the ``train`` and ``test``
    functions of this module to understand how to structure it in that format.

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more nodes based on the highest, lowest,
    or random alignment. Can either do it for each layer separately or all togehter using
    the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get index to each alignment layer
    idx_dropout_layers = nets[0].module.get_alignment_layer_indices()

    # get alignment of networks if not provided
    if alignment is None:
        alignment = test(nets, dataset, **parameters)["alignment"]

    # If using distributed processing, take only alignment corresponding to subset of data in sampler.
    if dataset.distributed:
        rank = dist.get_rank()  # will be index of gather eigenvalues/vectors
        alignment = [layer[:, ::dist.get_world_size(), :] for layer in alignment]

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert len(alignment) == len(idx_dropout_layers), "the number of layers in **alignment** doesn't correspond to the number of alignment layers"

    # don't dropout classification layer if included as an alignment layer
    classification_layer = nets[0].module.num_layers(all=True) - 1  # index to last layer in network
    if classification_layer in idx_dropout_layers:
        idx_dropout_layers.pop(-1)
        alignment.pop(-1)

    # get average alignment (across batches) and index of average alignment by node
    alignment = [torch.mean(align, dim=1) for align in alignment]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

    # only need to store scores on GPU if using distributed processing.
    internal_device = dataset.device if dataset.distributed else 'cpu'

    # preallocate tracker tensors
    scores = {'high': {},
              'low': {},
              'rand': {}
             }
    for key in scores:
        scores[key]['progdrop_loss'] = torch.zeros((num_nets, num_drops, num_layers), device=internal_device)
        scores[key]['progdrop_acc'] = torch.zeros((num_nets, num_drops, num_layers), device=internal_device)

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_train = parameters.get("train_set", False)
    dataloader = dataset.train_loader if use_train else dataset.test_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_alignment, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_nodes = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [idx_dropout_layers[layer]]
                else:
                    drop_nodes = (idx_high, idx_low, idx_rand)
                    drop_layer = copy(idx_dropout_layers)

                for drop_node, drop_type in zip(drop_nodes, scores):
                    
                    # get output with targeted dropout (batch_size x out_features)
                    scores[drop_type]['out'] = [
                        net.module.forward_targeted_dropout(
                            images, [drop[idx, :] for drop in drop_node], drop_layer
                        )[0]
                        for idx, net in enumerate(nets)
                    ]

                    # get loss with targeted dropout (loss_high, loss_low, loss_rand)
                    scores[drop_type]['loss'] = [dataset.measure_loss(out, labels).item()
                                                         for out in scores[drop_type]['out']]

                    # get accuracy with targeted dropout
                    scores[drop_type]['acc'] = [dataset.measure_accuracy(out, labels) for out in scores[drop_type]['out']]

                    scores[drop_type]['progdrop_loss'][:, dropidx, layer] += torch.tensor(scores[drop_type]['loss'], device=internal_device)
                    scores[drop_type]['progdrop_acc'][:, dropidx, layer] += torch.tensor(scores[drop_type]['acc'],  device=internal_device)

    if dataset.distributed:
        for drop_type in scores:
            dist.all_reduce(scores[drop_type]['progdrop_loss'], op=dist.ReduceOp.SUM)
            dist.all_reduce(scores[drop_type]['progdrop_acc'], op=dist.ReduceOp.SUM)
            # Move back to cpu.
            scores[drop_type]['progdrop_loss'] = scores[drop_type]['progdrop_loss'].cpu()
            scores[drop_type]['progdrop_acc'] = scores[drop_type]['progdrop_acc'].cpu()
        num_batches = torch.tensor(num_batches, device=internal_device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        num_batches = num_batches.cpu()
    
    results = {
        "progdrop_loss_high": scores['high']['progdrop_loss'] / num_batches,
        "progdrop_loss_low": scores['low']['progdrop_loss'] / num_batches,
        "progdrop_loss_rand": scores['rand']['progdrop_loss'] / num_batches,
        "progdrop_acc_high": scores['high']['progdrop_acc'] / num_batches,
        "progdrop_acc_low": scores['low']['progdrop_acc'] / num_batches,
        "progdrop_acc_rand": scores['rand']['progdrop_acc'] / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results


@torch.no_grad()
@test_nets
def eigenvector_dropout(nets, dataset, eigenvalues, eigenvectors, **parameters):
    """
    method for testing network on supervised learning problem with eigenvector dropout

    takes as input a list of networks (usually trained) and a dataset, along with the
    eigenvectors of the input to each alignment layer along with some experiment
    parameters (although there are defaults coded into the method)

    note that this only works when each network in the list has the same architecture!
    to analyze a group of networks with different architectures, run this function multiple
    times and concatenate the results.

    eigenvectors must have the following structure:
    a list of lists of eigenvectors to each layer for each network such that:
    len(eigenvectors) = num_networks
    len(eigenvectors[i]) = num_alignment_layers for all i
    eigenvectors[i][j].shape = (num_dim_input_to_j, num_dim_input_to_j)

    eigenvalues should have same structure except have vectors instead of square matrices

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more eigenvectors based on the highest,
    lowest, or random eigenvalues. Can either do it for each layer separately or all
    together using the parameters['by_layer'] kwarg.
    """

    # input argument check
    if not (isinstance(nets, list)):
        nets = [nets]

    # get index to each alignment layer
    idx_dropout_layers = nets[0].module.get_alignment_layer_indices()

    # If using distributed processing, take only eigenvalues corresponding to subset of data in sampler.
    if dataset.distributed:
        rank = dist.get_rank()  # will be index of gather eigenvalues/vectors
        eigenvalues = [[layer[rank] for layer in evals] for evals in eigenvalues]
        eigenvectors = [[layer[rank] for layer in evecs] for evecs in eigenvectors]

    # check if alignment has the right length (ie number of layers) (otherwise can't make assumptions about where the classification layer is)
    assert all(
        [len(ev) == len(idx_dropout_layers) for ev in eigenvectors]
    ), "the number of layers in **eigenvectors** doesn't correspond to the number of alignment layers"
    assert all(
        [len(ev) == len(idx_dropout_layers) for ev in eigenvalues]
    ), "the number of layers in **eigenvalues** doesn't correspond to the number of alignment layers"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get("num_drops", 9)
    drop_fraction = torch.linspace(0, 1, num_drops + 2)[1:-1]
    by_layer = parameters.get("by_layer", False)
    num_layers = len(idx_dropout_layers) if by_layer else 1

    # create index of eigenvalue for compatibility with get_dropout_indices
    idx_eigenvalue = [torch.fliplr(torch.tensor(range(0, ev.size(1))).expand(num_nets, -1)) for ev in eigenvectors[0]]

    # only need to store scores on GPU if using distributed processing.
    internal_device = dataset.device if dataset.distributed else 'cpu'

    # preallocate tracker tensors
    scores = {'high': {},
              'low': {},
              'rand': {}
             }
    for key in scores:
        scores[key]['progdrop_loss'] = torch.zeros((num_nets, num_drops, num_layers), device=internal_device)
        scores[key]['progdrop_acc'] = torch.zeros((num_nets, num_drops, num_layers), device=internal_device)

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_test = not parameters.get("train_set", True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # let dataloader be outer loop to minimize extract / load / transform time
    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)
        num_batches += 1

        # get dropout indices for this fraction of dropouts
        for dropidx, fraction in enumerate(drop_fraction):
            idx_high, idx_low, idx_rand = get_dropout_indices(idx_eigenvalue, fraction)

            # do drop out for each layer (or across all depending on parameters)
            for layer in range(num_layers):
                if by_layer:
                    drop_nodes = (
                        [idx_high[layer]],
                        [idx_low[layer]],
                        [idx_rand[layer]],
                    )
                    drop_layer = [idx_dropout_layers[layer]]
                    drop_evals = [[evals[layer]] for evals in eigenvalues]
                    drop_evecs = [[evecs[layer]] for evecs in eigenvectors]
                else:
                    drop_nodes = idx_high, idx_low, idx_rand
                    drop_layer = copy(idx_dropout_layers)
                    drop_evals = deepcopy(eigenvalues)
                    drop_evecs = deepcopy(eigenvectors)

                for drop_node, drop_type in zip(drop_nodes, scores):
                    
                    # get output with targeted dropout (batch_size x out_features)
                    scores[drop_type]['out'] = [
                    net.module.forward_eigenvector_dropout(
                        images, evals, evecs, [drop[idx, :] for drop in drop_node], drop_layer
                    )[0]
                    for idx, (net, evals, evecs) in enumerate(zip(nets, drop_evals, drop_evecs))
                ]

                    # get loss with targeted dropout (loss_high, loss_low, loss_rand)
                    scores[drop_type]['loss'] = [dataset.measure_loss(out, labels).item()
                                                         for out in scores[drop_type]['out']]

                    # get accuracy with targeted dropout
                    scores[drop_type]['acc'] = [dataset.measure_accuracy(out, labels) for out in scores[drop_type]['out']]

                    scores[drop_type]['progdrop_loss'][:, dropidx, layer] += torch.tensor(scores[drop_type]['loss'], device=internal_device)
                    scores[drop_type]['progdrop_acc'][:, dropidx, layer] += torch.tensor(scores[drop_type]['acc'],  device=internal_device)

    if dataset.distributed:
        for drop_type in scores:
            dist.all_reduce(scores[drop_type]['progdrop_loss'], op=dist.ReduceOp.SUM)
            dist.all_reduce(scores[drop_type]['progdrop_acc'], op=dist.ReduceOp.SUM)
            # Move back to cpu.
            scores[drop_type]['progdrop_loss'] = scores[drop_type]['progdrop_loss'].cpu()
            scores[drop_type]['progdrop_acc'] = scores[drop_type]['progdrop_acc'].cpu()
        num_batches = torch.tensor(num_batches, device=internal_device)
        dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)
        num_batches = num_batches.cpu()
    
    results = {
        "progdrop_loss_high": scores['high']['progdrop_loss'] / num_batches,
        "progdrop_loss_low": scores['low']['progdrop_loss'] / num_batches,
        "progdrop_loss_rand": scores['rand']['progdrop_loss'] / num_batches,
        "progdrop_acc_high": scores['high']['progdrop_acc'] / num_batches,
        "progdrop_acc_low": scores['low']['progdrop_acc'] / num_batches,
        "progdrop_acc_rand": scores['rand']['progdrop_acc'] / num_batches,
        "dropout_fraction": drop_fraction,
        "by_layer": by_layer,
        "idx_dropout_layers": idx_dropout_layers,
    }

    return results
