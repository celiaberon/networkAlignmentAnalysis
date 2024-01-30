import time

import torch
from tqdm import tqdm

from networkAlignmentAnalysis.utils import (condense_values, save_checkpoint,
                                            transpose_list, value_by_layer)


def train(nets, optimizers, dataset, **parameters):
    """method for training network on supervised learning problem"""

    # input argument checks
    if not(isinstance(nets, list)): nets = [nets]
    if not(isinstance(optimizers, list)): optimizers = [optimizers]
    assert len(nets) == len(optimizers), "nets and optimizers need to be equal length lists"

    # preallocate variables and define metaparameters
    num_nets = len(nets)
    use_train = parameters.get('train_set', True)
    dataloader = dataset.train_loader if use_train else dataset.test_loader
    num_steps = len(dataset.train_loader)*parameters['num_epochs']
    # track_loss = torch.zeros((num_steps, num_nets))
    # track_accuracy = torch.zeros((num_steps, num_nets))

    # --- optional analyses ---
    measure_alignment = parameters.get('alignment', True)
    measure_delta_weights = parameters.get('delta_weights', False)
    measure_avgcorr = parameters.get('average_correlation', True)
    measure_fullcorr = parameters.get('full_correlation', False)
    results = parameters.get('results', False)
    num_complete = parameters.get('num_complete', 0)
    save_ckpt, freq_ckpt, path_ckpt, dev = parameters.get('save_checkpoints', (False, 1, '', ''))
    if not results:
        print('initializing new results dictionary')
        # initialize dictionary for storing performance across epochs
        results = {'loss': torch.zeros((num_steps, num_nets)),
                   'accuracy': torch.zeros((num_steps, num_nets))}

        # measure alignment throughout training
        if measure_alignment:
            results['alignment'] = []

        # measure weight norm throughout training
        if measure_delta_weights:
            results['delta_weights'] = []
            results['init_weights'] = [net.get_alignment_weights() for net in nets]

        # measure average correlation for each layer
        if measure_avgcorr:
            results['avgcorr'] = []

        # measure full correlation for each layer
        if measure_fullcorr:
            results['fullcorr'] = []

    # If loaded from checkpoint but running more epochs than initialized for.
    elif results['loss'].shape[0] < num_steps:
        add_steps = num_steps - results['loss'].shape[0]
        assert (add_steps / (parameters['num_epochs'] - num_complete)) == len(dataset.train_loader), (
            'Number of new steps needs to multiple of epochs and num minibatches')
        results['loss'] = torch.vstack((results['loss'],
                                        torch.zeros((add_steps, num_nets))))
        results['accuracy'] = torch.vstack((results['accuracy'],
                                            torch.zeros((add_steps, num_nets))))

    if num_complete > 0: print('resuming training from checkpoint on epoch', num_complete)

    # --- training loop ---
    for epoch in tqdm(range(num_complete, parameters['num_epochs']), desc="training epoch"):
        for idx, batch in enumerate(tqdm(dataloader, desc="minibatch", leave=False)):
            cidx = epoch*len(dataloader) + idx
            images, labels = dataset.unwrap_batch(batch)

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

            results['loss'][cidx] = torch.tensor([l.item() for l in loss])
            results['accuracy'][cidx] = torch.tensor([dataset.measure_accuracy(output, labels) for output in outputs])

            if measure_alignment:
                # Measure alignment if requested
                results['alignment'].append([net.measure_alignment(images, precomputed=True, method='alignment') 
                                  for net in nets])
            
            if measure_delta_weights:
                # Measure change in weights if requested
                results['delta_weights'].append([net.compare_weights(init_weight)
                                      for net, init_weight in zip(nets, results['init_weights'])])
                
            # note: the double use of measure_correlation is inefficient and could probably 
            # be precomputed once then operated on and appended differently for each term
            if measure_avgcorr:
                results['avgcorr'].append([net.measure_correlation(images, precomputed=True, reduced=True) for net in nets])
            
            if measure_fullcorr:
                results['fullcorr'].append([net.measure_correlation(images, precomputed=True, reduced=False) for net in nets])

        if save_ckpt & (epoch % freq_ckpt == 0):
            save_checkpoint(nets,
                            optimizers,
                            results | {'prms': parameters,
                                       'epoch': epoch,
                                       'device': dev},
                            path_ckpt)

    # condense optional analyses
    for k in ['alignment', 'delta_weights', 'avgcorr', 'fullcorr']:
        if k not in results.keys(): continue
        results[k] = condense_values(transpose_list(results[k]))

    return results


@torch.no_grad()
def test(nets, dataset, **parameters):
    """method for testing network on supervised learning problem"""

    # input argument checks
    if not(isinstance(nets, list)): nets = [nets]

    # preallocate variables and define metaparameters
    num_nets = len(nets)

    # retrieve requested dataloader from dataset
    use_test = not parameters.get('train_set', False) # if train_set=True, use_test=False
    dataloader = dataset.test_loader if use_test else dataset.train_loader

    # Performance Measurements
    total_loss = [0 for _ in range(num_nets)]
    num_correct = [0 for _ in range(num_nets)]
    num_batches = 0
    alignment = []

    # put networks in evaluation mode
    in_training_mode = [net.training for net in nets]
    for net in nets:
        net.eval()

    for batch in tqdm(dataloader):
        images, labels = dataset.unwrap_batch(batch)

        # Perform forward pass
        outputs = [net(images, store_hidden=True) for net in nets]

        # Performance Measurements
        for idx, output in enumerate(outputs):
            total_loss[idx] += dataset.measure_loss(output, labels).item()
            num_correct[idx] += dataset.measure_accuracy(output, labels)
        
        # Keep track of number of batches
        num_batches += 1

        # Measure Alignment
        alignment.append([net.measure_alignment(images, precomputed=True, method='alignment')
                          for net in nets])
    
    results = {
        'loss': [loss / num_batches for loss in total_loss],
        'accuracy': [correct / num_batches for correct in num_correct],
        'alignment': condense_values(transpose_list(alignment)),
    }

    # return networks to whatever mode they used to be in 
    for train_mode, net in zip(in_training_mode, nets):
        if train_mode:
            net.train()

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
    idx_rand = [torch.stack([torch.randperm(nodes)[:drop] for _ in range(num_nets)], dim=0) 
                for nodes, drop in zip(num_nodes, num_drop)]
    return idx_high, idx_low, idx_rand

@torch.no_grad()
def progressive_dropout(nets, dataset, alignment=None, **parameters):
    """
    method for testing network on supervised learning problem with progressive dropout

    takes as input a list of networks (usually trained) and a dataset, along with some
    experiment parameters (although there are defaults coded into the method)

    will measure the loss and accuracy on the dataset using targeted dropout, where
    the method will progressively dropout more and more nodes based on the highest, lowest,
    or random alignment. Can either do it for each layer separately or all togehter using
    the parameters['by_layer'] kwarg. 
    """

    # input argument check
    if not(isinstance(nets, list)): nets = [nets]

    _, dropout_layers = nets[0].get_alignment_layers(include_pos=True)
    last_layer_pos = len(nets[0].layers) - 1
    dropout_layers = [layer for layer in dropout_layers if layer != last_layer_pos]
    dropout_layers
    # number layers that dropout can be performed in
    num_dropout_layers = len(dropout_layers) # (never dropout classification layer)

    # put networks in evaluation mode
    in_training_mode = [net.training for net in nets]
    for net in nets:
        net.eval()

    # get alignment and index of alignment
    if alignment is None:
        alignment = test(nets, dataset, **parameters)['alignment']
    
    alignment = [torch.mean(align, dim=1) for align in alignment[:num_dropout_layers]]
    idx_alignment = [torch.argsort(align, dim=1) for align in alignment]
    
    # preallocate variables and define metaparameters
    num_nets = len(nets)
    num_drops = parameters.get('num_drops', 9)
    drop_fraction = torch.linspace(0,1,num_drops+2)[1:-1]
    by_layer = parameters.get('by_layer', False)
    num_layers = num_dropout_layers if by_layer else 1

    # preallocate tracker tensors
    progdrop_loss_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_loss_rand = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_high = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_low = torch.zeros((num_nets, num_drops, num_layers))
    progdrop_acc_rand = torch.zeros((num_nets, num_drops, num_layers))

    # to keep track of how many values have been added
    num_batches = 0

    # retrieve requested dataloader from dataset
    use_test = not parameters.get('train_set', True)
    dataloader = dataset.test_loader if use_test else dataset.train_loader

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
                    drop_high, drop_low, drop_rand = [idx_high[layer]], [idx_low[layer]], [idx_rand[layer]]
                    drop_layer = [layer]
                else:
                    drop_high, drop_low, drop_rand = idx_high, idx_low, idx_rand
                    drop_layer = [layer for layer in dropout_layers]
                
                # get output with targeted dropout
                out_high = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_high], drop_layer)[0]
                            for idx, net in enumerate(nets)]
                out_low = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_low], drop_layer)[0] 
                           for idx, net in enumerate(nets)]
                out_rand = [net.forward_targeted_dropout(images, [drop[idx, :] for drop in drop_rand], drop_layer)[0] 
                            for idx, net in enumerate(nets)]

                # get loss with targeted dropout
                loss_high = [dataset.measure_loss(out, labels).item() for out in out_high]
                loss_low = [dataset.measure_loss(out, labels).item() for out in out_low]
                loss_rand = [dataset.measure_loss(out, labels).item() for out in out_rand]

                # get accuracy with targeted dropout
                acc_high = [dataset.measure_accuracy(out, labels) for out in out_high]
                acc_low = [dataset.measure_accuracy(out, labels) for out in out_low]
                acc_rand = [dataset.measure_accuracy(out, labels) for out in out_rand]

                # add to storage tensors
                progdrop_loss_high[:, dropidx, layer] += torch.tensor(loss_high)
                progdrop_loss_low[:, dropidx, layer] += torch.tensor(loss_low)
                progdrop_loss_rand[:, dropidx, layer] += torch.tensor(loss_rand)

                progdrop_acc_high[:, dropidx, layer] += torch.tensor(acc_high)
                progdrop_acc_low[:, dropidx, layer] += torch.tensor(acc_low)
                progdrop_acc_rand[:, dropidx, layer] += torch.tensor(acc_rand)
    
    results = {
        'progdrop_loss_high': progdrop_loss_high / num_batches,
        'progdrop_loss_low': progdrop_loss_low / num_batches,
        'progdrop_loss_rand': progdrop_loss_rand / num_batches,
        'progdrop_acc_high': progdrop_acc_high / num_batches,
        'progdrop_acc_low': progdrop_acc_low / num_batches,
        'progdrop_acc_rand': progdrop_acc_rand / num_batches,
        'dropout_fraction': drop_fraction,
        'by_layer': by_layer,
    }

    # return networks to whatever mode they used to be in 
    for train_mode, net in zip(in_training_mode, nets):
        if train_mode:
            net.train()

    return results

