import logging
import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from . import train
from .utils import (construct_zeros_obj, fgsm_attack, gather_list_of_lists,
                    get_list_dims, get_nested_depth, load_checkpoints,
                    replicate_dimension, test_nets, transpose_list)

logger = logging.getLogger(__name__)

def train_networks(exp, nets, optimizers, dataset, **special_parameters):
    """train and test networks"""
    # do training loop
    parameters = dict(
        train_set=True,
        num_epochs=exp.args.epochs,
        alignment=not (exp.args.no_alignment),
        delta_weights=exp.args.delta_weights,
        frequency=exp.args.measure_frequency,
        run=exp.run,
    )

    # update with special parameters
    parameters.update(**special_parameters)

    if special_parameters.get('max_change', False):
        train_func = train.train_to_plateau
        parameters['max_change'] = special_parameters.get('max_change')
    else:
        train_func = train.train

    if exp.args.use_prev & any(list(exp.get_dir(create=False).glob('checkpoint*'))):
        nets, optimizers, results = load_checkpoints(
            nets, optimizers, exp.device, exp.get_dir(create=False)
        )
        parameters = results.pop('prms')

        for net in nets:
            net.train()

        parameters["num_complete"] = results["epoch"] + 1
        parameters["results"] = results
        print("loaded networks from previous checkpoint")

    if exp.args.save_ckpts:
        parameters["save_ckpt"] = exp.args.save_ckpts
        parameters["freq_ckpt"] = exp.args.rate_ckpts
        parameters["path_ckpt"] = (exp.get_checkpoint_path(), exp.args.unique_ckpts)

    print("training networks...")
    train_results = train_func(nets, optimizers, dataset, **parameters)

    # do testing loop
    print("testing networks...")
    parameters["train_set"] = False
    test_results = train.test(nets, dataset, **parameters)

    return train_results, test_results


def progressive_dropout_experiment(exp, nets, dataset, alignment=None, train_set=False):
    """
    perform a progressive dropout (of nodes) experiment
    alignment is optional, but will be recomputed if you've already measured it. You can provide it
    by setting: alignment=test_results['alignment'] if ``train_networks`` has already been run.
    """
    # do targeted dropout experiment
    print("performing targeted dropout...")
    if dataset.distributed:
        logger.info(f'rank {dist.get_rank()} starting dropout')
    else:
        logger.info('starting dropout')
    dropout_parameters = dict(
        num_drops=exp.args.num_drops, by_layer=exp.args.dropout_by_layer, train_set=train_set
    )
    dropout_results = train.progressive_dropout(
        nets, dataset, alignment=alignment, **dropout_parameters
    )

    return dropout_results, dropout_parameters


def min_samples_per_class(labels):
    # get stacked indices to the elements of each class
    classes, num_samples = torch.unique(labels, return_counts=True)
    min_per_class = min(num_samples)

    return min_per_class


def measure_eigenfeatures(exp, nets, dataset, train_set=False):
    # measure eigenfeatures
    print("measuring eigenfeatures...")
    if dataset.distributed:
        logger.info(f'rank {dist.get_rank()} measuring eigenfeatures')
    results = {'beta': [],
               'eigvals': [],
               'eigvecs': [],
               'class_betas': []}
    for net in tqdm(nets):
        # get inputs to each layer from whole dataloader
        inputs, labels = net.module._process_collect_activity(
            dataset,
            train_set=train_set,
            with_updates=False,
            use_training_mode=False,
        )
        
        if dataset.distributed:
            min_per_class = torch.tensor(min_samples_per_class(labels), device=dataset.device)
            logger.info(f'{dist.get_rank()} sample limit = {min_per_class}')
            dist.all_reduce(min_per_class, op=dist.ReduceOp.MIN)
            # min_per_class = min_per_class.cpu()
            logger.info(f'{dist.get_rank()} sample limit = {min_per_class}')
        else:
            min_per_class = torch.inf

        beta, eigvals, eigvecs = net.module.measure_eigenfeatures(inputs, with_updates=False)
        beta_by_class = net.module.measure_class_eigenfeatures(
            inputs, labels, eigvecs, rms=False, with_updates=False, num_samples_per_class=min_per_class
        )
        results['beta'].append(beta)
        results['eigvals'].append(eigvals)
        results['eigvecs'].append(eigvecs)
        results['class_betas'].append(beta_by_class)

    if dataset.distributed:
        logger.info(f'rank {dist.get_rank()} waiting on barrier')
        dist.barrier()
        for key, metric in results.items():
            depth = get_nested_depth(metric)
            agg_metric = replicate_dimension(construct_zeros_obj(metric, device=dataset.device),
                                             target_dim=depth,
                                             n_reps=dist.get_world_size())

            logger.info(f'{key}\n{get_list_dims(agg_metric)}')
            logger.info(f'rank {dist.get_rank()} pre gather for {key}')
            gather_list_of_lists(metric, agg_metric, device=dataset.device, move_to_gpu=True)
            logger.info(f'rank {dist.get_rank()} post gather for {key}')
            # Consider: Transpose agg_metric to put process back on outer dimension for easy allocation.
            # Currently: (nets, layer x proc)
            results[key] = agg_metric

    results['class_names'] = getattr(
        dataset.train_loader if train_set else dataset.test_loader, "dataset"
    ).classes

    if dataset.distributed:
        print(dist.get_rank(), results['class_names'])  # need to confirm always the same, even as dataset grows

    return results


def eigenvector_dropout(exp, nets, dataset, eigen_results, train_set=False):
    """
    do targeted eigenvector dropout with precomputed eigenfeatures
    """
    # do targeted dropout experiment
    print("performing targeted eigenvector dropout...")
    evec_dropout_parameters = dict(num_drops=exp.args.num_drops, by_layer=exp.args.dropout_by_layer, train_set=train_set)
    evec_dropout_results = train.eigenvector_dropout(nets, dataset, eigen_results["eigvals"], eigen_results["eigvecs"], **evec_dropout_parameters)
    return evec_dropout_results, evec_dropout_parameters


@test_nets
def measure_adversarial_attacks(nets, dataset, exp, eigen_results, train_set=False, **parameters):
    """
    do adversarial attack and measure structure with regards to eigenfeatures
    """

    def get_beta(inputs, eigenvectors):
        # get projection of input onto eigenvectors across layers
        return [input.cpu() @ evec for input, evec in zip(inputs, eigenvectors)]

    # experiment parameters
    epsilons = parameters.get("epsilons")
    use_sign = parameters.get("use_sign")
    fgsm_transform = parameters.get("fgsm_transform", lambda x: x)

    # data from eigenvectors
    eigenvectors = eigen_results["eigvecs"]

    num_eps = len(epsilons)
    num_nets = len(nets)
    accuracy = torch.zeros((num_nets, num_eps))
    examples = [[[] for _ in range(num_eps)] for _ in range(num_nets)]
    betas = [[torch.zeros((num_nets, evec.size(0))) for evec in eigenvectors[0]] for _ in range(num_eps)]

    # dataloader
    dataloader = dataset.train_loader if train_set else dataset.test_loader

    for batch in tqdm(dataloader):
        input, labels = dataset.unwrap_batch(batch)

        inputs = [input.clone() for _ in range(num_nets)]

        for input in inputs:
            input.requires_grad = True

        # Forward pass the data through the model
        outputs = [net(input, store_hidden=True) for net, input in zip(nets, inputs)]
        input_to_layers = [net.get_layer_inputs(input, precomputed=True) for net in nets]
        init_preds = [torch.argmax(output, axis=1) for output in outputs]  # find true prediction
        least_likely = [torch.argmin(output, axis=1) for output in outputs]  # find least likely digit according to model

        c_betas = transpose_list([get_beta(input, evec) for input, evec in zip(input_to_layers, eigenvectors)])
        s_betas = [torch.stack(cb) for cb in c_betas]

        # Calculate the loss
        loss = [dataset.measure_loss(output, labels) for output in outputs]
        # loss = dataset.measure_loss(output, least_likely)

        # Zero all existing gradients
        for net in nets:
            net.zero_grad()

        # Calculate gradients of model in backward pass
        for l in loss:
            l.backward()

        # Collect datagrad
        data_grads = [input.grad.data for input in inputs]

        for epsidx, eps in enumerate(epsilons):

            # Call FGSM Attack
            perturbed_inputs = [fgsm_attack(input, eps, data_grad, fgsm_transform, use_sign) for input, data_grad in zip(inputs, data_grads)]

            # Re-classify the perturbed image
            outputs = [net(perturbed_input, store_hidden=True) for net, perturbed_input in zip(nets, perturbed_inputs)]
            input_to_layers = [net.get_layer_inputs(perturbed_input, precomputed=True) for net, perturbed_input in zip(nets, perturbed_inputs)]
            c_eps_betas = transpose_list([get_beta(input, evec) for input, evec in zip(input_to_layers, eigenvectors)])
            s_eps_betas = [torch.stack(ceb) for ceb in c_eps_betas]
            d_eps_betas = [sebeta - sbeta for sebeta, sbeta in zip(s_eps_betas, s_betas)]
            rms_betas = [torch.sqrt(torch.mean(db**2, dim=1)) for db in d_eps_betas]

            for ii, rbeta in enumerate(rms_betas):
                betas[epsidx][ii] += rbeta.detach()

            # Check for success
            final_preds = [torch.argmax(output, axis=1) for output in outputs]
            accuracy[:, epsidx] += torch.tensor([sum(final_pred == labels).cpu() for final_pred in final_preds])

            # Idx where adversarial example worked
            idx_success = [
                torch.where((init_pred == labels) & (final_pred != labels))[0].cpu() for init_pred, final_pred in zip(init_preds, final_preds)
            ]

            adv_exs = [perturbed_input.detach().cpu().numpy() for perturbed_input in perturbed_inputs]
            for ii, (adv_ex, idx, init_pred, final_pred) in enumerate(zip(adv_exs, idx_success, init_preds, final_preds)):
                examples[ii][epsidx].append((init_pred[idx], final_pred[idx], adv_ex[idx]))

    # Calculate final accuracy for this epsilon
    accuracy = accuracy / float(len(dataloader.dataset))

    # Average across betas
    betas = transpose_list([[cb / float(len(dataloader.dataset)) for cb in beta] for beta in betas])

    # Return the accuracy and an adversarial example
    return dict(accuracy=accuracy, betas=betas, examples=examples, epsilons=epsilons, use_sign=use_sign)


@test_nets
def measure_alignment_distribution(nets, dataset, **parameters):
    """
    method for measuring alignment distribution and several associated analyses
    """
    # do training loop
    parameters = dict(
        train_set=True,
   )
