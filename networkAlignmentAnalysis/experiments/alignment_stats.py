import os

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

import wandb

from .. import train
from ..datasets import get_dataset
from ..models.registry import get_model
from ..utils import (compute_stats_by_type, load_checkpoints, named_transpose,
                     rms, transpose_list)
from .experiment import Experiment


class AlignmentStatistics(Experiment):
    def get_basename(self):
        return 'alignment_stats'
    
    def prepare_path(self):
        return [self.args.network, self.args.dataset, self.args.optimizer]
    
    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """

        # Network & Dataset
        parser.add_argument('--network', type=str, default='MLP') # what base network architecture to use
        parser.add_argument('--dataset', type=str, default='MNIST') # what dataset to use
        parser.add_argument('--optimizer', type=str, default='Adam') # what optimizer to train with
        parser.add_argument('--batch_size', type=int, default=1024) # batch size to pass to DataLoader


        # default parameters
        parser.add_argument('--default-lr', type=float, default=1e-3) # default learning rate
        parser.add_argument('--default-dropout', type=float, default=0) # default dropout rate
        parser.add_argument('--default-wd', type=float, default=0) # default weight decay

        # progressive dropout parameters
        parser.add_argument('--num_drops', type=int, default=9, help='number of dropout fractions for progressive dropout')
        parser.add_argument('--dropout_by_layer', default=False, action='store_true', 
                            help='whether to do progressive dropout by layer or across all layers')
        
        # some metaparameters for the experiment
        parser.add_argument('--epochs', type=int, default=100) # how many rounds of training to do
        parser.add_argument('--replicates', type=int, default=5) # how many copies of identical networks to train
        parser.add_argument('--use-flag', default=False, action='store_true', help='if used, will include flagged layers in analyses')


        # checkpointing parameters
        parser.add_argument('--use_prev', default=False, action='store_true', help='if used, will pick up training off previous checkpoint')
        parser.add_argument('--save_ckpts', default=False, action='store_true', help='if used, will save checkpoints of models')
        parser.add_argument('--use_wandb', default=False, action='store_true', help='if used, will log experiment to WandB')

        # return parser
        return parser
    
    def main(self):
        """
        main experiment loop
        
        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """

        run = self.configure_wandb()

        # load networks 
        nets, optimizers = self.load_networks()

        # load dataset
        dataset = get_dataset(self.args.dataset,
                              build=True,
                              transform_parameters=nets[0],
                              device=self.args.device)

        # train networks
        train_results, test_results = self.train_networks(nets, optimizers, dataset, run)

        # do targeted dropout experiment
        print('performing targeted dropout...')
        dropout_parameters = dict(num_drops=self.args.num_drops, by_layer=self.args.dropout_by_layer)
        dropout_results = train.progressive_dropout(nets, dataset, alignment=test_results['alignment'], **dropout_parameters)

        # measure eigenfeatures
        print('measuring eigenfeatures...')
        beta, eigvals, eigvecs, class_betas = [], [], [], []
        for net in tqdm(nets):
            eigenfeatures = net.measure_eigenfeatures(dataset.test_loader, with_updates=False)
            beta_by_class = net.measure_class_eigenfeatures(dataset.test_loader, eigenfeatures[2], rms=False, with_updates=False)
            beta.append(eigenfeatures[0])
            eigvals.append(eigenfeatures[1])
            eigvecs.append(eigenfeatures[2])
            class_betas.append(beta_by_class)

        # we don't actually use the eigvecs for anything right now, eigvecs=eigvecs)    
        eigen_results = dict(beta=beta, eigvals=eigvals, class_betas=class_betas, class_names=dataset.test_loader.dataset.classes) 

        # make full results dictionary
        results = dict(
            train_results=train_results,
            test_results=test_results,
            dropout_results=dropout_results,
            dropout_parameters=dropout_parameters,
            eigen_results=eigen_results
        )    

        # return results and trained networks
        return results, nets

    def configure_wandb(self):
        if self.args.use_wandb:
            wandb.login()
            run = wandb.init(
                project='alignment_stats',
                name='',
                config=self.args
            )
        if str(self.basepath).startswith('/n/home00/cberon'):
            os.environ['WANDB_MODE'] = 'offline'

        return run

    def plot(self, results):
        """
        main plotting loop
        """
        self.plot_train_results(results['train_results'], results['test_results'])
        self.plot_dropout_results(results['dropout_results'], results['dropout_parameters'])
        self.plot_eigenfeatures(results['eigen_results'])

    # ----------------------------------------------
    # ------ methods for main experiment loop ------
    # ----------------------------------------------
    def load_networks(self):
        """
        method for loading networks

        depending on the experiment parameters (which comparison, which metaparams etc)
        this method will create multiple networks with requested parameters and return
        their optimizers and a params dictionary with the experiment parameters associated
        with each network
        """
        # get optimizer
        if self.args.optimizer == 'Adam':
            optim = torch.optim.Adam
        elif self.args.optimizer == 'SGD':
            optim = torch.optim.SGD
        else:
            raise ValueError(f"optimizer ({self.args.optimizer}) not recognized")
        
        nets = [get_model(self.args.network, build=True, dataset=self.args.dataset, dropout=self.args.default_dropout, ignore_flag=not(self.args.use_flag))
                for _ in range(self.args.replicates)]
        nets = [net.to(self.device) for net in nets]
        optimizers = [optim(net.parameters(), lr=self.args.default_lr, weight_decay=self.args.default_wd)
                      for net in nets]
        
        return nets, optimizers


    def train_networks(self, nets, optimizers, dataset, run=None):
        """train and test networks"""
        # do training loop
        parameters = dict(
            train_set=True,
            num_epochs=self.args.epochs,
            alignment=True,
            delta_weights=True,
            average_correlation=True,
            full_correlation=False,
            run=run
        )

        if self.args.use_prev & os.path.isfile(self.get_checkpoint_path()):
            nets, optimizers, results = load_checkpoints(nets,
                                                         optimizers,
                                                         self.args.device,
                                                         self.get_checkpoint_path())
            [net.train() for net in nets]
            parameters['num_complete'] = results['epoch'] + 1
            parameters['results'] = results
            print('loaded networks from previous checkpoint')

        if self.args.save_ckpts:
            parameters['save_checkpoints'] = (True, 1, self.get_checkpoint_path(), self.args.device)

        print('training networks...')
        train_results = train.train(nets, optimizers, dataset, **parameters)

        # do testing loop
        print('testing networks...')
        parameters['train_set'] = False
        test_results = train.test(nets, dataset, **parameters)

        return train_results, test_results

    # ----------------------------------------------
    # ------- methods for main plotting loop -------
    # ----------------------------------------------
    def plot_train_results(self, train_results, test_results):
        """
        plotting method for training trajectories and testing data
        """

        num_train_epochs = train_results['loss'].size(0)
        num_types = 1
        labels = [f"{self.args.network}"]

        print("getting statistics on run data...")
        alignment = torch.stack([torch.mean(align, dim=2) for align in train_results['alignment']])
        correlation = torch.stack([torch.mean(corr, dim=2) for corr in train_results['avgcorr']])
        
        cmap = mpl.colormaps['tab10']

        train_loss_mean, train_loss_se = compute_stats_by_type(train_results['loss'], 
                                                                num_types=num_types, dim=1, method='se')
        train_acc_mean, train_acc_se = compute_stats_by_type(train_results['accuracy'],
                                                                num_types=num_types, dim=1, method='se')

        align_mean, align_se = compute_stats_by_type(alignment, num_types=num_types, dim=1, method='se')

        corr_mean, corr_se = compute_stats_by_type(correlation, num_types=num_types, dim=1, method='se')

        test_loss_mean, test_loss_se = compute_stats_by_type(torch.tensor(test_results['loss']),
                                                                num_types=num_types, dim=0, method='se')
        test_acc_mean, test_acc_se = compute_stats_by_type(torch.tensor(test_results['accuracy']),
                                                                num_types=num_types, dim=0, method='se')


        print("plotting run data...")
        xOffset = [-0.2, 0.2]
        get_x = lambda idx: [xOffset[0]+idx, xOffset[1]+idx]

        # Make Training and Testing Performance Figure
        alpha = 0.3
        figdim = 3
        figratio = 2
        width_ratios = [figdim, figdim/figratio, figdim, figdim/figratio]

        fig, ax = plt.subplots(1, 4, figsize=(sum(width_ratios), figdim), width_ratios=width_ratios, layout='constrained')

        # plot loss results fot training and testing
        for idx, label in enumerate(labels):
            cmn = train_loss_mean[:, idx]
            cse = train_loss_se[:, idx]
            tmn = test_loss_mean[idx]
            tse = test_loss_se[idx]

            ax[0].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[0].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))
            ax[1].plot(get_x(idx), [tmn]*2, color=cmap(idx), label=label, lw=4)
            ax[1].plot([idx, idx], [tmn-tse, tmn+tse], color=cmap(idx), lw=1.5)
            
        ax[0].set_xlabel('Training Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].set_title('Training Loss')
        ax[0].set_ylim(0, None)
        ylims = ax[0].get_ylim()
        ax[1].set_xticks(range(num_types))
        ax[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[1].set_ylabel('Loss')
        ax[1].set_title('Testing')
        ax[1].set_xlim(-0.5, num_types-0.5)
        ax[1].set_ylim(ylims)

        # plot loss results fot training and testing
        for idx, label in enumerate(labels):
            cmn = train_acc_mean[:, idx]
            cse = train_acc_se[:, idx]
            tmn = test_acc_mean[idx]
            tse = test_acc_se[idx]

            ax[2].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
            ax[2].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))
            ax[3].plot(get_x(idx), [tmn]*2, color=cmap(idx), label=label, lw=4)
            ax[3].plot([idx, idx], [tmn-tse, tmn+tse], color=cmap(idx), lw=1.5)
            
        ax[2].set_xlabel('Training Epoch')
        ax[2].set_ylabel('Accuracy (%)')
        ax[2].set_title('Training Accuracy')
        ax[2].set_ylim(0, 100)
        ax[3].set_xticks(range(num_types))
        ax[3].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax[3].set_ylabel('Accuracy (%)')
        ax[3].set_title('Testing')
        ax[3].set_xlim(-0.5, num_types-0.5)
        ax[3].set_ylim(0, 100)

        self.plot_ready('train_test_performance')

        # Make Alignment Figure
        num_layers = align_mean.size(0)
        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained', sharex=True)
        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                cmn = align_mean[layer, idx] * 100
                cse = align_se[layer, idx] * 100
                ax[layer].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))

        for layer in range(num_layers):
            ax[layer].set_ylim(0, None)
            ax[layer].set_xlabel('Training Epoch')
            ax[layer].set_ylabel('Alignment (%)')
            ax[layer].set_title(f"Layer {layer}")

        ax[0].legend(loc='lower right')

        self.plot_ready('train_alignment_by_layer')


        # Make Correlation Figure
        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained', sharex=True)
        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                cmn = corr_mean[layer, idx]
                cse = corr_se[layer, idx]
                ax[layer].plot(range(num_train_epochs), cmn, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_train_epochs), cmn+cse, cmn-cse, color=(cmap(idx), alpha))

        for layer in range(num_layers):
            ax[layer].set_ylim(0, None)
            ax[layer].set_xlabel('Training Epoch')
            ax[layer].set_ylabel('Correlation')
            ax[layer].set_title(f"Layer {layer}")

        ax[0].legend(loc='lower right')

        self.plot_ready('train_correlation_by_layer')


    def plot_dropout_results(self, dropout_results, dropout_parameters):
        num_types = 1
        labels = [f"{self.args.network}"]
        cmap = mpl.colormaps['Set1']
        alpha = 0.3
        msize = 10
        figdim = 3

        num_layers = dropout_results['progdrop_loss_high'].size(2)
        names = ['From high', 'From low', 'Random']
        num_exp = len(names)
        dropout_fraction = dropout_results['dropout_fraction']
        by_layer = dropout_results['by_layer']
        extra_name = 'by_layer' if by_layer else 'all_layers'

        # Get statistics across each network type for progressive dropout experiment
        print("measuring statistics on dropout analysis...")
        loss_mean_high, loss_se_high = compute_stats_by_type(dropout_results['progdrop_loss_high'], 
                                                                num_types=num_types, dim=0, method='se')
        loss_mean_low, loss_se_low = compute_stats_by_type(dropout_results['progdrop_loss_low'], 
                                                                num_types=num_types, dim=0, method='se')
        loss_mean_rand, loss_se_rand = compute_stats_by_type(dropout_results['progdrop_loss_rand'], 
                                                                num_types=num_types, dim=0, method='se')

        acc_mean_high, acc_se_high = compute_stats_by_type(dropout_results['progdrop_acc_high'], 
                                                                num_types=num_types, dim=0, method='se')
        acc_mean_low, acc_se_low = compute_stats_by_type(dropout_results['progdrop_acc_low'], 
                                                                num_types=num_types, dim=0, method='se')
        acc_mean_rand, acc_se_rand = compute_stats_by_type(dropout_results['progdrop_acc_rand'], 
                                                                num_types=num_types, dim=0, method='se')

        # Contract into lists for looping through to plot
        loss_mean = [loss_mean_high, loss_mean_low, loss_mean_rand]
        loss_se = [loss_se_high, loss_se_low, loss_se_rand]
        acc_mean = [acc_mean_high, acc_mean_low, acc_mean_rand]
        acc_se = [acc_se_high, acc_se_low, acc_se_rand]


        print("plotting dropout results...")
        # Plot Loss for progressive dropout experiment
        fig, ax = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim), sharex=True, sharey=True, layout='constrained')
        ax = np.reshape(ax, (num_layers, num_types))

        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                for iexp, name in enumerate(names):
                    cmn = loss_mean[iexp][idx, :, layer]
                    cse = loss_se[iexp][idx, :, layer]
                    ax[layer, idx].plot(dropout_fraction, cmn, color=cmap(iexp), marker='.', markersize=msize, label=name)
                    ax[layer, idx].fill_between(dropout_fraction, cmn+cse, cmn-cse, color=(cmap(iexp), alpha))
            
                if layer==0:
                    ax[layer, idx].set_title(label)

                if layer==num_layers-1:
                    ax[layer, idx].set_xlabel('Dropout Fraction')
                    ax[layer, idx].set_xlim(0, 1)
                
                if idx==0:
                    ax[layer, idx].set_ylabel('Loss w/ Dropout')

                if iexp==num_exp-1:
                    ax[layer, idx].legend(loc='best')
        
        self.plot_ready('prog_dropout_'+extra_name+'_loss')


        fig, ax = plt.subplots(num_layers, num_types, figsize=(num_types*figdim, num_layers*figdim), sharex=True, sharey=True, layout='constrained')
        ax = np.reshape(ax, (num_layers, num_types))

        for idx, label in enumerate(labels):
            for layer in range(num_layers):
                for iexp, name in enumerate(names):
                    cmn = acc_mean[iexp][idx, :, layer]
                    cse = acc_se[iexp][idx, :, layer]
                    ax[layer, idx].plot(dropout_fraction, cmn, color=cmap(iexp), marker='.', markersize=msize, label=name)
                    ax[layer, idx].fill_between(dropout_fraction, cmn+cse, cmn-cse, color=(cmap(iexp), alpha))

                ax[layer, idx].set_ylim(0, 100)

                if layer==0:
                    ax[layer, idx].set_title(label)

                if layer==num_layers-1:
                    ax[layer, idx].set_xlabel('Dropout Fraction')
                    ax[layer, idx].set_xlim(0, 1)
                
                if idx==0:
                    ax[layer, idx].set_ylabel('Accuracy w/ Dropout')

                if iexp==num_exp-1:
                    ax[layer, idx].legend(loc='best')
        
        self.plot_ready('prog_dropout_'+extra_name+'_accuracy')


    def plot_eigenfeatures(self, results):
        """method for plotting results related to eigen-analysis"""
        beta, eigvals, class_betas, class_names = results['beta'], results['eigvals'], results['class_betas'], results['class_names']
        beta = [[torch.abs(b) for b in net_beta] for net_beta in beta]
        class_betas = [[rms(cb, dim=2) for cb in net_class_beta] for net_class_beta in class_betas]

        num_types = 1
        labels = [f"{self.args.network}"]
        cmap = mpl.colormaps['tab10']
        class_cmap = mpl.colormaps['viridis'].resampled(len(class_names))

        print("measuring statistics of eigenfeature analyses...")

        # shape wrangling
        beta = [torch.stack(b) for b in transpose_list(beta)]
        eigvals = [torch.stack(ev) for ev in transpose_list(eigvals)]
        class_betas = [torch.stack(cb) for cb in transpose_list(class_betas)]

        # normalize to relative values
        beta = [b / b.sum(dim=2, keepdim=True) for b in beta]
        eigvals = [ev / ev.sum(dim=1, keepdim=True) for ev in eigvals]
        class_betas = [cb / cb.sum(dim=2, keepdim=True) for cb in class_betas]


        # reuse these a few times
        statprms = lambda method: dict(num_types=num_types, dim=0, method=method)

        # get mean and variance eigenvalues for each layer for each network type
        mean_evals, var_evals = named_transpose([compute_stats_by_type(ev, **statprms('var')) for ev in eigvals])

        # get sorted betas (sorted within each neuron)
        sorted_beta = [torch.sort(b, descending=True, dim=2).values for b in beta]

        # get mean / se beta for each layer for each network type
        mean_beta, se_beta = named_transpose([compute_stats_by_type(b, **statprms('var')) for b in beta])
        mean_sorted, se_sorted = named_transpose([compute_stats_by_type(b, **statprms('var')) for b in sorted_beta])
        mean_class_beta, se_class_beta = named_transpose([compute_stats_by_type(cb, **statprms('var')) for cb in class_betas])

        print("plotting eigenfeature results...")
        figdim = 3
        alpha = 0.3
        num_layers = len(mean_beta)
        fig, ax = plt.subplots(2, num_layers, figsize=(num_layers*figdim, figdim*2), layout='constrained')

        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            num_nodes = mean_beta[layer].size(1)
            for idx, label in enumerate(labels):
                mn_ev = mean_evals[layer][idx]
                se_ev = var_evals[layer][idx]
                mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
                se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
                mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
                se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
                ax[0, layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)
                ax[0, layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
                ax[0, layer].fill_between(range(num_input), mn_beta+se_beta, mn_beta-se_beta, color=(cmap(idx), alpha))
                ax[1, layer].plot(range(num_input), mn_sort, color=cmap(idx), label=label)
                ax[1, layer].fill_between(range(num_input), mn_sort+se_sort, mn_sort-se_sort, color=(cmap(idx), alpha))
                
                ax[0, layer].set_xscale('log')
                ax[1, layer].set_xscale('log')
                ax[0, layer].set_xlabel('Input Dimension')
                ax[1, layer].set_xlabel('Sorted Input Dim')
                ax[0, layer].set_ylabel('Relative Eigval / Beta')
                ax[1, layer].set_ylabel('Relative Beta (Sorted)')
                ax[0, layer].set_title(f"Layer {layer}")
                ax[1, layer].set_title(f"Layer {layer}")

                if layer==num_layers-1:
                    ax[0, layer].legend(loc='best')
                    ax[1, layer].legend(loc='best')

        self.plot_ready('eigenfeatures')


        fig, ax = plt.subplots(1, num_layers, figsize=(num_layers*figdim, figdim), layout='constrained')

        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            num_nodes = mean_beta[layer].size(1)
            for idx, label in enumerate(labels):
                mn_ev = mean_evals[layer][idx]
                se_ev = var_evals[layer][idx]
                mn_beta = torch.mean(mean_beta[layer][idx], dim=0)
                se_beta = torch.std(mean_beta[layer][idx], dim=0) / np.sqrt(num_nodes)
                mn_sort = torch.mean(mean_sorted[layer][idx], dim=0)
                se_sort = torch.std(mean_sorted[layer][idx], dim=0) / np.sqrt(num_nodes)
                ax[layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)
                ax[layer].plot(range(num_input), mn_beta, color=cmap(idx), label=label)
                ax[layer].fill_between(range(num_input), mn_beta+se_beta, mn_beta-se_beta, color=(cmap(idx), alpha))

                ax[layer].set_xscale('log')
                ax[layer].set_yscale('log')
                ax[layer].set_xlabel('Input Dimension')
                ax[layer].set_ylabel('Relative Eigval / Beta')
                ax[layer].set_title(f"Layer {layer}")

                if layer==num_layers-1:
                    ax[layer].legend(loc='best')

        self.plot_ready('eigenfeatures_loglog')


        fig, ax = plt.subplots(2, num_layers, figsize=(num_layers*figdim, figdim*2), layout='constrained')
        for layer in range(num_layers):
            num_input = mean_evals[layer].size(1)
            for idx, label in enumerate(labels):
                mn_ev = mean_evals[layer][idx]
                se_ev = var_evals[layer][idx]
                # plot eigenvalues of each eigenvector
                ax[0, layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)
                ax[1, layer].plot(range(num_input), mn_ev, color=cmap(idx), linestyle='--', label='eigvals' if idx==0 else None)

                for idx_class, class_name in enumerate(class_names):
                    mn_data = mean_class_beta[layer][idx][idx_class]
                    se_data = se_class_beta[layer][idx][idx_class]
                    ax[0, layer].plot(range(num_input), mn_data, color=class_cmap(idx_class), label=class_name)
                    ax[0, layer].fill_between(range(num_input), mn_data+se_data, mn_data-se_data, color=(class_cmap(idx_class), alpha))
                    ax[1, layer].plot(range(num_input), mn_data, color=class_cmap(idx_class), label=class_name)
                    ax[1, layer].fill_between(range(num_input), mn_data+se_data, mn_data-se_data, color=(class_cmap(idx_class), alpha))
                
                ax[0, layer].set_xscale('log')
                ax[1, layer].set_xscale('log')
                ax[1, layer].set_yscale('log')
                ax[0, layer].set_xlabel('Input Dimension')
                ax[1, layer].set_xlabel('Sorted Input Dim')
                ax[0, layer].set_ylabel('Relative Eigval / Class Loading (RMS)')
                ax[1, layer].set_ylabel('Relative Class Loading (RMS)')
                ax[0, layer].set_title(f"Layer {layer}")
                ax[1, layer].set_title(f"Layer {layer}")

                if layer==num_layers-1:
                    ax[0, layer].legend(loc='best', fontsize=8)
                    ax[1, layer].legend(loc='best', fontsize=8)

        self.plot_ready('class_eigenfeatures')


        
