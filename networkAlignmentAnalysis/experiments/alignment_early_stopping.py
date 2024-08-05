import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .. import plotting, processing
from ..experiments.alignment_stats import AlignmentStatistics
from ..models.registry import get_model
from . import arglib
from .experiment import Experiment


class AlignmentEarlyStopping(AlignmentStatistics):

    # def __init__(self, args=None):
    #     super().__init__(args=None)

    def get_basename(self):
        return "alignment_early_stopping"
    
    def make_args(self, parser):
        """
        Method for adding experiment specific arguments to the argument parser
        """
        parser = arglib.add_standard_training_parameters(parser)
        parser = arglib.add_checkpointing(parser)
        parser = arglib.add_dropout_experiment_details(parser)
        parser = arglib.add_network_metaparameters(parser)
        parser = arglib.add_alignment_analysis_parameters(parser)
        parser = arglib.add_early_stopping_parameters(parser)

        return parser

    def main(self):
        """
        main experiment loop

        create networks (this is where the specific experiment is determined)
        train and test networks
        do supplementary analyses
        """

        # create networks
        nets, optimizers, prms = self.create_networks()

        # load dataset
        dataset = self.prepare_dataset(nets[0])

        # train networks
        train_results, test_results = processing.train_networks(self, nets, optimizers, dataset, max_change=self.args.max_change)

        # do targeted dropout experiment
        dropout_results, dropout_parameters = processing.progressive_dropout_experiment(
            self, nets, dataset, alignment=test_results.get("alignment", None), train_set=False
        )

        # measure eigenfeatures
        eigen_results = processing.measure_eigenfeatures(self, nets, dataset, train_set=False)

        # do targeted dropout experiment
        evec_dropout_results, evec_dropout_parameters = processing.eigenvector_dropout(self, nets, dataset, eigen_results, train_set=False)

        # make full results dictionary
        results = dict(
            prms=prms,
            train_results=train_results,
            test_results=test_results,
            dropout_results=dropout_results,
            dropout_parameters=dropout_parameters,
            eigen_results=eigen_results,
            evec_dropout_results=evec_dropout_results,
            evec_dropout_parameters=evec_dropout_parameters,
        )

        # return results and trained networks
        return results, nets
