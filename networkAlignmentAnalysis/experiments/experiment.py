import os
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Tuple

import torch
import wandb
from matplotlib import pyplot as plt

from .. import files, train
from ..datasets import get_dataset

class Experiment(ABC):
    def __init__(self, args=None) -> None:
        """Experiment constructor"""
        self.basename = self.get_basename() # Register basename of experiment
        self.basepath = files.results_path() / self.basename # Register basepath of experiment
        self.get_args(args=args) # Parse arguments to python program
        self.register_timestamp() # Register timestamp of experiment
        self.device = self.args.device

    def report(self, init=False, args=False, meta_args=False) -> None:
        """Method for programmatically reporting details about experiment"""
        # Report general details about experiment 
        if init:
            print(f"Experiment object details:")
            print(f"basename: {self.basename}")
            print(f"basepath: {self.basepath}")
            print(f"experiment folder: {self.get_exp_path()}")
            print('using device: ', self.device)

            # Report any other relevant details
            if self.args.save_networks and self.args.nosave:
                print("Note: setting nosave to True will overwrite save_networks. Nothing will be saved.")

        # Report experiment parameters
        if args:
            for key, val in vars(self.args).items():
                if key in self.meta_args: continue
                print(f"{key}={val}")

        # Report experiment meta parameters
        if meta_args:
            for key, val in vars(self.args).items():
                if key not in self.meta_args: continue
                print(f"{key}={val}")

    def register_timestamp(self) -> None:
        """
        Method for registering formatted timestamp.
        
        If timestamp not provided, then the current time is formatted and used to identify this particular experiment.
        If the timestamp is provided, then that time is used and should identify a previously run and saved experiment.
        """
        if self.args.timestamp is not None:
            self.timestamp = self.args.timestamp
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.args.use_timestamp:
                self.args.timestamp = self.timestamp

    def get_dir(self, create=True) -> Path:
        """
        Method for return directory of target file using prepare_path.
        """
        # Make full path to experiment directory
        exp_path = self.basepath / self.get_exp_path()

        # Make experiment directory if it doesn't yet exist
        if create and not(exp_path.exists()): 
            exp_path.mkdir(parents=True)

        return exp_path
    
    def get_exp_path(self) -> Path:
        """Method for returning child directories of this experiment"""
        # exp_path is the base path followed by whatever folders define this particular experiment
        # (usually things like ['network_name', 'dataset_name', 'test', 'etc'])
        exp_path = Path('/'.join(self.prepare_path()))
        
        # if requested, will also use a timestamp to distinguish this run from others
        if self.args.use_timestamp:
            exp_path = exp_path / self.timestamp

        return exp_path
    
    def get_path(self, name, create=True) -> Path:
        """Method for returning path to file"""
        # get experiment directory
        exp_path = self.get_dir(create=create)

        # return full path (including stem)
        return exp_path / name
    
    def configure_wandb(self):
        """create a wandb run file and set environment parameters appropriately"""
        if self.args.use_wandb:
            wandb.login()
            run = wandb.init(
                project=self.get_basename(),
                name='',
                config=self.args,
            )
        
            if str(self.basepath).startswith('/n/home'):
                # ATL Note 240223: We can update the "startswith" list to be
                # a registry of path locations that require WANDB_MODE to be offline
                # in a smarter way, but I think that using /n/ is sufficient in general
                os.environ['WANDB_MODE'] = 'offline'

            return run
        
        return None
        
    @abstractmethod
    def get_basename(self) -> str:
        """Required method for defining the base name of the Experiment"""
        pass

    @abstractmethod
    def prepare_path(self) -> List[str]:
        """
        Required method for defining a pathname for each experiment.

        Must return a list of strings that will be appended to the base path to make an experiment directory.
        See ``get_dir()`` for details.
        """
        pass

    def get_args(self, args=None):
        """
        Method for defining and parsing arguments.
        
        This method defines the standard arguments used for any Experiment, and
        the required method make_args() is used to add any additional arguments
        specific to each experiment.
        """
        self.meta_args = [] # a list of arguments that shouldn't be updated when loading an old experiment
        parser = ArgumentParser(description=f"arguments for {self.basename}")
        parser = self.make_args(parser)
        
        # saving and new experiment loading parameters
        parser.add_argument('--nosave', default=False, action='store_true', help="prevents saving of results or plots")
        parser.add_argument('--justplot', default=False, action='store_true', help="plot saved data without retraining and analyzing networks")
        parser.add_argument('--save-networks', default=False, action='store_true', help="if --nosave wasn't provided, will also save networks that are trained")
        parser.add_argument('--showprms', default=False, action='store_true', help='show parameters of previously saved experiment without doing anything else')
        parser.add_argument('--showall', default=False, action='store_true', help='if true, will show all plots at once rather than having the user close each one for the next')
        parser.add_argument('--device', type=str, default=None, help='which device to use (automatic if not provided)')

        # add meta arguments 
        self.meta_args += ['nosave', 'justplot', 'save_networks', 'showprms', 'showall', 'device']
        
        # common parameters that shouldn't be updated when loading old experiment
        parser.add_argument('--use-timestamp', default=False, action='store_true', help='if used, will save data in a folder named after the current time (or whatever is provided in --timestamp)')
        parser.add_argument('--timestamp', default=None, help='the timestamp of a previous experiment to plot or observe parameters')
        
        # parse arguments (passing directly because initial parser will remove the "--experiment" argument)
        self.args = parser.parse_args(args=args)

        # manage device
        if self.args.device is None:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # do checks
        if self.args.use_timestamp and self.args.justplot:
            assert self.args.timestamp is not None, "if use_timestamp=True and plotting stored results, must provide a timestamp"
    
    @abstractmethod
    def make_args(self, parser) -> ArgumentParser:
        """
        Required method for defining special-case arguments.

        This should just use the add_argument method on the parser provided as input.
        """
        pass

    def get_prms_path(self):
        """Method for loading path to experiment parameters file"""
        return self.get_dir() / 'prms.pth'
    
    def get_results_path(self):
        """Method for loading path to experiment results files"""
        return self.get_dir() / 'results.pth'
    
    def get_network_path(self, name):
        """Method for loading path to saved network file"""
        return self.get_dir() / f"{name}.pt"
    
    def get_checkpoint_path(self):
        '''Method for loading path to network checkpoint file'''
        return self.get_dir() / 'checkpoint.tar'

    def _update_args(self, prms):
        """Method for updating arguments from saved parameter dictionary"""
        # First check if saved parameters contain unknown keys
        if prms.keys() > vars(self.args).keys():
            raise ValueError(f"Saved parameters contain keys not found in ArgumentParser:  {set(prms.keys()).difference(vars(self.args).keys())}")
        
        # Then update self.args while ignoring any meta arguments
        for ak in vars(self.args):
            if ak in self.meta_args: continue # don't update meta arguments
            if ak in prms and prms[ak] != vars(self.args)[ak]:
                print(f"Requested argument {ak}={vars(self.args)[ak]} differs from saved, which is: {ak}={prms[ak]}. Using saved...")
                setattr(self.args, ak, prms[ak])

    def save_experiment(self, results):
        """Method for saving experiment parameters and results to file"""
        # Save experiment parameters
        torch.save(vars(self.args), self.get_prms_path())
        # Save experiment results 
        torch.save(results, self.get_results_path())

    def load_experiment(self, no_results=False):
        """Method for loading saved experiment parameters and results"""
        # Check if prms path is there
        if not self.get_prms_path().exists():
            raise ValueError(f"saved parameters at: f{self.get_prms_path()} not found!")
        
        # Check if results directory is there
        if not self.get_results_path().exists():
            raise ValueError(f"saved results at: f{self.get_results_path()} not found!")

        # Load parameters into object
        prms = torch.load(self.get_prms_path())
        self._update_args(prms)
        
        # Don't load results if requested
        if no_results: return None

        # Load and return results
        return torch.load(self.get_results_path())
    
    def save_networks(self, nets, id=None):
        """
        Method for saving any networks that were trained
        
        Names networks with index in list of **nets**
        If **id** is provided, will use id in addition to the index
        """
        name = f"net_{id}_" if id is not None else "net_"
        for idx, net in enumerate(nets):
            cname = name + f"{idx}"
            torch.save(net, self.get_network_path(cname))

    @abstractmethod
    def main(self) -> Tuple[Dict, List[torch.nn.Module]]:
        """
        Required method for operating main experiment functions. 
        
        This method should perform any core training and analyses related to the experiment
        and return a results dictionary and a list of pytorch nn.Modules. The second requirement
        (torch modules) can probably be relaxed, but doesn't need to yet so let's keep it as is
        for overall clarity.
        """
        pass

    @abstractmethod
    def plot(self, results: Dict) -> None:
        """
        Required method for operating main plotting functions.
        
        Should accept as input a results dictionary and run plotting functions.
        If any plots are to be saved, then each plotting function must do so 
        accordingly. 
        """
        pass

    # -- support for main processing loop --
    def setup_ddp(self):
        
        """Set parameters for DDP management and initialize process"""
        os.environ['MASTER_PORT'] = os.environ['HOST_PORT'] # self.find_available_port(29500, 30000)
        os.environ['MASTER_ADDR'] = os.environ['HOST_ADDR']
        host_ip = os.environ['MASTER_ADDR']
        port_id = os.environ['MASTER_PORT']
        print(f'host address = {os.environ["HOST_ADDR"]}')
        print(f'master address = {os.environ["MASTER_ADDR"]}')
        # DDP setting: Turn on distributed if multiple GPUs in environment.
        if "WORLD_SIZE" in os.environ:
            self.args.world_size = int(os.environ["WORLD_SIZE"])
            print('world size from env: ', self.args.world_size)
        self.args.distributed = self.args.world_size > 1
        ngpus_per_node = torch.cuda.device_count()
        print(f'{ngpus_per_node=}')
        self.args.batch_size = int(self.args.batch_size / ngpus_per_node)
        print(f'adjusted batch size={self.args.batch_size}')
        if self.args.distributed:
            #if 'LOCAL_RANK' in os.environ: # from torchrun
            #    self.args.local_rank = int(os.environ['LOCAL_RANK'])
            #    self.args.rank = int(os.environ['RANK'])
            #    self.args.device = self.args.local_rank
            #elif self.args.local_rank != -1: # for torch.distributed.launch
            #    self.args.rank = self.args.local_rank
            #    self.args.device = self.args.local_rank
            if 'SLURM_PROCID' in os.environ: # for slurm scheduler
           #     print(f'env rank: {os.environ["RANK"]}')
                print(f"slurm procid: {os.environ['SLURM_PROCID']}")
                self.args.rank = int(os.environ['SLURM_PROCID'])
                self.args.local_rank = self.args.rank - ngpus_per_node * (self.args.rank // ngpus_per_node)
                self.args.device = self.args.rank % ngpus_per_node
            print(f'rank = {self.args.rank}')
            print(f'device = {self.args.device}')
            print(f'local_rank = {self.args.local_rank}')
            self.device = f'cuda:{self.args.device}' # update device for moddel and dataset
            dist.init_process_group(backend=self.args.dist_backend,
                                    world_size=self.args.world_size, rank=self.args.rank)
            print(f'created process for {dist.get_rank()}')
            dist.barrier()

    def prepare_dataset(self, transform_parameters):
        """simple method for getting dataset """   
        return get_dataset(self.args.dataset,
                           build=True,
                           transform_parameters=transform_parameters,
                           loader_parameters={'batch_size': self.args.batch_size},
                           device=self.args.device,
                           ddp_parameters={'world_size': self.args.world_size,
                                           'rank': self.args.rank})

    def plot_ready(self, name):
        """standard method for saving and showing plot when it's ready"""
        # if saving, then save the plot
        if not self.args.nosave:
            plt.savefig(str(self.get_path(name)))
        if self.run is not None:
            self.run.log({name: wandb.Image(plt)})
        # show the plot now if not doing showall
        if not self.args.showall:
            plt.show()


    
