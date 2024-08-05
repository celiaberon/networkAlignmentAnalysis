# Distributed training

To scale our training, we've used Pytorch's Distributed Data Parallel (DDP). 

In terms of implementation and workflow, this involves:

## Initialization
Initializing processing within the `Experiment` base class with the `setup_ddp` method. This method checks environment variables to configure all processes via `torch.distributed.init_process_group` with NCCL, and also updates relevant attributes for downstream events (e.g. correctly configuring dataloaders for distributed training).

## Wrapping in DDP
After networks have been created within a specific experiment, they get wrapped into DDP instances. This forces referencing to any attribute written for the `AlignmentNetwork` class (required since not using standard nn.Module). Handler to reference any custom attribute using the form `net.module.[attribute]` defined in the base models module.


## Train loop
The main portion of implementation of course involves the training loop. The current approach aims to sync up measurements for each batch, most (perhaps all) of which is not truly necessary and was only motivated by preserving existing non-parallelized workflow and analysis (followed up below for potential simpler approach).
    - Loss and accuracy are just calculated in aggregate across all processes (averaged across sub-batches) with `all_reduce()` (this is all standard/easily implemented).
    - Alignment is a bit different because it's not a metric that can just be averaged across processes (one option is that it could be, but that will change the way we're thinking about the metric slightly). Instead, we use `all_gather` to combine each process's alignment measurements into a list on the main process, so that there are `world_size` replicates of alignment metrics per step. These become somewhat deep nested lists, such that there are `world_size` times the original alignment measurement, comprised of:
        ```[(num_nets, num_steps, num_neurons) for layer in alignment_layers]```
        - After gathering on the main process, alignment measurements are permuted such that replicate from different processes are grouped.
    - Delta weights does not need aggregation because each process contains the same model/weight update.

Using aggregations that send values to all processes rather than just main process (`all_reduce` and `all_gather`) to ensure function returns throughout experiment match across all processes.

Note: given the depth of some nested lists, we've provided some convenient functions in `utils.py` to view, reproduce, and manipulate list dimensions.

## Analysis on test set
1) Targeted dropout experiments, aggregate all dropout inferences passes across processes with `all_reduce()`. This is implemented as a true weighted average (manual implementation).
2) Measure eigenfeatures: since measuring eigenfeatures on the distributed dataloader, we need to `gather()` the features. There are a few assumptions baked into the current implementation about consistency across distributed batches that will likely not hold.
3) Eigenfeature dropout is handled similarly to targeted dropout, but here we need to take the appropriate eigenfeatures to match the dataset (therefore, we end up effectively backtracking some of the synchronization...so we may want to consider skipping the all_reduce and instead keep separate scories corresponding to each set of inputs)

## Possible improvement
There's a lot of room for improvement here. At the moment, there's frequent movement of data between devices that could likely be optimized if it's causing substantial delays. One consideration is whether metrics that depend directly on the input data (such as alignment and eigenfeatures), and therefore cannot be combined, should instead just be propagated fully on their process and treated as replicates at the end.

Another inefficiency comes from the need to reimplement some of the initial setup code (such as wrapping with DDP) in every specific type of experiment, which can likely be managed better.

## Usage notes

Has worked on the `alignment_stats` experiment for multple GPUs on a single node, but needs to be tested across multiple nodes. There may be some race conditions in the workflow that throw errors when initializing a new experiment with multiple nodes.

Haven't tested it yet with checkpointing (which can just be called with a flag), but there likely needs to be some bottlenecking up front to ensure everything executes properly.

To consider (scientifically): in using DDP, the effective batch size for training (weight updates) is the `batch_size` x `world_size`. However, alignment is just calculated on `batch_size` (wuch that we get `world_size` measurements of alignment per step). This influences the sampling rate of alignment across training and is constrained by the necessary batch_size to obtain accurate estimates of alignment.