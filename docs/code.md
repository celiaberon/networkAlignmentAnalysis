# Code Structure
This file describes the structure of the codebase (as it is currently designed on July 28th, 2024). The point is to explain the what and why of the code, and also how things work together. It ends with a discussion of wish-list items and ideas that could help with refactoring and/or could be useful later but aren't required right now. 

# Alignment Network
As described in the [background](./background.md) section, this repo is based on a statistical metric we call "alignment" which measures the relationship between the variance in the input to a layer and the weights of that layer. Because this is the key point of the project, I built an ``AlignmentNetwork`` class (found [here](../networkAlignmentAnalysis/models/base.py)) that is designed to wrap pytorch networks such that it is easy to perform all alignment related computations.

The ``AlignmentNetwork`` class is an abstract base class that is intended to be a parent of all models used in the project. It contains all the methods required for alignment computations, along with a forward pass method. 

### Initialization
Since Alignment is measured between the input to a layer and the layers weights, AlignmentNetworks are initialized with a constrained structure to make alignment measurements and all related computations straightforward. This is implemented by a method intended to be used during initialization called ``register_layer``. The main inputs to ``register_layer`` are 1) any pytorch module that has a forward method and 2) and some extra "metaparameters" that determine how the layer should be handled in alignment related computations. Every time ``register_layer`` is called, it adds the pytorch module to a list (``self.layers``) and the metaparamaters to a different list (``self.metaparameters``). 

All operations using the network, including the standard forward pass along with alignment computations are performed using the ``self.layers`` list. For example, the forward pass simply passes the input tensor `x` through each layer sequentially. <span style="background-color: #005c4e">Note: this isn't ideal because we'll want to use some networks that do operations in parallel (like transformers), so this will almost certainly need to be revisited.</span> To measure the alignment statistic for each layer, the method ``measure_alignment`` is provided. This works by getting the inputs to each layer with ``get_layer_inputs``, getting the weights to each layer with ``get_layer_weights``, preprocessing the inputs if necessary (like for convolutional layers), then measuring the alignment between inputs and weights with simple list comprehension. 

The forward pass has an optional kwarg ``store_hidden``, which is usually ``False`` but if set to ``True`` then the output of each layer is stored in the ``self.hidden`` attribute as a list. This is intended to facilitate fast computation where ``hidden`` is used as a cache. For example, for each mini-batch in a training loop, the forward method is required to measure the network output, but the user might also want to measure alignment. If they set ``store_hidden=True``, then they can use ``precomputed=True`` in the ``measure_alignment`` method to not redo the forward pass. <span style="background-color: #005c4e">Note: this will probably be replaced with a more intelligent hook structure... see below.</span>

### Example Initialization
This repo has 3 models at the moment (found [here](../networkAlignmentAnalysis/models/models.py)). I'll use the model called ``CNN2P2`` as an example since it's simple and contains both feedforward and convolutional layers. The following code snippet is an abbreviated version of the ``initialize`` method of the ``CNN2P2`` network with extra comments to explain what's going on. 

```python
def initialize(self):
    """architecture definition"""
    # create layers

    # each layer is a sequential module with at exactly one layer intended to be analyzed with alignment metrics
    # it is important that the input to the layer (whatever comes out of the previous layer) is the activations whose
    # variance structure is meant to be compared with the structure in the weights
    # note how the dropout is included before the linear layer, such that dropout doesn't artificially change 
    # the structure when using alignment computations
    layer1 = nn.Sequential(nn.Conv2d(), nn.ReLU(), nn.MaxPool2d(2))
    layer2 = nn.Sequential(nn.Conv2d(), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(start_dim=1))
    layer3 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_hidden[0], num_hidden[1]), nn.ReLU())
    layer4 = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(num_hidden[1], output_dim))

    # each layer is "registered" to the alignment network. 
    # the ``default_metaprms_####`` method builds metaparameters based on the layer type. The index (0, 0, 1, 1) indicates
    # which item in the sequential object has the relevant weight matrix. 
    self.register_layer(layer1, **default_metaprms_conv2d(0, flag=flag))
    self.register_layer(layer2, **default_metaprms_conv2d(0, flag=flag))
    self.register_layer(layer3, **default_metaprms_linear(1))
    self.register_layer(layer4, **default_metaprms_linear(1))
```

### Idea for improvement
Yoking the forward pass structure with the simple step-by-step forward method of each component registered by ``register_layer`` is a bit too constraining. I think there's a way we could improve this and make the network more flexible with hook points (this is partially based on the use of hook points in the package called [transformer_lens](https://github.com/TransformerLensOrg/TransformerLens)). Here's the idea: 

#### 1. Build a network the normal way
In the ``__init__`` method, just add layers as attributes to the ``nn.Module``. Then, write a custom forward pass method. 

#### 2. Register layers with pointers to the relevant model attributes
Register layers - but this new register method will just store a list of registered layers for lookup later, without constraining their usage in the forward pass like before. Example:
```python
self.layer1 = nn.Linear(100, 50)
self.layer2 = nn.Linear(50, 10)
self.register_layer(self.layer1)
self.register_layer(self.layer2)
```

#### 3. Perform forward with alignment measurement when requested
In the ``transformer_lens`` library, there is a second forward method called ``run_with_cache`` (found [here](https://github.com/TransformerLensOrg/TransformerLens/blob/main/transformer_lens/hook_points.py#L509)) that uses a context manager to create (and remove) hooks to the relevant locations, perform the forward pass and return the output, and store the output of intermediate activations (captured in the hooks) in a second output element. 

I think we can implement something almost identical here, where the ``run_with_cache`` method has options to retrieve intermediate activations, perform online alignment measurements, and potentially more. 

Details in ``transformer_lens`` library and in my head, I don't want to start implementing if you all have better ideas. 

### Other features of AlignmentNetworks
There are other methods in AlignmentNetworks that are used in some of the pipelines. Without much explanation I'll just mention some here. 

- ``_preprocess_inputs``: preprocess inputs using the details in ``self.metaparameters`` to prepare inputs for the alignment method (e.g. for convolutional layer inputs). 
- ``forward_targeted_dropout``: perform a forward pass with specific nodes set to 0 based on pre-computed indices. 
- ``measure_eigenfeatures``: perform PCA on input activity and also measure projection of weights onto PCs. 

### Final note on AlignmentNetworks
There's some extra engineering in the [models](../networkAlignmentAnalysis/models/) module to help with building networks. I'm not going to explain it all here because it's mostly just implementation details. Let me know if you want me to explain more. 

# Datasets
The [datasets](../networkAlignmentAnalysis/datasets.py) module is simple and is designed to facilitate easy integration with the rest of the repository. I won't spend much time because I think it's standard pytorch. 

There is an abstract base class called ``DataSet`` that is used as a parent for specific datasets (like MNIST). It builds datasets, dataloaders, and preps distributed sampling if required. 

One key feature of a dataset is a model/dataset specific data transformation. Since we use different network architectures on the same dataset, the dataset initialization receives an input called ``transform_parameters``, which can either be a dictionary of kwargs or an example of a model intended to be used with the dataset. ``AlignmentNetwork`` models are required to define a method called ``get_transform_parameters`` which accepts the dataset as input and returns the relevant data transformation. 

Each dataset that we've used so far is implemented at the bottom of the datasets module. 

### Note on dataloading:
There is a weird idiosyncracy of pytorch transforms where converting an RGB image to a grayscale doesn't work with their transforms. I tried very hard to get it to work and even submitted an issue and discussed it with pytorch engineers on Github. Still doesn't work. So, I wrote a special dataloading method called ``unwrap_batch``. It simply does a typical batch loading (with the pytorch dataloader), then performs an optional "extra_transform", which has only been implemented for the RGB $\rightarrow$ grayscale transform. It's integrated into the train loop and test loop. 

# Experiments
We have built an experiment module designed to facilitate pipelining and making the running and development of experiments easy. There are two modules called "experiment". One is a script that is in the top-level of the repo ([here](../experiment.py)). This is what is used to call and run all experiments from the command line. The second is a module that contains an experiment base class, supporting functions, and more modules containing the program for specific experiments ([here](../networkAlignmentAnalysis/experiments/)). 

### Experiment Script
The experiment script can do three things: 
1. Run a new experiment (train networks, do analyses, make plots, save, etc.)
2. Plot the results of an old experiment (load results, make plots, save, etc.)
3. Show the parameters used for a saved experiment

A specific experiment is requested with the first positional argument of the script. For example, to run the ``alignment_stats`` experiment, the user would run this command:
```bash
python experiment.py alignment_stats --extra_args <arg_values>
```

### Experiment Modules
Experiments are required to have two central methods for training and plotting called ``main()`` and ``plot()``. These are defined in the children of the [``Experiment``](../networkAlignmentAnalysis/experiments/experiment.py) base class. They are what are used when the experiment script is called- if the user wants to run an experiment, the ``main()`` method is performed. If the user wants to plot a new or saved experiment, the ``plot()`` method is called. 

#### Saving and directory management
The experiment module has a defined system for saving networks, analysis data, and plots. These are mostly straightforward, so I'll just mention the relevant methods: ``get_dir``, ``get_exp_path``, ``get_path``, ``get_basename``, ``prepare_path``, ``register_timestamp``. You can probably tell from the names that this is a bit clunky... it works so I haven't adjusted it much but I'm sure it could be improved. Configuration of WandB is also performed in the base class here. 

#### Experiment Arguments
The experiment module is where arguments for a command-line argument parser are defined. There are two kinds of arguments: "meta_args" and "args". 

Meta Arguments are general arguments that are used in every experiment. These determine whether to run a new experiment or just plot an old experiment, whether to save, whether to save networks in addition to plots/results, and which device to run the experiment on. These are defined in the ``get_args`` method of the experiment base class definition. 

Arguments are extended to the specific requirements of each experiment. These extensions are added in the experiment child classes. As an example, look at the ``make_args`` method of the alignment_stats experiment ([here](https://github.com/landoskape/networkAlignmentAnalysis/blob/dev/networkAlignmentAnalysis/experiments/alignment_stats.py#L16)). The experiment module provides standard arguments in the ``arglib.py`` file which are added to the parser there. 

#### Main experiment function
In general, the main experiment function will 1) create networks (using a ``create_networks`` method defined in each experiment child), 2) prepare a dataset (using ``prepare_dataset`` which is provided by the experiment base class), 3) train the networks (using a standard train loop described below), 4) perform auxiliary analyses, then 5) return a tuple of the results (as a dictionary) and the trained networks (as a list). 

#### Plotting function
The plotting function accepts as input the results dictionary which was generated by the ``main`` method and does whatever plotting is required. There is a special module called [``plotting.py``](../networkAlignmentAnalysis/plotting.py) that contains standard plotting methods which can be reused. 


# Train Loop
Lastly, we have coded a one-size-fits-all training loop. It's a bit busy, becuase it contains all the code related to training models, analyzing statistics during training, DDP handling, and probably more. I'll just cover the key points. (I think there's a chance this might be reworked a lot as we optimize). The train loop is it's own module (along with a similar test loop and some other supporting functions). It's found [here](../networkAlignmentAnalysis/train.py).

- The train loop is designed to accept a list of networks and optimizers, a dataset, and kwarg parameters. 
- Some of the parameters determine what analyses to run (for example, the ``measure_alignment = parameters.get("alignment", True)`` line). 
- All results are stored in a dictionary which facilitates checkpointing and is the output of the function (to interface witht he experiment module). 
- There is standard boiler plate code (load a batch, get the network outputs, do a backward pass and weight updates). 
- The alignment related analyses that we have developed are all here (sorry it's so busy!). Some quick examples: 
  - alignment measurements happen here
  - there are methods for measuring the changes in weights (and the alignment of the changes in weights)
  - there is some code for comparing the measured alignment distribution to the "expected" alignment distribution (related to a paper by Ila Fiete)
  - There is wandb code here
  - Checkpointing code is at the end of the train loop

### Possible improvement
I don't think the ``**parameters`` system for determining how to run the training loop is very good, it depends heavily on the user knowing the possible parameters, it is hard coded and can yield failures, and is opaque in the sense that you have to look through the whole ``train.train`` function to see what the possible parameters are. 

One possible method for improvement is to use a configuration management system. I haven't used one before, but I understand that good systems exist that are designed for machine learning experiments (like [Hydra](https://hydra.cc/docs/intro/)). 




