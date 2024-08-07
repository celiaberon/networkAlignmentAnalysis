from argparse import ArgumentParser

from .adversarial_shaping import AdversarialShaping
from .alignment_comparison import AlignmentComparison
from .alignment_early_stopping import AlignmentEarlyStopping
from .alignment_stats import AlignmentStatistics

EXPERIMENT_REGISTRY = {
    "alignment_stats": AlignmentStatistics,
    "alignment_comparison": AlignmentComparison,
    "adversarial_shaping": AdversarialShaping,
    "early_stopping": AlignmentEarlyStopping
}


def get_experiment(experiment_name, build=False, **kwargs):
    """
    lookup model constructor from model registry by name

    if build=True, builds experiment and returns an experiment object using any kwargs
    otherwise just returns the class constructor
    """
    if experiment_name not in EXPERIMENT_REGISTRY:
        raise ValueError(f"Experiment ({experiment_name}) is not in EXPERIMENT_REGISTRY")
    experiment = EXPERIMENT_REGISTRY[experiment_name]
    if build:
        return experiment(**kwargs)
    return experiment


def create_experiment():
    """
    method to create experiment using initial argument parser

    the argument parser looks for a known argument called "--experiment", and the resulting
    string is used to retrieve an experiment constructor from the EXPERIMENT_REGISTRY

    any remaining arguments (args) are passed to the experiment constructor which has it's
    own argument parser in the class definition (but doesn't define the --experiment argument
    which is why the remaining args need to be passed to it directly)

    note:
    add_help=False so adding a --help argument will show a help message for the specific experiment
    that is requested rather than showing a help message for this little parser then blocking the
    rest of the execution. It means using --help requires a valid 'experiment' positional argument.
    """
    parser = ArgumentParser(description=f"ArgumentParser for loading experiment constructor", add_help=False)
    parser.add_argument("experiment", type=str, help="a string that defines which experiment to run")
    exp_args, args = parser.parse_known_args()
    return get_experiment(exp_args.experiment, build=True, args=args)
