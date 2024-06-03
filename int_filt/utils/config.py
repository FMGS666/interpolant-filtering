"""
File containing the namespace definition for the CLI tool
"""

from datetime import datetime
import argparse


def configuration(args = None):
    ## default output directory
    run_id = datetime.now().strftime('%Y-%m-%d')+'/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out_dir = './out/' + run_id
    dump_dir = 'exp/' + run_id
    parser = argparse.ArgumentParser(description='CDTSurrSI')
    ## interpolant options
    interpolant_group = parser.add_argument_group("Interpolant Options")
    interpolant_group.add_argument("--interpolant-method", "-mth", default = "pffp_v0")
    ## mc sampling options
    mc_group = parser.add_argument_group("MC Integration Options")
    mc_group.add_argument("--num-samples", "-ns", default = 100, type = int)
    ## nn options
    nn_group = parser.add_argument_group("NN Options")
    nn_group.add_argument("--backbone", "-bkbn", default = "mlp")
    ## bnet options
    b_net_group = parser.add_argument_group("BNet Options")
    b_net_group.add_argument("--b-net-hidden-dims", "-bhd",  nargs="+", type=int, default = [16])
    b_net_group.add_argument("--b-net-activation", "-bact", default = "relu")
    b_net_group.add_argument("--b-net-activate-final", "-bactfin", action = "store_true")
    ## experiment options
    experiment_group = parser.add_argument_group("Experiment Options")
    experiment_group.add_argument("--experiment", "-exp", default = "ornstein-uhlenbeck")
    ## ou experiment options
    ou_group = parser.add_argument_group("Ornstein Uhlenbeck Experiment Options")
    ou_group.add_argument("--sigma-x", "-sx", default = 1.0, type = float)
    ou_group.add_argument("--sigma-y", "-sy", default = 1.0, type = float)
    ou_group.add_argument("--beta", "-b", default = 1.0, type = float)
    ou_group.add_argument("--num-dims", "-nd", default = 2, type = int)
    ou_group.add_argument("--num-sims", "-nsm", default = 10000, type = int)
    ## logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument('--log_dir', type=str, default = out_dir,
                        help='Where to store the outputs of this simulation.')
    logging_group.add_argument('--dump_dir', type=str, default = dump_dir,
                        help='Where to store the saved models.')
    ## b-net training options
    train_b_net_group = parser.add_argument_group("BNet Training Options")
    train_b_net_group.add_argument("--b-net-num-grad-steps", "-bngs", default = 2000, type = int)
    train_b_net_group.add_argument("--b-net-optimizer", "-bopt", default = "adam")
    train_b_net_group.add_argument("--b-net-scheduler", "-bsched", default = "none")
    train_b_net_group.add_argument("--b-net-lr", "-dlr", default = 0.01, type = float)
    ## reproducibility options
    reproducibility_group = parser.add_argument_group("Reproducibility Options")
    reproducibility_group.add_argument("--random-seed", "-rs", default = 128, type = int)
    ## gde options
    gde_group = parser.add_argument_group("Gradient Density Estimator Options")
    gde_group.add_argument("--gde", "-gde", default = "pffp")
    ## observation model options
    observation_group = parser.add_argument_group("Observation Model Options")
    observation_group.add_argument("--observation-model", "-os", default = "gaussian")
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args = args)
