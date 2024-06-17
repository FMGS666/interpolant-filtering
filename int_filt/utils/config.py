"""
File containing the namespace definition for the CLI tool
"""

from datetime import datetime
import argparse

class OPTIONS:
    interpolant_method = ["pffp_v0", "pffp_v1"]
    backbone = ["mlp"]
    b_net_activation = ["relu"]
    experiment = ["nlg", "ou"]
    non_linearity = ["exp", "sin", "cos", "tan"]
    optimizer = ["adam", "adam-w"]
    scheduler = ["cosine-annealing", "none"]
    device = ["cuda", "cpu"]
    preprocessing = ["none", "sim"]

def configuration(args = None):
    ## default output directory
    run_id = datetime.now().strftime("%Y-%m-%d")+"/run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = "./log/out/" + run_id
    dump_dir = "./log/exp/" + run_id
    parser = argparse.ArgumentParser(description="CDTSurrSI")
    ## interpolant options
    interpolant_group = parser.add_argument_group("Interpolant Options")
    interpolant_group.add_argument("--interpolant-method", "-mth", default = "pffp_v0", help = "The Interpolant method used", choices = OPTIONS.interpolant_method)
    interpolant_group.add_argument("--epsilon", "-e", default = 1.0, type = float, help = "The Interpolant method used")
    ## mc sampling options
    mc_group = parser.add_argument_group("MC Integration Options")
    mc_group.add_argument("--num-mc-samples", "-nmc", default = 100, type = int, help = "The number of samples for MC integration")
    ## nn options
    nn_group = parser.add_argument_group("NN Options")
    nn_group.add_argument("--backbone", "-bkbn", default = "mlp", help = "The backbone for the neural network", choices = OPTIONS.backbone)
    nn_group.add_argument("--b-net-hidden-dims", "-bhd",  nargs="+", type=int, default = [16], help = "List of integers containing the number of hidden neurons for each layer of the $b$ model")
    nn_group.add_argument("--b-net-activation", "-bact", default = "relu", help = "The activation of the hidden layers of the $b$ model", choices = OPTIONS.b_net_activation)
    nn_group.add_argument("--b-net-activate-final", "-bactfin", action = "store_true", help = "Whether to activate the last layer for the $b$ model")
    nn_group.add_argument("--b-net-amortized","-bamrt", action = "store_true", help = "Whether to perform amortized learning by concatenating the observation to the input for the $b$ model")
    ## experiment options
    experiment_group = parser.add_argument_group("Experiment Options")
    experiment_group.add_argument("--experiment", "-exp", default = "nlg", help = "The experiment to be run", choices = OPTIONS.experiment)
    ## ssm experiment options
    ssm_group = parser.add_argument_group("SSM Options")
    ssm_group.add_argument("--sigma-x", "-sx", default = 1.0, type = float, help = "The standard deviation of the latent states for the OU model")
    ssm_group.add_argument("--sigma-y", "-sy", default = 1.0, type = float, help = "The standard deviation of the observation model")
    ssm_group.add_argument("--beta", "-b", default = 1.0, type = float, help = "Multiplier for the standard deviation of the latent states")
    ssm_group.add_argument("--num-dims", "-nd", default = 2, type = int, help = "The dimensionality of the space")
    ssm_group.add_argument("--num-sims", "-nsm", default = 1_000, type = int, help = "The number of independent simulations to be ran in each batch")
    ssm_group.add_argument("--num-iters", "-nit", default = 100_000, type = int, help = "The number of iterations to run each simulation for")
    ssm_group.add_argument("--non-linearity", "-nl", default = "sin", help = "The non-linearity to apply for the gaussian model", choices = OPTIONS.non_linearity)
    ssm_group.add_argument("--step-size", "-sz", default = 0.01, type = float, help = "The step size for the non-linearity of gaussian model")
    ## logging options
    logging_group = parser.add_argument_group("Logging Options")
    logging_group.add_argument("--log-results", "-nlg", action = "store_true", help = "Whether to log results to target directories")
    logging_group.add_argument("--logging-step", "-ls", default = 1, type = int, help = "The interval for logging")
    logging_group.add_argument("--log_dir", "-ld", type=str, default = out_dir,
                        help="Where to store the outputs of this simulation.")
    logging_group.add_argument("--dump_dir", "-dd", type=str, default = dump_dir,
                        help="Where to store the saved models.")
    ## b-net training options
    train_b_net_group = parser.add_argument_group("Training Options")
    train_b_net_group.add_argument("--b-net-num-grad-steps", "-bngs", default = 250, type = int, help = "The number of gradient steps to be taken during the training of the $b$ model")
    train_b_net_group.add_argument("--b-net-optimizer", "-bopt", default = "adam-w", help = "The optimizer used for training the $b$ model", choices = OPTIONS.optimizer)
    train_b_net_group.add_argument("--b-net-scheduler", "-bsched", default = "none", help = "The learning rate scheduler for training the $b$ model", choices = OPTIONS.scheduler)
    train_b_net_group.add_argument("--b-net-lr", "-blr", default = 0.001, type = float, help = "The initial learning rate used for training the $b$ model")
    ## reproducibility options
    reproducibility_group = parser.add_argument_group("Reproducibility Options")
    reproducibility_group.add_argument("--random-seed", "-rs", default = 128, type = int, help = "The random seed for the experiment")
    ## device options
    device_group = parser.add_argument_group("Device Options")
    device_group.add_argument("--device", "-d", default = "cuda", help = "The device to run the computation on", choices = OPTIONS.device)
    ## preprocessing options
    preprocessing_group = parser.add_argument_group("Preprocessing Options")
    device_group.add_argument("--preprocessing", "-pp", default = "sim", help = "The preprocessing method to be used", choices = OPTIONS.preprocessing)
    ## sampling options
    sampling_group = parser.add_argument_group("Sampling Options")
    sampling_group.add_argument("--num-samples", "-ns", default = 500, type = int, help = "The number of samples to be drawn from the learned distribution")
    sampling_group.add_argument("--num-time-steps", "-nts", default = 100, type = int, help = "The number of time steps in the discretization of the SDE")
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args = args)