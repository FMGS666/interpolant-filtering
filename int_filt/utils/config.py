"""
File containing the namespace definition for the CLI tool
"""

from datetime import datetime
import argparse

class OPTIONS:
    interpolant_method = ["pffp_v0", "pffp_v1"]
    backbone = ["mlp"]
    b_net_activation = ["relu"]
    experiment = ["nlg", "nlg-controlled"]
    non_linearity = ["exp", "sin", "cos", "tan", "rw", "affine"]
    optimizer = ["sgd", "adam", "adam-w"]
    scheduler = ["cosine-annealing", "none"]
    device = ["cuda", "cpu"]
    preprocessing = ["none", "sim"]
    observation_model = ["none", "gaussian"]
    gde = ["none", "interpolant-gde"]

def configuration(args = None):
    ## default output directory
    run_id = datetime.now().strftime("%Y-%m-%d")+"/run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = "./log/out/" + run_id
    dump_dir = "./log/exp/" + run_id
    parser = argparse.ArgumentParser(description="CDTSurrSI")
    ## interpolant options
    interpolant_group = parser.add_argument_group("Interpolant Options")
    interpolant_group.add_argument("--interpolant-method", "-mth", default = "pffp_v0", help = "The Interpolant method used", choices = OPTIONS.interpolant_method)
    interpolant_group.add_argument("--epsilon", "-e", default = 2e-2, type = float, help = "The Interpolant method used")
    ## mc sampling options
    mc_group = parser.add_argument_group("MC Integration Options")
    mc_group.add_argument("--num-mc-samples", "-nmc", default = 750, type = int, help = "The number of samples for MC integration")
    ## nn options
    nn_group = parser.add_argument_group("NN Options")
    nn_group.add_argument("--backbone", "-bkbn", default = "mlp", help = "The backbone for the neural network", choices = OPTIONS.backbone)
    nn_group.add_argument("--b-net-hidden-dims", "-bhd",  nargs="+", type=int, default = [64], help = "List of integers containing the number of hidden neurons for each layer of the $b$ model")
    nn_group.add_argument("--b-net-activation", "-bact", default = "relu", help = "The activation of the hidden layers of the $b$ model", choices = OPTIONS.b_net_activation)
    nn_group.add_argument("--b-net-activate-final", "-bactfin", action = "store_true", help = "Whether to activate the last layer for the $b$ model")
    nn_group.add_argument("--b-net-amortized","-bamrt", action = "store_true", help = "Whether to perform amortized learning by concatenating the observation to the input for the $b$ model")
    nn_group.add_argument("--c-net-hidden-dims", "-chd",  nargs="+", type=int, default = [64], help = "List of integers containing the number of hidden neurons for each layer of the $b$ model")
    nn_group.add_argument("--c-net-activation", "-cact", default = "relu", help = "The activation of the hidden layers of the $b$ model", choices = OPTIONS.b_net_activation)
    nn_group.add_argument("--c-net-activate-final", "-cactfin", action = "store_true", help = "Whether to activate the last layer for the $b$ model")
    ## experiment options
    experiment_group = parser.add_argument_group("Experiment Options")
    experiment_group.add_argument("--experiment", "-exp", default = "nlg", help = "The experiment to be run", choices = OPTIONS.experiment)
    ## ssm experiment options
    ssm_group = parser.add_argument_group("SSM Options")
    ssm_group.add_argument("--sigma-x", "-sx", default = 1e-2, type = float, help = "The standard deviation of the latent states for the OU model")
    ssm_group.add_argument("--sigma-y", "-sy", default = 1e-2, type = float, help = "The standard deviation of the observation model")
    ssm_group.add_argument("--beta", "-b", default = 1.0, type = float, help = "Multiplier for the standard deviation of the latent states")
    ssm_group.add_argument("--num-dims", "-nd", default = 1, type = int, help = "The dimensionality of the space")
    ssm_group.add_argument("--num-sims", "-nsm", default = 1_000, type = int, help = "The number of independent simulations to be ran in each batch")
    ssm_group.add_argument("--num-iters", "-nit", default = 1_000_000, type = int, help = "The number of iterations to run each simulation for")
    ssm_group.add_argument("--num-burn-in-steps", "-nbis", default = 0, type = int, help = "The number of iterations to run each simulation for")
    ssm_group.add_argument("--non-linearity", "-nl", default = "sin", help = "The non-linearity to apply for the gaussian model", choices = OPTIONS.non_linearity)
    ssm_group.add_argument("--step-size", "-sz", default = 1e-3, type = float, help = "The step size for the non-linearity of gaussian model")
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
    train_b_net_group.add_argument("--num-grad-steps", "-ngs", default = 400, type = int, help = "The number of gradient steps to be taken during training")
    train_b_net_group.add_argument("--b-net-optimizer", "-bopt", default = "adam-w", help = "The optimizer used for training the $b$ model", choices = OPTIONS.optimizer)
    train_b_net_group.add_argument("--b-net-scheduler", "-bsched", default = "none", help = "The learning rate scheduler for training the $b$ model", choices = OPTIONS.scheduler)
    train_b_net_group.add_argument("--b-net-lr", "-blr", default = 1e-3, type = float, help = "The initial learning rate used for training the $b$ model")
    train_b_net_group.add_argument("--b-net-amortized-optimizer", "-baopt", default = "adam-w", help = "The optimizer used for training the amortized $b$ model", choices = OPTIONS.optimizer)
    train_b_net_group.add_argument("--b-net-amortized-scheduler", "-basched", default = "none", help = "The learning rate scheduler for training the amortized $b$ model", choices = OPTIONS.scheduler)
    train_b_net_group.add_argument("--b-net-amortized-lr", "-balr", default = 1e-3, type = float, help = "The initial learning rate used for training the amortized $b$ model")
    train_b_net_group.add_argument("--c-net-optimizer", "-copt", default = "adam-w", help = "The optimizer used for training the $c$ model", choices = OPTIONS.optimizer)
    train_b_net_group.add_argument("--c-net-scheduler", "-csched", default = "none", help = "The learning rate scheduler for training the $c$ model", choices = OPTIONS.scheduler)
    train_b_net_group.add_argument("--c-net-lr", "-clr", default = 1e-3, type = float, help = "The initial learning rate used for training the $c$ model")
    ## observation model options
    observation_model_options = parser.add_argument_group("Observation Model Options")
    observation_model_options.add_argument("--observation-model", "-os", default = "gaussian", help = "The observation model to be used in the experiments", choices = OPTIONS.observation_model)
    ## gradient density estimation options
    gde_options = parser.add_argument_group("Gradient Density Estimation Options")
    gde_options.add_argument("--gde", "-gde", default = "interpolant-gde", help = "The Gradient Density Estimation method to be used in the experiments", choices = OPTIONS.gde)
    ## drift control options
    control_options = parser.add_argument_group("Additional Experiment Options")
    control_options.add_argument("--controlled", "-ctrl", action = "store_true", help = "Whether to explicitly model the control term in the experiments")
    ## reproducibility options
    reproducibility_group = parser.add_argument_group("Reproducibility Options")
    reproducibility_group.add_argument("--random-seed", "-rs", default = 128, type = int, help = "The random seed for the experiment")
    ## device options
    device_group = parser.add_argument_group("Device Options")
    device_group.add_argument("--device", "-d", default = "cuda", help = "The device to run the computation on", choices = OPTIONS.device)
    ## preprocessing options
    preprocessing_group = parser.add_argument_group("Preprocessing Options")
    device_group.add_argument("--preprocessing", "-pp", default = "sim", help = "The preprocessing method to be used", choices = OPTIONS.preprocessing)
    device_group.add_argument("--postprocessing", "-pstp", action = "store_true", help = "Whether to postprocess the data after the forward pass of the model")
    ## sampling options
    sampling_group = parser.add_argument_group("Sampling Options")
    sampling_group.add_argument("--num-samples", "-ns", default = 500, type = int, help = "The number of samples to be drawn from the learned distribution")
    sampling_group.add_argument("--num-time-steps", "-nts", default = 1000, type = int, help = "The number of time steps in the discretization of the SDE")
    sampling_group.add_argument("--num-ar-steps", "-nars", default = 1_000, type = int, help = "The number of autoregressive time steps to be taken during generation")
    sampling_group.add_argument("--initial-time-step", "-its", default = 0, type = int, help = "The time index at which to start the autoregressive sampling")
    sampling_group.add_argument("--ar-sample-train", "-arst", action = "store_true", help = "Whether to perform autoregressive sampling on the training dataset")
    sampling_group.add_argument("--full-out", "-fo", action = "store_true", help = "Whether to store the full output during sampling")
    ## memory management options
    memory_group = parser.add_argument_group("Memory Management Options")
    memory_group.add_argument("--clear-memory", "-cm", action = "store_true", help = "Whether to clear memory and cache after each forward call")
    if args is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args = args)