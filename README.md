# interpolant-filtering

All the code lies in the `int_filt` module, which is structured in the following way

The `src` submodule contains the source code for:
* the implementation of the Neural Networks (`src.nn`)
* the implementation of Stochastic Interpolants and of the drift objective function (`src.si`)
* the implementation of the state space models to run the experiments on (`src.ssm`)

The `utils` submodule contains different utility functions used by the other modules

The `test` can be used for debugging

The `experiment` submodule contains the code for running the experiments

To be able to use the code, the required anaconda environment can be created by running:
```{bash}
$ conda env create -f environment.yml
$ conda activate interpolant-filtering
```

It is possible to run the OU experiment with default settings with the command:

```{bash}
$ python -m int_filt
```

To check for all the available arguments, you can run

```{bash}
$ python -m int_filt --help
```

To run the tests for debugging, run:

```{bash}
$ python -m int_filt.test
```

