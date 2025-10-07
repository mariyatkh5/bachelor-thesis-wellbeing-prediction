# config.py
from __future__ import annotations
from ConfigSpace import ConfigurationSpace, Categorical, Float, Integer
from ConfigSpace.conditions import EqualsCondition


def get_cnn_config_space(seed: int | None = None) -> ConfigurationSpace:
    """Define the hyperparameter search space for CNN models."""
    cs = ConfigurationSpace(name="CNN", seed=seed)

    cs.add_hyperparameters([
        Categorical("num_filters", [64, 128]),           # filters per Conv layer
        Integer("num_layers", (3, 8)),                   # number of Conv blocks
        Categorical("kernel_size", [7, 9, 11, 13]),      # must be odd
        Categorical("pooling", ["max", "average"]),      # pooling type
        Integer("pooling_size", (2, 4)),                 # pooling window size
        Float("dropout_rate", (0.0, 0.4)),               # dropout regularization
        Float("learning_rate", (1e-5, 1e-3), log=True),  # AdamW learning rate
        Float("weight_decay", (1e-5, 1e-2), log=True),   # AdamW weight decay
        Categorical("activation", ["relu", "gelu", "elu", "tanh"]),  # activation function
    ])
    return cs


def get_lstm_config_space(seed: int | None = None) -> ConfigurationSpace:
    """Define the hyperparameter search space for LSTM models."""
    cs = ConfigurationSpace(name="LSTM", seed=seed)

    hidden_size   = Categorical("hidden_size", [32, 64, 128])  # LSTM hidden units
    num_layers    = Integer("num_layers", (1, 2))              # number of LSTM layers
    bidirectional = Categorical("bidirectional", [True, False])
    dropout_rate  = Float("dropout_rate", (0.0, 0.5))
    learning_rate = Float("learning_rate", (1e-5, 1e-3), log=True)

    wd_choice     = Categorical("weight_decay_choice", ["zero", "log"])
    weight_decay  = Float("weight_decay", (1e-5, 1e-2), log=True)

    cs.add_hyperparameters([
        hidden_size, num_layers, bidirectional,
        dropout_rate, learning_rate, wd_choice, weight_decay
    ])
    cs.add_condition(EqualsCondition(weight_decay, wd_choice, "log"))  # only active if chosen
    return cs
