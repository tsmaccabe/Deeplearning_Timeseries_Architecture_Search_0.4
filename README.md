## Deep Learning Timeseries Forecasting Hyperparameter Search

A program for tuning the architecture of a neural network. Uses Simulated Annealing or Grid Search to find ideal architecture hyperparameters for a given deep neural network type. Intended for time series forecasting. Built in Julia on `Flux.jl`.

(Includes data generation for the chaotic Lorenz system for testing.)


## Instructions

(For working examples, see `TEST_annealing()` within `Search.jl` and `lorenz_predict_test.jl`.)

1. Prepare data.
  - Define model input & output shapes. For single-channel data, these are `(num_nodes_in,)` and `(num_nodes_out,)`, and for multi-channel data, they are `(num_nodes_in, num_channels_in)` and `(num_nodes_out, num_channels_out)`.
  - Use `pair_data.jl` to create input-output pairs for forecasting.
  - Use `split_data.jl` to split it into a training set and a testing set.
     
3. Set architecture configuration in `config.ini`.
  - Select an architecture type key (`:fixed`, `:encoder`, or `:decoder`).
  - For each other config variable (eg, the length of the network, or the activation function), choose multiple possible values over which to search. Write them in vector format with square brackets, as shown below.
  ```
  # Architecture Settings
  #   Legend
  #       Architecture Type | (:fixed, <length>, <width>),
  #                           (:encoder, <length>),
  #                           (:decoder, <length>)
  #       Activations       | :leakyrelu, :relu, :sigmoid, :tanh
  ```
  - Eg:
  ```
  architecture_type = (:fixed, [5, 10, 15], [5, 10, 15])
  activation = [:relu, :sigmoid]
  ```

4. Set the search settings in `config.ini` to define the behavior of the hyperparameter search algorithm (simulated annealing).
  - Choose one value per variable.
  ```
  # Search Settings
  #   Legend
  #       Stop Conditions   | (:max_epochs, <max_epochs>),
  #                           (:epochs_stagnant, <epochs_stagnant>)
  #       Cooling Schedule  | (:exp, <init_temp>, <final_temp>, <final_temp_epoch>),
  #                           (:linear, <init_temp>, <final_temp>, <final_temp_epoch>)
  #       Shift Settings    | (:exp, <radial_scale>),
  #                           (:uniform, <radial_scale>)
  #       Trials Per State  | <trials_per_state>
  ```
  - Eg:
  ```
  stop_condition_search = (:max_epochs, 100)
  cool_schedule = (:exp, 0.5, 0.05, 100)
  shift_settings = (:exp, 10)
  trials_per_state = 100
  ```
  
5. Set the training settings in `config.ini`, which are used to execute training for each iteration of the architecture hyperparameter search.
  ```
  # Training Settings
  #   Legend
  #       Stop Conditions   | (:max_epochs, <max_epochs>),
  #                           (:epochs_stagnant, <epochs_stagnant>)
  #       Learning Rate     | (:constant, <learn_rate>),
  #                           (:exp, <init_rate>, <final_rate>, <final_rate_epoch>),
  #                           (:linear, <init_rate>, <final_rate>, <final_rate_epoch>)
  #       Loss Functions    | :mae, :mse, :cross_entropy
  #       Dropout           | <dropout_rate>
  #       Regularization    | :none,
  #                           (:l1, <reg_scale>),
  #                           (:l2, <reg_scale>)
  ```
  - Eg:
  ```
  stop_condition_train = (:max_epochs, 100)
  loss_function = :mae
  learn_rate = (:exp, 0.01, 0.002, 100)
  dropout_rate = 0.2
  regularization = (:l1, 0.1)
  ```

6. Run `init_env.jl` to import the configuration variables.

7. Use your input & output data settings and the configuration variables to define the following objects:
   - `RegularizationFunction`
   - 'LearnRateFunction'
	train_stop = TrainStopFunction(:epochs_stagnant, 25)
