# Online-Sparse-and-Non-Negative-Tensor-decomposition
# Sparse Non-Negative CPD (SNNCPD)

## Description

This repository contains MATLAB code for performing Sparse Non-Negative Canonical Polyadic Decomposition (SNNCPD) of 3D tensors using dictionary learning and the NADAM optimizer. The method is particularly useful for decomposing fluorescence tensors and other similar datasets where non-negativity and sparsity are desired.

## Features

- **Sparse Non-Negative CPD**: Efficient decomposition of 3D tensors with non-negativity and sparsity constraints.
- **NADAM Optimizer**: Uses the NADAM optimization algorithm, a variant of ADAM with Nesterov momentum.
- **Automatic Rank Selection**: The algorithm can overestimate the rank and apply L1 regularization to select the best rank.
- **Customizable Parameters**: Easily adjustable parameters for learning rate, regularization, and non-negativity constraints.

## Requirements

- MATLAB R2016b or later
- Signal Processing Toolbox (for certain functions)
- Statistics and Machine Learning Toolbox (optional)

## Installation

1. Clone the repository to your local machine

2. Add the cloned repository to your MATLAB path:
    ```matlab
    addpath(genpath('path_to_your_cloned_repository'));
    ```

## Usage

### 1. Preparing the Data

Load your 3D tensor data into MATLAB. The tensor should be of size `I x J x K`, where `I`, `J`, and `K` represent the dimensions of the data.

### 2. Running the Decomposition

Use the `SNNCPD_gradientNadam` function to perform the decomposition:

```matlab
% Load your tensor data
load('your_tensor_data.mat'); % Assume the tensor is stored in a variable called X

% Set the desired rank
R = 7; % Example rank

% Set the options for SNNCPD
options = createOptions(1, 1, 1.5, 0.9, 0.9, 1e-3);

% Perform the decomposition
[A, B, C, Da, Db, Va, Vb, hist, MU] = SNNCPD_gradientNadam(X, R, options);
displaycomponent(A, B, 4); % Display the first 4 components

% Define a grid of parameters 
param_grid.step = [1e-1, 1e-2, 1e-3];
param_grid.beta1 = [0.9, 0.95];
param_grid.beta2 = [0.9, 0.99];
param_grid.alpha = [0.1, 0.5, 1.0];

% Run the optimization to find the best parameters
[best_A, best_B, best_C, best_params, best_hist] = optimize_SNNCPD(X, R, param_grid);

% The best parameters found
disp(best_params);

## References
@article{sanou2024online,
  title={Online Canonical Polyadic Decomposition: Application of Fluorescence Tensors with Nonnegative Orthogonality and Sparse Constraint},
  author={Sanou, Isaac Wilfried and Luciani, Xavier and Redon, Roland and Mounier, St{\'e}phane},
  journal={Optimization Algorithms-Classics and Recent Advances},
  year={2024},
  publisher={IntechOpen}
}

@article{sanou2022online,
  title={Online nonnegative and sparse canonical polyadic decomposition of fluorescence tensors},
  author={Sanou, Isaac Wilfried and Redon, Roland and Luciani, Xavier and Mounier, St{\'e}phane},
  journal={Chemometrics and Intelligent Laboratory Systems},
  volume={225},
  pages={104550},
  year={2022},
  publisher={Elsevier}
}


@inproceedings{sanou2021online,
  title={Online nonnegative canonical polyadic decomposition: Algorithms and application},
  author={Sanou, Isaac Wilfried and Redon, Roland and Luciani, Xavier and Mounier, St{\'e}phane},
  booktitle={2021 29th European Signal Processing Conference (EUSIPCO)},
  pages={1805--1809},
  year={2021},
  organization={IEEE}
}



