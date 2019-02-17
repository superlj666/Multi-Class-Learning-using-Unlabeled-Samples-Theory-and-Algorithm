addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rng('default');

model.n_folds = 10;
model.n_repeats = 30;
model.rate_test = 0.2;
model.rate_labeled = 0.2;
model.data_name = 'dna';
model.n_batch = 32;
model.can_tau_I = 2 .^ -9;
model.can_tau_A = 2^-2;
model.can_tau_S = [2^-9, 0];
model.can_step = 2.^(2 : 5);

% load datasets
[X, y] = load_data(model.data_name);    
L = construct_laplacian_graph(model.data_name, X, 10);

errors_validate = cross_validation(L, X, y, model);
