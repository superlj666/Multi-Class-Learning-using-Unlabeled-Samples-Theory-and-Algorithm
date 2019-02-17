addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rng('default');

model.n_folds = 10;
model.n_repeats = 30;
model.rate_test = 0.2;
model.rate_labeled = 0.2;
model.data_name = 'segment';
model.n_batch = 32;
model.can_tau_I = 0;%10 .^ -(2:4);
model.can_tau_A = 2 .^ -(2:2:20);
model.can_tau_S = 0;%10 .^ -(2:4);
model.can_step = 2.^(0 :1: 4);

% load datasets
[X, y] = load_data(model.data_name);    
L = construct_laplacian_graph(model.data_name, X, 10);

errors_validate = cross_validation(L, X, y, model);
