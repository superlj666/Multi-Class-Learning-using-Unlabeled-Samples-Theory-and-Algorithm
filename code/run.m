addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rand('state', 0);

model.n_folds = 10;
model.rate_labeled = 0.2;
model.data_name = 'dna';
model.n_batch = 32;
model.can_tau_I = [2.^(-11:-3), 0];
model.can_tau_A = 2.^(-5:-3);
model.can_tau_S = [2.^(-10:-3), 0];

% model_train(model);

model.n_repeats = 30;
experiment_1(model);