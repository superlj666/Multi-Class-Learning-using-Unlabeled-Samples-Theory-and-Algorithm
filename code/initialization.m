addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng('default');

datasets = {
'iris', ...
'wine', ...
'glass', ...
'svmguide2', ...
'vowel', ...
'vehicle', ...
'dna', ...
'segment', ...
'satimage', ...
'pendigits', ...
'letter', ...
'poker', ...
'shuttle', ...
'usps', ...
'protein', ...
'Sensorless', ...
};

model.n_folds = 5;
model.n_repeats = 3;
model.rate_test = 0.3;
model.rate_labeled = 0.1;
model.n_batch = 1;
model.T = 10;
model.can_tau_I = [10.^(-9:-7), 0];
model.can_tau_A = 10.^-(7:2:11);
model.can_tau_S = [10.^-(3:2:7), 0];
model.can_step = 10.^(3:4);
