addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng('default');

datasets = {
'iris', ...
'wine', ...
'glass', ...
'svmguide4', ...
'svmguide2', ...
'vowel', ...
'vehicle', ...
%'dna', ...
%'segment', ...
%'satimage', ...
%'usps', ...
%'pendigits', ...
%'letter', ...
%'protein', ...
%'poker', ...
%'shuttle', ...
%'Sensorless', ...
%'mnist', ...
};

for i_dataset = 1 : size(datasets, 2)
    rng('default');
    fprintf('cv for %s\n', char(datasets{i_dataset}));
    model.n_folds = 5;
    model.n_repeats = 30;
    model.rate_test = 0.3;
    model.rate_labeled = 0.3;
    model.data_name = char(datasets{i_dataset});
    model.n_batch = 32;
    model.T = 50;
    model.can_tau_I = [2 .^ -(7:2:11), 0];
    model.can_tau_A = 2 .^ -(3:4);
    model.can_tau_S = [2 .^ -(5:2:9), 0];
    model.can_step = 2 .^ (3.5:0.5:4.5);

    % load datasets
    [X, y] = load_data(model.data_name);    
    L = construct_laplacian_graph(model.data_name, X, 10);

    % cross validation to choose parameters
    errors_validate = cross_validation(L, X, y, model);
end

function model = learner_lrc_ssl_single(errors_validate, model)
    cv_results = reshape([errors_validate{:, 1}], [numel(model.can_step), numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)]);
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3, d4] = ind2sub([numel(model.can_step), numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)], loc_best);

    model.tau_I = model.can_tau_I(d4);
    model.tau_A = model.can_tau_A(d3);
    model.tau_S = model.can_tau_S(d2);
    model.step = model.can_step(d1);
end