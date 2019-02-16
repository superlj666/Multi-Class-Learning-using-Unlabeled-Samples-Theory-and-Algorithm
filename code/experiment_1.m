addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rand('state', 0);

model.n_folds = 5;
model.rate_train = 0.8;
model.rate_labeled = 0.2;
model.data_name = 'dna';
model.n_batch = 32;
model.can_tau_I = [2.^(-11:-9), 0];%[2.^(-11:-3), 0];
model.can_tau_A = 2.^(-3);
model.can_tau_S = [2.^(-6), 0];%[2.^(-10:-3), 0];

% model.step = 100;
% model_train(model);

model.n_repeats = 10;

% load datasets
[X, y] = load_data(model.data_name);    
L = construct_laplacian_graph(model.data_name, X, 10);

% cross validation to choose parameters
if exist(['../data/', model.data_name, '/', 'cross_validation.mat'], 'file')
    load(['../data/', model.data_name, '/', 'cross_validation.mat'], 'errors_validate');
else
    errors_validate = cross_validation(L, X, y, model);
end

%% Choose parameters for every methods
model_linear = learner_linear(errors_validate, model);
model_lrc = learner_lrc(errors_validate, model);
model_ssl = learner_ssl(errors_validate, model);
model_lrc_ssl = learner_lrc_ssl(errors_validate, model);

%% Choose parameters for our method
% model_lrc_ssl = learner_lrc_ssl(errors_validate, model);
% model_ssl = model_lrc_ssl; model_ssl.tau_S = 0;
% model_lrc = model_lrc_ssl; model_lrc.tau_I = 0;
% model_linear = model_lrc; model_linear.tau_S = 0;

repeat_test(model_linear, 'linear', X, y, L);
repeat_test(model_lrc, 'lrc', X, y, L);
repeat_test(model_ssl, 'ssl', X, y, L);
repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
% model.tau_I = 2^-10;
% model.tau_A = 2^-4;
% model.tau_S = 2^-9;
% % for i_star = 5 : 10
% %     model.step = 2^i_star;
% %     repeat_test(model, 'test_0216', X, y, L);
% % end
% model = single_test(model, X, y, L);

