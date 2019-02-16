addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rand('state', 0);

model.n_folds = 5;
model.n_repeats = 1;
model.rate_labeled = 0.2;
model.data_name = 'dna';
model.n_batch = 32;
% model.can_tau_I = [2.^(-11:-3), 0];
% model.can_tau_A = 2.^(-5:-3);
% model.can_tau_S = [2.^(-10:-3), 0];
model.can_tau_I = 2.^(-20:-2);
model.can_tau_A = 2.^(-5);
model.can_tau_S = 2.^(-7);

% load datasets
[X, y] = load_data(model.data_name);    
L = construct_laplacian_graph(model.data_name, X, 10);

model_lrc_ssl = model;
model_ssl = model;
model_lrc = model;
model_linear = model;
for i_repeat = 1 : model.n_repeats
    idx_rand = randperm(numel(y));
    % take use of Laplacian matrix
    idx_train = idx_rand(1:ceil(0.7 * numel(y)));
    idx_test = setdiff(idx_rand, idx_train);

    idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
    X_train = X(idx_labeled, :); 
    y_train = y(idx_labeled);
    X_test = X(idx_test, :);
    y_test = y(idx_test);

    % choose parameters by cross validation
    errors_validate = cross_validation(L, X(idx_train, :), y(idx_train), model);
    
    % define models
    model_lrc_ssl = learner_lrc_ssl(errors_validate, model_lrc_ssl);
    model_ssl = learner_ssl(errors_validate, model_ssl);
    model_lrc = learner_lrc(errors_validate, model_lrc);
    model_linear = learner_linear(errors_validate, model_linear);

    % record training and testing    
    XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
    XLX = min(1, 1 / (sqrt(model_lrc_ssl.tau_I) * norm(XLX,'fro'))) * XLX;
    model_lrc_ssl = ps3vt_multi_train(XLX, X_train, y_train, model_lrc_ssl);
    model_lrc_ssl = record_batch(XLX, X_test, y_test, model_lrc_ssl, 'test');
    model_lrc_ssl.iter_batch = 0;
    model_lrc_ssl.time_train = 0;
    model_lrc_ssl.epoch = 0;

    XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
    XLX = min(1, 1 / (sqrt(model_ssl.tau_I) * norm(XLX,'fro'))) * XLX;
    model_ssl = ps3vt_multi_train(XLX, X_train, y_train, model_ssl);
    model_ssl = record_batch(XLX, X_test, y_test, model_ssl, 'test');
    model_ssl.iter_batch = 0;
    model_ssl.time_train = 0;
    model_ssl.epoch = 0;

    XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
    XLX = min(1, 1 / (sqrt(model_lrc.tau_I) * norm(XLX,'fro'))) * XLX;
    model_lrc = ps3vt_multi_train(XLX, X_train, y_train, model_lrc);
    model_lrc = record_batch(XLX, X_test, y_test, model_lrc, 'test');
    model_lrc.iter_batch = 0;
    model_lrc.time_train = 0;
    model_lrc.epoch = 0;

    XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
    XLX = min(1, 1 / (sqrt(model_linear.tau_I) * norm(XLX,'fro'))) * XLX;
    model_linear = ps3vt_multi_train(XLX, X_train, y_train, model_linear);
    model_linear = record_batch(XLX, X_test, y_test, model_linear, 'test');
    model_linear.iter_batch = 0;
    model_linear.time_train = 0;
    model_linear.epoch = 0;
    fprintf('Round: %.0f/%.0f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f\n', ...
    i_repeat, model.n_repeats, model_lrc_ssl.tau_I, model_lrc_ssl.tau_A, model_lrc_ssl.tau_S);
end

fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\n', ... 
    model_lrc_ssl.data_name, 'model_lrc_ssl', mean(model_lrc_ssl.test_err), std(model_lrc_ssl.test_err));
save(['../data/', model_lrc_ssl.data_name, '/', 'model_lrc_ssl','.mat']);

fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\n', ... 
    model_ssl.data_name, 'model_ssl', mean(model_ssl.test_err), std(model_ssl.test_err));
save(['../data/', model_ssl.data_name, '/', 'model_ssl','.mat']);

fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\n', ... 
    model_lrc.data_name, 'model_lrc', mean(model_lrc.test_err), std(model_lrc.test_err));
save(['../data/', model_lrc.data_name, '/', 'model_lrc','.mat']);

fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\n', ... 
    model_linear.data_name, 'model_linear', mean(model_linear.test_err), std(model_linear.test_err));
save(['../data/', model_linear.data_name, '/', 'model_linear','.mat']);

error_result = [model_lrc_ssl.test_err; model_ssl.test_err; ];

% % cross validation to choose parameters
% if exist(['../data/', model.data_name, '/', 'cross_validation.mat'], 'file')
%     load(['../data/', model.data_name, '/', 'cross_validation.mat'], 'errors_validate');
% else
%     errors_validate = cross_validation(L, X, y, model);
% end


%% Choose parameters for every methods
% model_linear = learner_linear(errors_validate, model);
% model_lrc = learner_lrc(errors_validate, model);
% model_ssl = learner_ssl(errors_validate, model);
% model_lrc_ssl = learner_lrc_ssl(errors_validate, model);

%% Choose parameters for our method
% model_lrc_ssl = learner_lrc_ssl(errors_validate, model);
% model_ssl = model_lrc_ssl; model_ssl.tau_S = 0;
% model_lrc = model_lrc_ssl; model_lrc.tau_I = 0;
% model_linear = model_lrc; model_linear.tau_S = 0;
% repeat_test(model_linear, 'linear', X, y, L);
% repeat_test(model_lrc, 'lrc', X, y, L);
% repeat_test(model_ssl, 'ssl', X, y, L);
% repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);

% model.tau_I = 2^-10;
% model.tau_A = 2^-4;
% model.tau_S = 2^-9;
% % for i_star = 5 : 10
% %     model.step = 2^i_star;
% %     repeat_test(model, 'test_0216', X, y, L);
% % end
% model = single_test(model, X, y, L);

