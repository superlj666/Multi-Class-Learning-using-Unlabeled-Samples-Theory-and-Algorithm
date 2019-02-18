addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng(64);

datasets = {
'dna', ...
%'segment', ...
%'satimage', ...
% 'usps', ...
% 'pendigits', ...
% 'letter', ...
% 'protein', ...
% 'poker', ...
% 'shuttle', ...
% 'Sensorless', ...
% 'mnist', ...
};

for dataset = datasets
    parameter_observe(char(dataset));
    exp2_dataset(char(dataset));
end

function exp2_dataset(data_name)    
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    model_linear = model_initialization(data_name, model_linear);
    model_lrc = model_initialization(data_name, model_lrc);
    model_ssl = model_initialization(data_name, model_ssl);
    model_lrc_ssl = model_initialization(data_name, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    model_linear = training_process(model_linear, X, y, L);
    model_lrc = training_process(model_lrc, X, y, L);
    model_ssl = training_process(model_ssl, X, y, L);
    model_lrc_ssl = training_process(model_lrc_ssl, X, y, L);

    plot(1:size(model_linear.test_err, 2), [model_linear.test_err; model_ssl.test_err; model_lrc.test_err; model_lrc_ssl.test_err]);
    save(['../result/', data_name, '_converge.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
end

function model = training_process(model, X, y, L) 
    idx_rand = randperm(numel(y));
    % take use of Laplacian matrix
    idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
    idx_train = setdiff(idx_rand, idx_test);
    idx_train = idx_train(randperm(numel(idx_train)));

    XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
    XLX = min(1, 1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;

    idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
    
    % record training process
    model.n_record_batch = ceil(numel(idx_labeled) / model.n_batch);
    model.test_batch = true;
    model.X_test = X(idx_test, :); 
    model.y_test = y(idx_test);
    model = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model);
end   

function model = model_initialization(data_name, model)    
    model.data_name = data_name;
    model.n_folds = 5;
    model.n_repeats = 30;
    model.rate_test = 0.3;
    model.rate_labeled = 0.3;
    model.n_batch = 32;
    model.T = 50;
end