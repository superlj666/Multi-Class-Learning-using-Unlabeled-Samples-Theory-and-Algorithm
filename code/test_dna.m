addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rand('state', 0);

% load datasets
data_name = 'dna';
[X_train, y_train, X_test, y_test] = load_data(data_name);
n_dimension = size(X_train,2);
n_class = numel(unique(y_train));
rate_labeled = 0.25;

% parameters
model.tau_I = 0.0156;
model.tau_A = 0.0078;
model.tau_S = 0;
model.tail_start = 2;
model.T = 50;
model.step = 128;
model.n_batch = 1;
model.n_record_batch = 0;ceil(ceil(numel(y_train) * rate_labeled) / model.n_batch);
model.test_batch = true;
model.X_test = X_test;
model.y_test = y_test;

% regularize labels to 1..C
[y_train, y_test] = regularize_label(y_train, y_test);

% take use of Laplacian matrix
XLX = sparse(n_dimension, n_dimension);
if model.tau_I ~= 0
    L = construct_laplacian_graph(data_name, X_train, 3);
    XLX = X_train' * L * X_train;
    XLX = min(1, 1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;
end

% use a apart of training data as labeled data
X_train = X_train(1 : ceil(numel(y_train) * rate_labeled), :);
y_train = y_train(1 : ceil(numel(y_train) * rate_labeled));

model = ps3vt_multi_train(XLX, X_train, y_train, model);
model = record_batch(XLX, X_test, y_test, model, 'test');
fprintf('ERR: %.4f\ttau_A: %.4f\ttau_I: %.4f\ttau_S: %.4f\tstep:%4.2f\tn_batch: %.0f\n', ... 
                        model.test_err(end), model.tau_A, model.tau_I, model.tau_S, model.step, model.n_batch);
% visualization
% figure(1);
% plot(1:numel(model.train_loss), model.test_err)
% figure(2);
% plot(1:numel(model.train_loss),...
%     [model.train_loss; model.train_objective;...
%     model.train_trace; model.train_complexity; model.train_unlabeled])