addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rand('state', 0);

data_name = 'dna';

can_tau_I = [2.^(-7:-4),0];
can_tau_A = 2.^(-7:-4);
can_tau_S = [2.^(-7:-4),0];
n_batch = [1, 32];

% load datasets
[X_train, y_train, X_test, y_test] = load_data(data_name);
n_dimension = size(X_train,2);
n_class = numel(unique(y_train));
rate_labeled = 0.25;

% regularize labels to 1..C
[y_train, y_test] = regularize_label(y_train, y_test);

% take use of Laplacian matrix
L = construct_laplacian_graph(data_name, X_train, 3);
XLX = X_train' * L * X_train;

% use a apart of training data as labeled data
X_train = X_train(1 : ceil(numel(y_train) * rate_labeled), :);
y_train = y_train(1 : ceil(numel(y_train) * rate_labeled));

% cross-validation
for para_I = can_tau_I
    model.tau_I = para_I;
    XLX = min(1,1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;
    for para_A = can_tau_A
        model.tau_A = para_A;
        for para_S = can_tau_S
            model.tau_S = para_S;
            for para_n_batch = n_batch
                for i = 1 : 2
                    model.n_batch = para_n_batch;
                    model.tail_start = floor(min(n_class, n_dimension) * 0.8);
                    model.step = i / para_A;
                    model.n_batch = 32;
                    model.T = 50;
                    model.iter_batch = 0;
                    model.time_train = 0;
                    model = ps3vt_multi_train(XLX, X_train, y_train, model);
                    model = record_batch(XLX, X_test, y_test, model, 'test');

                    fprintf('ERR: %.4f\ttau_A: %.4f\ttau_I: %.4f\ttau_S: %.4f\tstep:%4.2f\tn_batch: %.0f\n', ... 
                        model.test_err(end), para_A, para_I, para_S, model.step, para_n_batch);
                end
            end
        end
    end
end

[~, loc_best] = min(model.test_err);
[d1, d2, d3, d4, d5] = ind2sub([numel(can_tau_I), numel(can_tau_A), numel(can_tau_S), numel(n_batch), 2], loc_best);

fprintf('ERR: %3.4f\ttau_A: %.4f\ttau_I: %.4f\ttau_S: %.4f\tn_batch: %.0f\tstep: %.0f\n', ...
    model.test_err(loc_best), can_tau_A(d1), can_tau_I(d2), can_tau_S(d3), n_batch(d4), d5);
save(['../data/', data_name, '/', sprintf('%.5f', model.test_err(loc_best)*100),'.mat']);
