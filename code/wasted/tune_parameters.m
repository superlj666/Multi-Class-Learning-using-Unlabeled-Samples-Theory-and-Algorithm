addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rng('default');

data_name = 'dna';

can_tau_I = [2.^(-7:-3),0];
can_tau_A = 2.^(-7:-3);
can_tau_S = [2.^(-7:-3),0];

% load datasets
[X_train, y_train, X_test, y_test] = load_data(data_name);
n_dimension = size(X_train,2);
n_class = max(y_train);
rate_labeled = 0.25;

% regularize labels to 1..C
[y_train, y_test] = regularize_label(y_train, y_test);

% take use of Laplacian matrix
L = construct_laplacian_graph(data_name, X_train, 10);
XLX = X_train' * L * X_train;

% use a apart of training data as labeled data
X_train = X_train(1 : ceil(numel(y_train) * rate_labeled), :);
y_train = y_train(1 : ceil(numel(y_train) * rate_labeled));

% cross-validation
counter = 1;
for para_I = can_tau_I
    model.tau_I = para_I;
    XLX = min(1,1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;
    for para_A = can_tau_A
        model.tau_A = para_A;
        for para_S = can_tau_S
            model.tau_S = para_S;
            model.tail_start = floor(min(n_class, n_dimension) * 0.8);
            model.step = 1 / para_A;
            model.n_batch = 32;
            model.T = 50;
            model.iter_batch = 0;
            model.time_train = 0;
            model = ps3vt_multi_train(XLX, X_train, y_train, model);
            model = record_batch(XLX, X_test, y_test, model, 'test');
            
            fprintf('Round: %.0f/%.0f\t ERR: %.4f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f\n', ...
                counter, numel(can_tau_I) * numel(can_tau_A) * numel(can_tau_S), model.test_err(end), para_I, para_A, para_S);
            counter = counter + 1;
        end
    end
end

[~, loc_best] = min(model.test_err);
[d1, d2, d3] = ind2sub([numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
m_results = reshape(model.test_err, [numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);

fprintf('-----Best ERR: %.4f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f-----\n', ...
    model.test_err(loc_best), can_tau_I(d3), can_tau_A(d2), can_tau_S(d1));
save(['../data/', data_name, '/', sprintf('%.5f', model.test_err(loc_best)*100),'.mat']);
