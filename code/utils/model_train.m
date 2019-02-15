function model_train(model)
    % load datasets
    [X_train, y_train, X_test, y_test] = load_data(model.data_name);
    L = construct_laplacian_graph(model.data_name, X_train, 3);
    n_dimension = size(X_train,2);
    
    % regularize labels to 1..C
    [y_train, y_test] = regularize_label(y_train, y_test);

    % cross validation to choose parameters
    if exist(['../data/', model.data_name, '/', 'cross_validation.mat'], 'file')
        load(['../data/', model.data_name, '/', 'cross_validation.mat'], 'errors_validate');
    else
        errors_validate = cross_validation(L, X_train, y_train, model);
    end
    
%     [~, loc_best] = min(errors_validate);
%     [d1, d2, d3] = ind2sub([numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)], loc_best);
%     model.tau_I = model.can_tau_I(d3);
%     model.tau_A = model.can_tau_A(d2);
%     model.tau_S = model.can_tau_S(d1);

    %model = learner_linear(errors_validate, model);
    %model = learner_lrc(errors_validate, model);
    model = learner_ssl(errors_validate, model);
    %model = learner_lrc_ssl(errors_validate, model);
    
    % take use of Laplacian matrix
    XLX = sparse(n_dimension, n_dimension);
    if model.tau_I ~= 0
        XLX = X_train' * L * X_train;
        XLX = min(1, 1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;
    end

    % record training and testing
    X_train = X_train(1 : ceil(numel(y_train) * model.rate_labeled), :);
    y_train = y_train(1 : ceil(numel(y_train) * model.rate_labeled));
    model.n_batch = 32;
    model.n_record_batch = ceil(numel(y_train) / model.n_batch);
    model.test_batch = true;
    model.X_test = X_test;
    model.y_test = y_test;
    model = ps3vt_multi_train(XLX, X_train, y_train, model);
    fprintf('ERR: %.4f\ttau_A: %.4f\ttau_I: %.4f\ttau_S: %.4f\tstep:%4.2f\tn_batch: %.0f\n', ...
        model.test_err(end), model.tau_A, model.tau_I, model.tau_S, model.step, model.n_batch);

    % visualization
    figure(1);
    plot(1:numel(model.train_loss), model.test_err)
    figure(2);
    plot(1:numel(model.train_loss),...
        [model.train_loss; model.train_objective;...
        model.train_trace; model.train_complexity; model.train_unlabeled])
end