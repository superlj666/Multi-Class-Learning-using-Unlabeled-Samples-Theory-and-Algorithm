function model = single_test(model, X, y, L)
    idx_rand = randperm(numel(y));
    idx_train = idx_rand(1:ceil(0.7 * numel(y)));
    idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
    X_train = X(idx_labeled, :);
    y_train = y(idx_labeled);
    
    idx_test = setdiff(idx_rand, idx_train);
    X_test = X(idx_test, :);
    y_test = y(idx_test);
    n_dimension = numel(y_train);

    % take use of Laplacian matrix
    XLX = sparse(n_dimension, n_dimension);
    if model.tau_I ~= 0
        XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
        XLX = min(1, 1 / (1 * norm(XLX,'fro'))) * XLX;
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