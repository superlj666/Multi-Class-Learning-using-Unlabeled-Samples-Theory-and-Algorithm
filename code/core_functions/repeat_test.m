function test_errs = repeat_test(model, model_name, X, y, L)    
    % rng('default');
    for i_repeat = 1 : model.n_repeats
        idx_rand = randperm(numel(y));
        % take use of Laplacian matrix
        idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
        idx_train = setdiff(idx_rand, idx_test);
        idx_train = idx_train(randperm(numel(idx_train)));

        XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);
        XLX = min(1, 1 / (sqrt(model.tau_I) * norm(XLX,'fro'))) * XLX;

        idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
        % record training and testing
        model = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model);
        model = record_batch(XLX, X(idx_test, :), y(idx_test), model, 'test');
        model.iter_batch = 0;
        model.time_train = 0;
        model.epoch = 0;
    end
    
    fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f\t\n', ... 
        model.data_name, model_name, mean(model.test_err), std(model.test_err), model.tau_I, model.tau_A, model.tau_S);
    test_errs = model.test_err;
end