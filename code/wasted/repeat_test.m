function repeat_test(model, model_name, X, y, L)
    for i_repeat = 1 : model.n_repeats
        idx_rand = randperm(numel(y));
        % take use of Laplacian matrix
        idx_train = idx_rand(1:ceil(0.7 * numel(y)));
        idx_test = setdiff(idx_rand, idx_train);

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
    
    fprintf('Dateset: %s\t Method: %s\t Mean: %.4f\t STD: %.4f\n', ... 
        model.data_name, model_name, mean(model.test_err), std(model.test_err));
    save(['../data/', model.data_name, '/', model_name,'.mat']);
end