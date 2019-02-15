function experiment_1(model)    
    % load datasets
    [X, y] = load_data(model.data_name);    
    L = construct_laplacian_graph(model.data_name, X, 3);

    % cross validation to choose parameters
    if exist(['../data/', model.data_name, '/', 'cross_validation.mat'], 'file')
        load(['../data/', model.data_name, '/', 'cross_validation.mat'], 'errors_validate');
    else
        idx_rand = randperm(numel(y));
        idx_train = idx_rand(1:ceil(0.7 * numel(y)));
        errors_validate = cross_validation(L, X(idx_train, :), y(idx_train), model);
    end
    model_linear = learner_linear(errors_validate, model);
    model_lrc = learner_lrc(errors_validate, model);
    model_ssl = learner_ssl(errors_validate, model);
    model_lrc_ssl = learner_lrc_ssl(errors_validate, model);

    repeat_test(model_linear, 'linear', X, y, L);
    repeat_test(model_lrc, 'lrc', X, y, L);
    repeat_test(model_ssl, 'ssl', X, y, L);
    repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
end

function repeat_test(model, model_name, X, y, L)
    rand('state', 0);
    
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

