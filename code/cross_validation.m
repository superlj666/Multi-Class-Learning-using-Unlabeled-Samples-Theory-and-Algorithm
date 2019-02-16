function errors_validate = cross_validation(L, X_train, y_train, model)
    rand('state', 0);

    n_folds = model.n_folds;
    rate_labeled = model.rate_labeled;
    data_name = model.data_name;
    can_tau_I = model.can_tau_I;
    can_tau_A = model.can_tau_A;
    can_tau_S = model.can_tau_S;

    % data split
    n_samples = numel(y_train);
    idx_rand = randperm(n_samples);
    step_fold = ceil(n_samples / n_folds);
    folds_XLX = cell(n_folds, 1);
    folds_train_labeled = cell(n_folds, 1);
    folds_validate = cell(n_folds, 1);
    for i_fold = 1 : n_folds
        % i-th fold samples as validation data and the others as trainning data
        folds_validate{i_fold, 1} = idx_rand((i_fold - 1) * step_fold + 1:min(i_fold * step_fold, n_samples));
        i_fold_train = setdiff(idx_rand, folds_validate{i_fold, 1});
        folds_XLX{i_fold, 1} = X_train(i_fold_train, :)' * L(i_fold_train, i_fold_train) * X_train(i_fold_train, :);

        % a part of i-th fold data as labeled data
        folds_train_labeled{i_fold, 1} = i_fold_train(1 : ceil(numel(i_fold_train) * rate_labeled));
    end

    % choose the best parameters    
    counter = 1;
    errors_validate = zeros(numel(can_tau_I) * numel(can_tau_A) * numel(can_tau_S), 1);
    for para_I = can_tau_I
        for para_A = can_tau_A
            for para_S = can_tau_S
                model.tau_I = para_I;
                model.tau_A = para_A;
                model.tau_S = para_S;

                for i_fold = 1 : n_folds
                    model.iter_batch = 0;
                    model.time_train = 0;
                    
                    XLX = min(1,1 / (sqrt(para_I) * norm(folds_XLX{i_fold, 1},'fro'))) * folds_XLX{i_fold, 1};

                    % training
                    model = ps3vt_multi_train(XLX, X_train(folds_train_labeled{i_fold, 1}, :), ...
                    y_train(folds_train_labeled{i_fold, 1}), model);

                    % validating
                    model = record_batch(XLX, X_train(folds_validate{i_fold, 1}, :), ...
                    y_train(folds_validate{i_fold, 1}), model, 'test');
                end

                fprintf('Grid: %.0f/%.0f\t ERR: %.4f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f\n', ...
                    counter, numel(can_tau_I) * numel(can_tau_A) * numel(can_tau_S), ...
                    mean(model.test_err), para_I, para_A, para_S);
                errors_validate(counter) = mean(model.test_err);
                counter = counter + 1;
                
                clear model;
            end
        end
    end
        
    [~, loc_best] = min(errors_validate);
    [d1, d2, d3] = ind2sub([numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)], loc_best);
    fprintf('-----Best ERR: %.4f\t tau_I: %.4f\t tau_A: %.4f\t tau_S: %.4f-----\n', ...
    errors_validate(loc_best), can_tau_I(d3), can_tau_A(d2), can_tau_S(d1));
    save(['../data/', data_name, '/', 'cross_validation.mat']);

    % cv_results = reshape(errors_validate, [numel(can_tau_S), numel(can_tau_A), numel(can_tau_I)]);
end
