initialization;
for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    exp1_dataset(char(dataset), model);
end

function exp1_dataset(data_name, model)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    %[model_linear, model_lrc, model_ssl, model_lrc_ssl] = parameter_choose(model);
    %model_linear.step = 1e+6;
    
    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    %% training and testing
    test_errs = zeros(4, model.n_repeats, model.T);
    for i_repeat = 1 : model.n_repeats
        idx_rand = randperm(numel(y));
        % take use of Laplacian matrix
        idx_test = idx_rand(1:ceil(model.rate_test * numel(y)));
        idx_train = setdiff(idx_rand, idx_test);
        idx_train = idx_train(randperm(numel(idx_train)));

        XLX = X(idx_train, :)' * L(idx_train, idx_train) * X(idx_train, :);

        %idx_labeled = idx_train(1 : ceil(numel(idx_train) * model.rate_labeled));
        idx_labeled = idx_train(sampling_with_labels(y(idx_train), model.rate_labeled));

        % record training and testing
        i_model = model;
        i_model.n_record_batch = 0 : ceil(numel(idx_labeled) / i_model.n_batch) :ceil(numel(idx_labeled) / i_model.n_batch) * model.T;
        i_model.test_batch = true;
        i_model.X_test = X(idx_test, :);
        i_model.y_test = y(idx_test);

        model_lrc_ssl = model_combination(i_model, model_lrc_ssl);
        model_ssl = model_combination(i_model, model_ssl);
        model_lrc = model_combination(i_model, model_lrc);
        model_linear = model_combination(i_model, model_linear);

        model_lrc_ssl = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model_lrc_ssl);
        model_ssl = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model_ssl);
        model_lrc = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model_lrc);
        model_linear = ps3vt_multi_train(XLX, X(idx_labeled, :), y(idx_labeled), model_linear);

        test_errs(1, i_repeat, :) = model_lrc_ssl.test_err;
        test_errs(2, i_repeat, :) = model_ssl.test_err;
        test_errs(3, i_repeat, :) = model_lrc.test_err;
        test_errs(4, i_repeat, :) = model_linear.test_err;
    end
    
    lrc_ssl_errs = mean(test_errs(1, :, end-4 : end), 3);
    ssl_errs = mean(test_errs(2, :, end-4 : end), 3);
    lrc_errs = mean(test_errs(3, :, end-4 : end), 3);
    linear_errs = mean(test_errs(4, :, end-4 : end), 3);
    
    fprintf('Dateset: %s\t Method: model_lrc_ssl\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S: %s\t step: %.0f\t\n', ... 
        model.data_name, mean(lrc_ssl_errs), std(lrc_ssl_errs), num2str(model_lrc_ssl.tau_I), num2str(model_lrc_ssl.tau_A), num2str(model_lrc_ssl.tau_S), model_lrc_ssl.step);
    fprintf('Dateset: %s\t Method: model_ssl\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\t step: %.0f\t\n', ... 
        model.data_name, mean(ssl_errs), std(ssl_errs), num2str(model_ssl.tau_I), num2str(model_ssl.tau_A), num2str(model_ssl.tau_S), model_ssl.step);
    fprintf('Dateset: %s\t Method: model_lrc\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\t step: %.0f\t\n', ... 
        model.data_name, mean(lrc_errs), std(lrc_errs), num2str(model_lrc.tau_I), num2str(model_lrc.tau_A), num2str(model_lrc.tau_S), model_lrc.step);
    fprintf('Dateset: %s\t Method: model_linear\t Mean: %.4f\t STD: %.4f\t tau_I: %s\t tau_A: %s\t tau_S:  %s\t step: %.0f\t\n', ... 
        model.data_name, mean(linear_errs), std(linear_errs), num2str(model_linear.tau_I), num2str(model_linear.tau_A), num2str(model_linear.tau_S), model_linear.step);
    
    save(['../result/', data_name, '_results.mat'], ...
        'test_errs');
    
    errs = [linear_errs; lrc_errs; ssl_errs; lrc_ssl_errs];
    output(errs, data_name);
end

function output(errs, data_name)
    errs = errs' .* 100;

    [~, loc_min] = min(mean(errs));
    d = errs - errs(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    % if bigger than 1.676, it is significantly better one.

    fid = fopen('../result/exp1/table_result.txt', 'a');
    fprintf(fid, '%s\t', data_name);
    for i = 1 : size(errs, 2)
        if i == loc_min
            fprintf(fid, '&\\textbf{%2.2f$\\pm$%.2f}\t', mean(errs(:, i)), std(errs(:, i)));
        elseif t(i) < 1.676
            fprintf(fid, '&\\underline{%2.2f$\\pm$%.2f}\t', mean(errs(:, i)), std(errs(:, i)));
        else
            fprintf(fid,'&%2.2f$\\pm$%.2f\t', mean(errs(:, i)), std(errs(:, i)));
        end
    end
    fprintf(fid, '\\\\\n');
    fclose(fid);
end
