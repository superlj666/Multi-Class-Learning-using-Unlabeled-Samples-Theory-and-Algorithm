initialization;
for dataset = datasets
    model.data_name = char(dataset);
    parameter_observe(char(dataset));
    exp1_dataset(char(dataset), model);
end

function exp1_dataset(data_name, model)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    model_linear = model_combination(model, model_linear);
    model_lrc = model_combination(model, model_lrc);
    model_ssl = model_combination(model, model_ssl);
    model_lrc_ssl = model_combination(model, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    %% training and testing
    lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
    ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
    lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
    linear_errs = repeat_test(model_linear, 'linear', X, y, L);
    save(['../result/', data_name, '_results.mat'], ...
        'linear_errs', 'lrc_errs', 'ssl_errs', 'lrc_ssl_errs');
    
    errs = [mean(linear_errs(:,end-4:end), 2), mean(lrc_errs(:,end-4:end), 2), mean(ssl_errs(:,end-4:end), 2), mean(lrc_ssl_errs(:,end-4:end), 2)];
    output(errs, data_name);
end

function output(errs, data_name)
    errs = errs .* 100;

    [~, loc_min] = min(mean(errs));
    d = errs - errs(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    % if bigger than 1.676, it is significantly better one.

    fid = fopen('../result/exp1/table_result.txt', 'a');
    fprintf(fid, '%s\t', data_name);
    for i = 1 : size(errs, 2)
        if i == loc_min
            fprintf(fid, '&\\textbf{%2.3f$\\pm$%.3f}\t', mean(errs(:, i)), std(errs(:, i)));
        elseif t(i) < 1.676
            fprintf(fid, '&\\underline{%2.3f$\\pm$%.3f}\t', mean(errs(:, i)), std(errs(:, i)));
        else
            fprintf(fid,'&%2.3f$\\pm$%.3f\t', mean(errs(:, i)), std(errs(:, i)));
        end
    end
    fprintf(fid, '\\\\\n');
    fclose(fid);
end

