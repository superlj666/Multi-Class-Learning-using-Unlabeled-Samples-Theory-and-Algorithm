addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng(64);

can_datasets = {'segment'};

for dataset = can_datasets
    %% Choose parameters for our method
    model_lrc_ssl = learner_lrc_ssl_single(errors_validate, model);
    model_ssl = model_lrc_ssl; model_ssl.tau_S = 0;
    model_lrc = model_lrc_ssl; model_lrc.tau_I = 0;
    model_linear = model_lrc; model_linear.tau_S = 0;

    %% training and testing
    linear_errs = repeat_test(model_linear, 'linear', X, y, L);
    lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
    ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
    lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
    save(['../result/', model.data_name, '_errors.mat'], 'linear_errs', 'lrc_errs', 'ssl_errs', 'lrc_ssl_errs');

    errs = [linear_errs; lrc_errs; ssl_errs; lrc_ssl_errs];
    errs = errs' .* 100;

    [~, loc_min] = min(mean(errs));
    d = errs - errs(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    % if bigger than 1.676, it is significantly better one.

    fid = fopen('table_result.txt', 'a');
    fprintf(fid, '%s\t', model.data_name);
    for i = 1 : 4
        if i == loc_min
            fprintf(fid, '&\\textbf{&%2.3f$\\pm$%.3f}\t', mean(errs(:, i)), std(errs(:, i)));
        elseif t(i) < 1.676
            fprintf(fid, '&\\underline{&%2.3f$\\pm$%.3f}\t', mean(errs(:, i)), std(errs(:, i)));
        else
            fprintf(fid,'&%2.3f$\\pm$%.3f\t', mean(errs(:, i)), std(errs(:, i)));
        end
    end
    fprintf(fid, '\\\\\n');
    fclose(fid);
end
