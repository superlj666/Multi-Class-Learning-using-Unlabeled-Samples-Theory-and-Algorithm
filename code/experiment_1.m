addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng(64);

datasets = {
'dna', ...
'segment', ...
'satimage', ...
'usps', ...
'pendigits', ...
'letter', ...
'protein', ...
'poker', ...
'shuttle', ...
'Sensorless', ...
% 'mnist', ...
};

for dataset = datasets
    parameter_observe(char(dataset));
    exp1_dataset(char(dataset));
end

function exp1_dataset(data_name)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], model_linear, model_lrc, model_ssl, model_lrc_ssl);
    model_linear = model_initialization(data_name, model_linear);
    model_lrc = model_initialization(data_name, model_lrc);
    model_ssl = model_initialization(data_name, model_ssl);
    model_lrc_ssl = model_initialization(data_name, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    %% training and testing
    linear_errs = repeat_test(model_linear, 'linear', X, y, L);
    lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
    ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
    lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
    save(['../result/', data_name, '_errors.mat'], 'linear_errs', 'lrc_errs', 'ssl_errs', 'lrc_ssl_errs');

    errs = [linear_errs; lrc_errs; ssl_errs; lrc_ssl_errs];
    errs = errs' .* 100;

    [~, loc_min] = min(mean(errs));
    d = errs - errs(:, loc_min);
    t = mean(d)./(std(d)/size(d,1));
    t(isnan(t)) = Inf;
    % if bigger than 1.676, it is significantly better one.

    fid = fopen('table_result.txt', 'a');
    fprintf(fid, '%s\t', data_name);
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

function model = model_initialization(data_name, model)    
    model.data_name = data_name;
    model.n_folds = 5;
    model.n_repeats = 50;
    model.rate_test = 0.3;
    model.rate_labeled = 0.5;
    model.n_batch = 32;
    model.T = 50;
end