addpath('../libsvm/matlab/');
addpath('./utils/');
clear;
rng(64);

can_datasets = {'dna'};

for dataset = can_datasets
model.n_folds = 5;
model.n_repeats = 30;
model.rate_test = 0.2;
model.rate_labeled = 0.2;
model.data_name = char(dataset);
model.n_batch = 32;
model.can_tau_I = 2 .^ -(7:2:11);
model.can_tau_A = 2 .^ -(3:4);
model.can_tau_S = 2 .^ -(5:2:9);
model.can_step = 2 .^ (3.5:0.5:4.5);

% load datasets
[X, y] = load_data(model.data_name);    
L = construct_laplacian_graph(model.data_name, X, 10);

% cross validation to choose parameters
if exist(['../data/', model.data_name, '/', 'cross_validation.mat'], 'file')
   load(['../data/', model.data_name, '/', 'cross_validation.mat'], 'errors_validate');
else
    errors_validate = cross_validation(L, X, y, model);
end
%errors_validate = cross_validation(L, X, y, model);

%% Choose parameters for every methods
% model_linear = learner_linear(errors_validate, model);
% model_lrc = learner_lrc(errors_validate, model);
% model_ssl = learner_ssl(errors_validate, model);
% model_lrc_ssl = learner_lrc_ssl(errors_validate, model);

%% Choose parameters for our method
model_lrc_ssl = learner_lrc_ssl_single(errors_validate, model);
model_ssl = model_lrc_ssl; model_ssl.tau_S = 0;
model_lrc = model_lrc_ssl; model_lrc.tau_I = 0;
model_linear = model_lrc; model_linear.tau_S = 0;

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
% model.tau_I = 2^-10;
% model.tau_A = 2^-4;
% model.tau_S = 2^-9;
% % for i_star = 5 : 10
% %     model.step = 2^i_star;
% %     repeat_test(model, 'test_0216', X, y, L);
% % end
% model = single_test(model, X, y, L);

function model = learner_lrc_ssl_single(errors_validate, model)
    cv_results = reshape([errors_validate{:, 1}], [numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)]);
    [~, loc_best] = min(cv_results(:));
    [d1, d2, d3] = ind2sub([numel(model.can_tau_S), numel(model.can_tau_A), numel(model.can_tau_I)], loc_best);

    model.tau_I = model.can_tau_I(d3);
    model.tau_A = model.can_tau_A(d2);
    model.tau_S = model.can_tau_S(d1);
end