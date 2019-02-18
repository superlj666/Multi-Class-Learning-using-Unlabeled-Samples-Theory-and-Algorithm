addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;
rng(64);

datasets = {
%'iris', ...
'wine', ...
%'glass', ...
% 'svmguide4', ...
% 'svmguide2', ...
%'vowel', ...
%'vehicle', ...
% 'dna', ...
% 'segment', ...
% 'satimage', ...
% 'usps', ...
% 'pendigits', ...
% 'letter', ...
% 'protein', ...
% 'poker', ...
% 'shuttle', ...
% 'Sensorless', ...
% 'mnist', ...
};

for dataset = datasets
    parameter_observe(char(dataset));
    exp2_dataset(char(dataset));
end

function exp2_dataset(data_name)    
    %% Choose parameters for our method
    load(['../result/', data_name, '_results.mat'], 'linear_errs', 'lrc_errs', 'ssl_errs', 'lrc_ssl_errs');
    file_path = ['../result/exp2/', data_name];
    error_curve_save(file_path, mean(linear_errs, 1), mean(lrc_errs, 1), mean(ssl_errs, 1), mean(lrc_ssl_errs, 1));
end

function error_curve(linear_errs, lrc_errs, ssl_errs, lrc_ssl_errs)
    x_length = min([size(linear_errs, 2), size(lrc_errs, 2), size(ssl_errs, 2), size(lrc_ssl_errs, 2)]);
    plot(1:x_length, [linear_errs; lrc_errs; ssl_errs; lrc_ssl_errs]);
end

function error_curve_save(file_path, linear_errs, lrc_errs, ssl_errs, lrc_ssl_errs)
    fig=figure;
    x_length = min([size(linear_errs, 2), size(lrc_errs, 2), size(ssl_errs, 2), size(lrc_ssl_errs, 2)]);
    plot(1:x_length, lrc_ssl_errs, '-','LineWidth',2);
    hold on;
    plot(1:x_length, lrc_errs, '-','LineWidth',2);
    plot(1:x_length, ssl_errs, '-','LineWidth',2);
    plot(1:x_length, linear_errs, '-','LineWidth',2);

    grid on
    legend({'Ours', 'LRC-MC', 'SS-MC', 'Linear-MC'}, 'FontSize',12);
    ylabel('Error Rate(%)');
    xlabel('The number of iterations');
    set(gca,'FontSize',20,'Fontname', 'Times New Roman');
    hold off;
    
    print(fig,file_path,'-depsc')
end