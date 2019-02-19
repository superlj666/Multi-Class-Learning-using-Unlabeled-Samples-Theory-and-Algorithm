addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;

datasets = {
'iris', ...
'wine', ...
'glass', ...
% 'svmguide4', ...
% 'svmguide2', ...
%'vowel', ...
%'vehicle', ...
%'dna', ...
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
    rng('default');
    %parameter_observe(char(dataset));
    exp3_dataset(char(dataset));
    draw_sample_curve(char(dataset));
end

function exp3_dataset(data_name)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    model_linear = model_initialization(data_name, model_linear);
    model_lrc = model_initialization(data_name, model_lrc);
    model_ssl = model_initialization(data_name, model_ssl);
    model_lrc_ssl = model_initialization(data_name, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    errs_partition = zeros(4, 9, model_lrc_ssl.n_repeats);
    for i_partition = 1 : 9        
        model_linear.rate_labeled = i_partition * 0.1 + 0.1;
        model_lrc.rate_labeled = i_partition * 0.1 + 0.1;
        model_ssl.rate_labeled = i_partition * 0.1 + 0.1;
        model_lrc_ssl.rate_labeled = i_partition * 0.1 + 0.1;

        linear_errs = repeat_test(model_linear, 'linear', X, y, L);
        lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
        ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
        lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
        
        errs_partition(1, i_partition, :) = mean(linear_errs(:,end-4:end), 2);
        errs_partition(2, i_partition, :) = mean(lrc_errs(:,end-4:end), 2);
        errs_partition(3, i_partition, :) = mean(ssl_errs(:,end-4:end), 2);
        errs_partition(4, i_partition, :) = mean(lrc_ssl_errs(:,end-4:end), 2);
    end
    
    save(['../result/exp3/', data_name, '_errs_partition.mat'], ...
    'errs_partition');
end

function model = model_initialization(data_name, model)    
    model.data_name = data_name;
    model.n_folds = 5;
    model.n_repeats = 10;
    model.rate_test = 0.3;
    model.n_batch = 32;
    model.T = 50;
    
    model.tau_I = 0;%[2 .^ -(7:2:11), 0];
    model.tau_A = 10^-7;%2 .^ -(3:4);
    model.tau_S = 0;%[2 .^ -(5:2:9), 0];
    model.step = 10^3;%2 .^ (3.5:0.5:4.5);
end

function draw_sample_curve(data_name)
    load(['../result/exp3/', data_name, '_errs_partition.mat'], 'errs_partition');
    linear_errs_mean = mean(errs_partition(1, :, :), 3);
    lrc_errs_mean = mean(errs_partition(2, :, :), 3);
    ssl_errs_mean = mean(errs_partition(3, :, :), 3);
    lrc_ssl_errs_mean = mean(errs_partition(4, :, :), 3);
    
    linear_errs_std = zeros(1, 9);
    lrc_errs_std = zeros(1, 9);
    ssl_errs_std = zeros(1, 9);
    lrc_ssl_errs_std = zeros(1, 9);
    for i = 1:9
        linear_errs_std(1, i) = std(errs_partition(1, i, :));
        lrc_errs_std(1, i) = std(errs_partition(2, i, :));
        ssl_errs_std(1, i) = std(errs_partition(3, i, :));
        lrc_ssl_errs_std(1, i) = std(errs_partition(4, i, :));
    end

    fig=figure;
    x_list=1:9;
    errorbar(x_list, linear_errs_mean, linear_errs_std, 'k--o','LineWidth',1);
    hold on;
    errorbar(x_list, lrc_errs_mean, lrc_errs_std,'b-.^','LineWidth',1);
    errorbar(x_list, ssl_errs_mean, ssl_errs_std,'r-x','LineWidth',1);
    errorbar(x_list, lrc_ssl_errs_mean, lrc_ssl_errs_std,'g-*','LineWidth',1);

%     max_level=max(lrc_ssl_errs_mean);
%     min_level=min(linear_errs_mean);
%     step=max_level-min_level;
    
    xticklabels({'20%', '30%', '40%', '50%' ,'60%', '70%', '80%', '90%', '100%'});
    grid on
%     set(gca,'XLim',[0.5 7.5])
%     set(gca,'YLim',[min_level-0.5*step max_level+1.2*step])
    legend({'Ours', 'LRC-MC', 'SS-MC', 'Linear-MC'}, 'FontSize',12);
    ylabel('Error Rate(%)');
    xlabel('%# Labeled Samples');
    set(gca,'FontSize',20,'Fontname', 'Times New Roman');
    hold off;

    print(fig,['../result/exp3/', data_name],'-depsc')
end