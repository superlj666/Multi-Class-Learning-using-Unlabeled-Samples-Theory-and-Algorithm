addpath('../libsvm/matlab/');
addpath('./utils/');
addpath('./core_functions/');
clear;

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
    rng('default');
    parameter_observe(char(dataset));
    exp1_dataset(char(dataset));
end

function exp1_dataset(data_name)
    %% Choose parameters for our method
    load(['../result/', data_name, '_models.mat'], 'model_linear', 'model_lrc', 'model_ssl', 'model_lrc_ssl');
    model_linear = model_initialization(data_name, model_linear);
    model_lrc = model_initialization(data_name, model_lrc);
    model_ssl = model_initialization(data_name, model_ssl);
    model_lrc_ssl = model_initialization(data_name, model_lrc_ssl);

    % load datasets
    [X, y] = load_data(data_name);    
    L = construct_laplacian_graph(data_name, X, 10);

    errs_partition = cell(9, 1);
    for i_partition = 0.2 : 0.1 :1
        linear_errs = repeat_test(model_linear, 'linear', X, y, L);
        lrc_errs = repeat_test(model_lrc, 'lrc', X, y, L);
        ssl_errs = repeat_test(model_ssl, 'ssl', X, y, L);
        lrc_ssl_errs = repeat_test(model_lrc_ssl, 'lrc_ssl', X, y, L);
        
        errs_partition{i_partition, 1} = [mean(linear_errs(:,end-4:end), 2), mean(lrc_errs(:,end-4:end), 2), mean(ssl_errs(:,end-4:end), 2), mean(lrc_ssl_errs(:,end-4:end), 2)];
    end
    
    save(['../result/exp3/', data_name, '_errs_partition.mat'], ...
    'errs_partition');

end

function model = model_initialization(data_name, model)    
    model.data_name = data_name;
    model.n_folds = 5;
    model.n_repeats = 10;
    model.rate_test = 0.3;
    model.rate_labeled = 0.3;
    model.n_batch = 32;
    model.T = 50;
end

function y=draw_sample_curve(data_name)
    load(['../result/exp3/', data_name, '_errs_partition.mat']);
    fig=figure;
    x_list=1:7;
    errorbar(x_list,res_RLS(:,1),res_RLS(:,2), 'k--o','LineWidth',1);
    hold on;
    errorbar(x_list,res_LapRLS_pcg(:,1),res_LapRLS_pcg(:,2),'b-.^','LineWidth',1);
    errorbar(x_list,res_nystrom_pcg(:,1),res_nystrom_pcg(:,2),'r-x','LineWidth',1);

    max_level=max(res_RLS(:,1));
    min_level=min(res_nystrom_pcg(:,1));
    step=max_level-min_level;
    
    xticklabels({'20%', '30%', '40%', '50%' ,'60%', '70%', '80%', '90%', '100%'});
    grid on
    set(gca,'XLim',[0.5 7.5])
    set(gca,'YLim',[min_level-0.5*step max_level+1.2*step])
    legend({'RLS', 'LapRLS CG&PCG', 'Nystrom LapRLS CG&PCG'}, 'FontSize',12);
    ylabel('Error Rate(%)');
    xlabel('%# Labeled Samples');
    set(gca,'FontSize',20,'Fontname', 'Times New Roman');
    hold off;

    print(fig,str,'-depsc')
end