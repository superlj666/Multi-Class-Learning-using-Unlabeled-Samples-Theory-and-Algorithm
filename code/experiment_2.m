initialization;
for dataset = datasets
    exp2_dataset(char(dataset));
end

function exp2_dataset(data_name)    
    %% Choose parameters for our method
    load(['../result/', data_name, '_results.mat'], 'test_errs');
    file_path = ['../result/exp2/', data_name];
    error_curve_save(file_path, mean(test_errs(4,:,:), 2), mean(test_errs(3,:,:), 2), mean(test_errs(2,:,:), 2), mean(test_errs(1,:,:), 2));
end

function error_curve_save(file_path, linear_errs, lrc_errs, ssl_errs, lrc_ssl_errs)
    linear_errs = linear_errs(:);
    lrc_errs = lrc_errs(:);
    ssl_errs = ssl_errs(:);
    lrc_ssl_errs = lrc_ssl_errs(:);
    fig=figure;
    x_length = min([size(linear_errs, 1), size(lrc_errs, 1), size(ssl_errs, 1), size(lrc_ssl_errs, 1)]);
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