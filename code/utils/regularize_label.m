function [y_train, y_test] = regularize_label(y_train, y_test)
    y_labels = unique(y_train);
    for i_label = 1 : numel(y_labels)
        y_train(y_train == y_labels(i_label)) = i_label;
        y_test(y_test == y_labels(i_label)) = i_label;
    end
end