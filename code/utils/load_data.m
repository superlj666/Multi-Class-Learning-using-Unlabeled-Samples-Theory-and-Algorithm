function [X_train, y_train, X_test, y_test] = load_data(data_name)
    dataset_path = ['../datasets/', data_name];
    [y_train, X_train] = libsvmread(dataset_path);
    test_path = ['../datasets/', [data_name,'_test']];
    [y_test, X_test] = libsvmread(test_path);
end