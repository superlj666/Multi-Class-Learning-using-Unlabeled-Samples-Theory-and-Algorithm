function error = test(X_test, Y_test, W)
%   Test function to output error rate of multi-class classification
%
%   Inputs : 
%   X_test  -   2 - D d * n_test Matrices, which is trainning features.
%   Y_test  -   1 - D n_test * 1 Matrices, which is training labels.
%   W       -   2 - D d * C Weight Matrix, which is result of model learning.
%
%
%   Author : Jian Li
%   Date : 2019/02/05  
%
    [~, Y_pre] = max(W' * X_test);
    error(t) = sum(Y_test' ~= Y_pre) / length(Y_test);
end