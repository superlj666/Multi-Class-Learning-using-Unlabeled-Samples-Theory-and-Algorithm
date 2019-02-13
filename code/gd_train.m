function model = gd_train(XLX, X, Y, model)
%   Gradient Descent Train for Multi Class Learning
%
%   Inputs : 
%   XLX -   2 - D d * d Precomputed Matrices, which is to take use of unlabeled data.
%   X   -   2 - D d * n Matrices, which is training features.
%   Y   -   1 - D n * 1 Vector, which is trainning labels.
%
%   Additional parameters : 
%   - model.tau_A is the weight of the frobenius norm term of weight matrix W. It regulates
%     the complexity of the model.
%     Default value is 0.01.
%   - model.tau_I is the weight of the trace norm term of Laplacain regularization. It regulates
%     the influnce of unlabeled data.
%     Default value is 0.01.
%   - model.tau_S is the weight of the tail sum of singular values. It regulates
%     the influnce of local Rademacher complexity.
%     Default value is 0.01.   
%   - model.T is numer of training epochs for the batch stage.
%     Default value is 100.
%   - model.Test_Iter is numer of iterations to test.
%     Default value is 0, meaning no testing.
%
%
%   Author : Jian Li
%   Date : 2019/02/05  
%

if ~isfield(model, 'tau_A')
    model.tau_A = 0.01;
end
if ~isfield(model, 'tau_I')
    model.tau_I = 0.01;
end
if ~isfield(model, 'tau_S')
    model.tau_S = 0.01;
end
if ~isfield(model, 'T')
    model.T = 100;
end
if ~isfield(model, 'Test_Iter')
    model.Test_Iter = 0;
end

dimension_size = size(X, 1);
sample_size = size(X, 2);
class_size = numel(unique(Y));

if ~isfield(model, 'tail_size')
    model.tail_size = ceil(min(class_size, dimension_size) * 0.8)
end

W = ones(dimension_size, class_size);
error = zeros(model.T, 1);
for t = 1 : model.T
    grad_g = zeros(dimension_size, class_size);
    for i = 1 : class_size
        h_x = W' * X(:, i);
        margin_true = h_x(Y(i));
        h_x(i) = - Inf;
        [margin_pre, loc_pre] = max(h_x);
        if margin_true - margin_pre < 1
            grad_g( :, Y(i)) = grad_g( :, Y(i)) - X(:, i);
            grad_g( :, loc_pre) = grad_g( :, loc_pre) + X(:, i);
        end
    end
    grad_g = grad_g ./ class_size + W .* (2 * model.tau_A) + XLX * W .* (2 * model.tau_I);
    %disp(norm(grad_g, 'fro'));
    if norm(grad_g, 'fro') < 1e - 6
        break;
    end
    W = W - grad_g * step;
    
    if (model.tau_S)
        [U, S, V] = svd(W);
        model.tail_size = min(model.tail_size, min(dimension_size, class_size));
        for j = model.tail_size : min(dimension_size, class_size)
            S(j, j) = max(0, S(j, j) - step * model.tau_S);
        end
        W = U * S * V';
    end

    if (model.Test_Iter && ~mod(t, model.Test_Iter))
        test()
    end
end

