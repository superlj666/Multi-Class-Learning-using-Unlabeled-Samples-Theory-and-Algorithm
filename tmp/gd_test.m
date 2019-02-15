function model = gd_test(XLX, X, Y, model)
%   Loss and Objective function in Gradient Descent Train for Multi Class Learning and Test Error.
%
%   Inputs : 
%   XLX -   2 - D d * d Precomputed Matrices, which is to take use of unlabeled data.
%   X   -   2 - D d * n Matrices, which is training features.
%   Y   -   1 - D n * 1 Vector, which is trainning labels.
%
%   Additional parameters : 
%   - model.X_test is test features, 
%   if it is empty means no test in each test iteration.
%   - model.Y_test is test labels.
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
if ~isfield(model, 'loss'):
    model.loss = [];
end
if ~isfield(model, 'complexity_regularization'):
    model.complexity_regularization = [];
end
if ~isfield(model, 'laplacian_regularization'):
    model.laplacian_regularization = [];
end
if ~isfield(model, 'lrc_regularization'):
    model.lrc_regularization = [];
end
if ~isfield(model, 'objective'):
    model.objective = [];
end

loss = 0;
for i = 1:numel(Y)
    h_x = model.weights' * X(:, i);
    margin_true = h_x(Y(i));
    h_x(i) = - Inf;
    margin_pre = max(h_x);
    loss = loss + max(1 - margin_true + margin_pre, 0);
end
model.loss(end+1) = loss / numel(Y);

model.complexity_regularization(end+1) = norm(model.weights, 'fro');
model.laplacian_regularization(end+1) = trace(W'*XLX*W);
[U, S, V] = svd(W);
model.lrc_regularization(end+1) = sum(S(model.tail_size,:));
model.objective = model.loss(end) + model.tau_A * model.complexity_regularization(end) + model.tau_I * model.laplacian_regularization(end) + model.tau_S * model.lrc_regularization(end);

if (model.)