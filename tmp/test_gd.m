addpath('../libsvm/matlab/');
clear;
tic();

data_name='dna';
tau_L=1;
tau_A=0.001;
tau_I=0;
tau_S=0;
tail_size=2;
T=1000;
step=300/2000;

dataset_path=['../datasets/', data_name];
[y_train, X_train]=libsvmread(dataset_path);
%X_train = X_train/norm(X_train, 'fro');

test_path=['../datasets/', [data_name,'_test']];
[y_test, X_test]=libsvmread(test_path);
%X_test = X_test/norm(X_test, 'fro');

y_labels = unique(y_train);
for i_label = 1 : numel(y_labels)
    y_train(y_train==y_labels(i_label))=i_label;
    y_test(y_test==y_labels(i_label))=i_label;
end

% TODO normalize y
% TODO sgd instead of gd
n_l=length(y_train);
dimension_size=size(X_train,2);
class_size=length(unique(y_train));

XLX=sparse(dimension_size, dimension_size);

W=zeros(dimension_size, class_size);
%W=rand(dimension_size, class_size);

train_err=zeros(T,1);
train_loss=zeros(T,1);
train_complexity=zeros(T,1);
train_objective=zeros(T,1);
n_update=zeros(T,1);

test_err=zeros(T,1);
test_loss=zeros(T,1);
for epoch=1:T
    grad_g=zeros(dimension_size, class_size);
    
    % calculate gradient for entire dataset
    for i_sample=1:n_l
        % find true margin and predict margin
        h_x=W'*X_train(i_sample,:)';
        margin_true=h_x(y_train(i_sample));
        h_x(y_train(i_sample))=-Inf;
        [margin_pre, loc_pre]=max(h_x);

        % calculate subgradient of loss
        if margin_true-margin_pre < 1
            grad_g(:,y_train(i_sample))=grad_g(:,y_train(i_sample))-X_train(i_sample,:)';
            grad_g(:,loc_pre)=grad_g(:,loc_pre)+X_train(i_sample,:)';
            n_update(epoch)=n_update(epoch)+1;
        end
    end
    grad_g=grad_g./n_l;
    %grad_g=tau_L*grad_g./n_l+tau_A*2*W+tau_I*2*XLX*W;

    % update weight matrix
    if norm(grad_g,'fro') < 1e-4
        % calculate test objective and test error
        [~,y_pre]=max(W'*X_test');
        test_err(epoch)=sum(y_test'~=y_pre)/length(y_test);
        loss = 0;
        for i_test = 1:numel(y_test)
            h_x=W'*X_test(i_test,:)';
            margin_true = h_x(y_test(i_test));
            h_x(y_test(i_test)) = - Inf;
            margin_pre = max(h_x);
            loss = loss + max(1 - margin_true + margin_pre, 0);
        end
        test_loss(epoch) = loss / numel(y_test);
        break;
    end
    %W=W-sqrt(step^2-epoch^2)*grad_g;
    %W = W - step/(1+exp(epoch-T)*epoch)*grad_g;
    W = W - step*grad_g;
    %W = W - step*grad_g;
    
    % SVT with proximal gradient
    if tau_S ~=0
        [U,S,V]=svd(W);
        tail_size=min(tail_size,min(dimension_size, class_size));
        for i_diag=tail_size:min(dimension_size, class_size)
            S(i_diag,i_diag)=max(0,S(i_diag,i_diag)-step*tau_S);
        end
        W=U*S*V';
    end
    %W=min(1,1/(sqrt(tau_A)*norm(W,'fro')))*W;

    % calculate test objective and test error
    [~,y_pre]=max(W'*X_test');
    test_err(epoch)=sum(y_test'~=y_pre)/length(y_test);
    loss = 0;
    for i_test = 1:numel(y_test)
        h_x=W'*X_test(i_test,:)';
        margin_true = h_x(y_test(i_test));
        h_x(y_test(i_test)) = - Inf;
        margin_pre = max(h_x);
        loss = loss + max(1 - margin_true + margin_pre, 0);
    end
    test_loss(epoch) = loss / numel(y_test);

    % calculate train objective and train error every ep
    loss = 0;
    err = 0;
    for i = 1:numel(y_train)
        h_x=W'*X_train(i,:)';
        margin_true = h_x(y_train(i));
        h_x(y_train(i)) = - Inf;
        margin_pre = max(h_x);
        loss = loss + max(1 - margin_true + margin_pre, 0);
        err = err + (margin_true <= margin_pre);
    end
    train_err(epoch) = err/numel(y_train);
    train_loss(epoch) = loss/numel(y_train);
    train_complexity(epoch) = norm(W, 'fro')^2;
    train_objective(epoch) = tau_L*train_loss(epoch) + tau_A*train_complexity(epoch);
    
    fprintf('#epoch %.0f\tAER:%5.2f\tAEL:%5.2f\tUpdates:%.0f\n', ...
        epoch, train_err(epoch)*100, train_loss(epoch), n_update(epoch));
end

toc();