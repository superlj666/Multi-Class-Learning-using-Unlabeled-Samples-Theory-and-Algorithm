addpath('../libsvm/matlab/');
clear;
tic();

data_name='dna';
tau_A=1e-5;
tau_I=0;
tau_S=0;
tail_size=2;
T=10;
step=1/tau_A;

dataset_path=['../data/', data_name];
[y_train, X_train]=libsvmread(dataset_path);

test_path=['../data/', [data_name,'_test']];
[y_test, X_test]=libsvmread(test_path);

y_labels = unique(y_train);
for i_label = 1 : numel(y_labels)
    y_train(y_train==y_labels(i_label))=i_label;
    y_test(y_test==y_labels(i_label))=i_label;
end

% TODO normalize y
% TODO sgd instead of gd
n_l=numel(y_train);
dimension_size=size(X_train,2);
class_size=length(unique(y_train));

XLX=sparse(dimension_size, dimension_size);

W=rand(dimension_size, class_size);

train_err=[];
train_loss=[];
train_complexity=[];
train_objective=[];
n_update=[];

test_err=[];
test_loss=[];

for epoch=1:T
    idx_rand = randperm(n_l);
    n_update(end+1)=0;
    for i_sample=1:n_l
        grad_g=zeros(dimension_size, class_size);
        i_idx=idx_rand(i_sample);
        
        % find true margin and predict margin
        h_x=W'*X_train(i_idx,:)';
        margin_true=h_x(y_train(i_idx));
        h_x(y_train(i_idx))=-Inf;
        [margin_pre, loc_pre]=max(h_x);
        
        % calculate gradient for every instance
        if margin_true-margin_pre < 1
            grad_g(:,y_train(i_idx))=grad_g(:,y_train(i_idx))-X_train(i_idx,:)';
            grad_g(:,loc_pre)=grad_g(:,loc_pre)+X_train(i_idx,:)';
            n_update(end)=n_update(end)+1;
        end
        grad_g=grad_g./n_l+2*tau_A*W+XLX*W.*(2*tau_I);
        
        % update weight matrix
        if norm(grad_g,'fro') < 1e-6
            continue;
        end
        W = W - step/(epoch*n_l+i_sample)*grad_g;
        
        % SVT with proximal gradient
        if tau_S ~=0
            [U,S,V]=svd(W);
            tail_size=min(tail_size,min(dimension_size, class_size));
            for i_diag=tail_size:min(dimension_size, class_size)
                S(i_diag,i_diag)=max(0,S(i_diag,i_diag)-step*tau_S);
            end
            W=U*S*V';
        end
        W=min(1,1/(sqrt(tau_A)*norm(W,'fro')))*W;
        %W=min(1,1/norm(W,'fro'))*W;
        
        % calculate test objective and test error
        if mod(i_sample,100) == 0
            [~,y_pre]=max(W'*X_test');
            test_err(end+1)=sum(y_test'~=y_pre)/length(y_test);
            
            loss = 0;
            for i_test = 1:numel(y_test)
                h_x=W'*X_test(i_test,:)';
                margin_true = h_x(y_test(i_test));
                h_x(y_test(i_test)) = - Inf;
                margin_pre = max(h_x);
                loss = loss + max(1 - margin_true + margin_pre, 0);
            end
            test_loss(end+1) = loss / numel(y_test);
        end
    end
    
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
    train_err(end+1) = err/numel(y_train);
    train_loss(end+1) = loss/numel(y_train);
    train_complexity(end+1) = norm(W, 'fro')^2;
    train_objective(end+1) = train_loss(end) + tau_A*train_complexity(end);
    
    fprintf('#epoch %.0f\tAER:%5.2f\tAEL:%5.2f\tUpdates:%.0f\n', ...
        epoch, train_err(end)*100, train_loss(end), n_update(end));
    
    if train_loss(end) == 0
        break;
    end
end

toc();