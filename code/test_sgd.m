addpath('../libsvm/matlab/');
clear;
rand('state', 0);
tic();

data_name='dna';
tau_A=1e-6;
tau_I=1e-10;
tau_S=2e-6;
tail_size=2;
T=30;
step=min(1e+5,1/tau_A);

dataset_path=['../datasets/', data_name];
[y_train, X_train]=libsvmread(dataset_path);

test_path=['../datasets/', [data_name,'_test']];
[y_test, X_test]=libsvmread(test_path);

y_labels = unique(y_train);
for i_label = 1 : numel(y_labels)
    y_train(y_train==y_labels(i_label))=i_label;
    y_test(y_test==y_labels(i_label))=i_label;
end

% TODO normalize y
% TODO sgd instead of gd
dimension_size=size(X_train,2);
class_size=length(unique(y_train));

XLX = sparse(dimension_size, dimension_size);
if tau_I ~=0
    L=construct_laplacian_graph(data_name, X_train, 3);
    XLX=X_train'*L*X_train;
end

X_train=X_train(1:500,:);
y_train=y_train(1:500);

W=rand(dimension_size, class_size);

n_l=numel(y_train);
train_err=[];
train_loss=[];
train_complexity=[];
train_unlabeled=[];
train_trace=[];
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
        grad_g=grad_g./n_l+2*tau_A*W+2*tau_I*XLX*W;
        
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
                S(i_diag,i_diag)=max(0,S(i_diag,i_diag)-step/(epoch*n_l+i_sample)*tau_S);
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
    train_complexity(end+1) = tau_A*norm(W, 'fro')^2;
    train_unlabeled(end+1) = tau_I*trace(W'*XLX*W);
    train_trace(end+1) =  tau_S*sum(sqrt(eig(S'*S)));
    train_objective(end+1) = train_loss(end) + train_complexity(end) + train_unlabeled(end) + train_trace(end);
    
    fprintf('#epoch %.0f\tAER:%5.2f\tAEL:%5.2f\tUpdates:%.0f\tTestErr:%.4f\n', ...
        epoch, train_err(end)*100, train_loss(end), n_update(end), mean(test_err(max(1,numel(test_err)-5):end)));
    
    if train_loss(end) < 1e-6
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
        break;
    end
end

toc();