data_name='dna';
tau_A=0.1;
tau_I=0;
tau_S=0;
tail_size=2;
T=100;
step=0.01;

dataset_path=['/home/lijian/datasets/', data_name];
[y, X]=libsvmread(dataset_path);

test_path=['/home/lijian/datasets/', [data_name,'_test']];
[y_test, X_test]=libsvmread(test_path);
% TODO normalize y
% TODO sgd instead of gd
n_l=length(y);
dimension_size=size(X,2);
class_size=length(unique(y));

XLX=sparse(dimension_size, dimension_size);

W=ones(dimension_size, class_size);
acc=zeros(T,1);
for t=1:T
    idx_rand = randperm(n_l);
    for j=1:n_l
        i=idx_rand(j);
        grad_g=zeros(dimension_size, class_size);
        h_x=W'*X(i,:)';
        h_x_i=h_x(y(i));
        h_x(i)=-Inf;
        [max_value, max_loc]=max(h_x);
        if h_x_i-max_value < 1
            grad_g(:,y(i))=grad_g(:,y(i))-X(i,:)';
            grad_g(:,max_loc)=grad_g(:,max_loc)+X(i,:)';
        end
        grad_g=grad_g./n_l+W.*(2*tau_A)+XLX*W.*(2*tau_I);
        %disp(norm(grad_g,'fro'));
        if norm(grad_g,'fro') < 1e-6
            break;
        end
        W=W-grad_g*step;
        
        [U,S,V]=svd(W);
        tail_size=min(tail_size,min(dimension_size, class_size));
        for j=tail_size:min(dimension_size, class_size)
            S(j,j)=max(0,S(j,j)-step*tau_S);
        end
        W=U*S*V';
        
    end
    [~,y_pre]=max(W'*X_test');
    acc(t)=sum(y_test'==y_pre)/length(y_test);
end

disp('test');