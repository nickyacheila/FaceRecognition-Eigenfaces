function [eigenfaces,indexes] = EigenAn(data,k)


X=(data'); 
N=size(X,1);
%scale pixels from 0 to 1
X=X.*(1/max(max(data)));
% Computation of eigenvectors
mean_im = (sum(X, 2) / N);
miew=repmat(mean_im',size(X,2),1)';
% remove mean
X_1=X-miew;
T=(X_1'*X_1)/N;
%calculate eigenvectors/values of T
[eigenvect,eigenvalues]=eig(T);
%calculate of original data
U =X_1 * eigenvect; 
U = normc(U);
d_eigen=diag(eigenvalues);
%top K indexes
[top_eigv,indexes]=maxk(d_eigen,k);
eigenvect=eigenvect(:,indexes);
% V=V';
eigenfaces=U(:,indexes);


end

