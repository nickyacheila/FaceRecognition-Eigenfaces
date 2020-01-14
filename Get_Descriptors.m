function [Weights] = Get_Descriptors(data,eigenfaces)
X=(data'); 
N=size(X,1);
%scale pixels from 0 to 1
X=X.*(1/max(max(data)));
% Computation of eigenvectors
mean_im = (sum(X, 2) / N);
miew=repmat(mean_im',size(X,2),1)';
% remove mean
X_1=X-miew;
X_1=X_1';
Weights=zeros(size(data,1),size(eigenfaces,2));
for i=1:size(data,1)
    image=X_1(i,:);
    Weights(i,:)=(eigenfaces'*image');    
end

end

