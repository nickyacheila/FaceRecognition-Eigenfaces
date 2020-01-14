clear;
load('5.mat');
% load('5.mat');
% load('7.mat');
load('ORL_32x32.mat');
train=fea(trainIdx,:);
test=fea(testIdx,:);
k=10;
[eigenfaces,indexes]=EigenAn(train,k);
f=figure;
set(f, 'Name', 'Eigen Faces');
for i = 1 : k
 subplot(5, k/5, i); imshow(reshape(eigenfaces(:,i), [32, 32]), []);
end
% project train data and get descriptors
[weights_train] = Get_Descriptors(train,eigenfaces);

% project test data and get descriptors
[weights_test] = Get_Descriptors(test,eigenfaces);

% test reconstruction
% choose image
% pos = randi(size(train,1));
pos=3;
image= train(pos,:);

f=figure;
set(f, 'Name', 'Face Reconstruction');
subplot(2,2,1);imshow(reshape(image, [32, 32]), []);
W=weights_train(pos,:);
is_this=zeros(1024,1);
% is_this=(mean_im);
for j=1:k
    is_this=is_this+W(1,j).*eigenfaces(:,j);
    
end

subplot(2,2,2);imshow(reshape(is_this, [32, 32]), []);

% classification
classes=zeros(length(testIdx),1);
err=zeros(length(testIdx),1);
for l=1:length(testIdx)
image_descr=weights_test(l,:);
[index_class] = NN_Classify(image_descr,weights_train);
classes(l)=index_class;
err(l,1)= sqrt(sum((train(index_class,:)-test(l,:)).^2))/1024;
end
image_descr=weights_test(160,:);
[index_class] = NN_Classify(image_descr,weights_train);
f=figure;
set(f, 'Name', 'Classification');
subplot(2,2,1);imshow(reshape(test((160),:), [32, 32]), []);
subplot(2,2,2);imshow(reshape(train(index_class,:), [32, 32]), []);
