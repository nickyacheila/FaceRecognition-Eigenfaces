function [index_class] = NN_Classify(image_descr,train_data_descr)
dist=zeros(size(train_data_descr,1),1);

for i=1:size(train_data_descr,1)
    
    dist(i)= sqrt(sum((image_descr-train_data_descr(i,:)).^2));
     
end

[m,index_class]=min(dist);