function result = mgSums(num_features, d)
if(num_features<=1)
    result = d;
else
    result = zeros(0, num_features);    
    for(i = d:-1:0)
        rc = mgSums(num_features - 1, d - i);
        result = [result; i * ones(size(rc,1), 1), rc];
    end    
end