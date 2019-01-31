function new_X = mapFeature(X, degree)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%

m  = size(X)(1);  % number of training examples
num_features = size(X)(2);     % number of features
max_degree  = 3;     % Order of polynomial



stacked = zeros(0, num_features); %this will collect all the coefficients...    
for(d = 1:degree)          % for degree 1 polynomial to degree 'order'
    stacked = [stacked; mgSums(num_features, d)];
end


new_X = zeros(size(X,1), size(stacked,1));
for(i = 1:size(stacked,1))
    accumulator = ones(m, 1);
    for(j = 1:num_features)
        accumulator = accumulator .* X(:,j).^stacked(i,j);
    end
    new_X(:,i) = accumulator;
end

% Add x0 intercept term to new_X
new_X = [ones(m, 1) new_X];
