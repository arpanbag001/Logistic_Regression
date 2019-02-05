%Arpan Bag

function [J, grad] = costFunctionReg(theta, X, Y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, Y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(Y); % number of training examples

J = 0;	%Cost
grad = zeros(size(theta));

% =========================== CODE HERE =========================
% Instructions: Compute the cost of a particular choice of theta.
%               Set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X*theta);


J = 1/m*(-Y'*log(h)-(1-Y)'*log(1-h)) + lambda/(2*m)*sum(theta(2:end).^2);


first_weight = (1/m*(X'*(h-Y)));
rest_weights = (1/m*(X'*(h-Y)) + lambda/m*theta);
grad(1) = first_weight(1);	%No need to use regularization for the first weight
grad(2:end) = rest_weights(2:end);


% =============================================================

end
