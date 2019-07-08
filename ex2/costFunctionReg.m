function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);

err = h - y;

unreg_grad = (1/m) * (X' * err);

left = sum(-y' * log(h));

right= sum((1- y)' * log(1-h));

unregCost = (1 / m) * (left - right);

theta(1) = 0;

reg_grad = (lambda / m) * theta;

grad = unreg_grad + reg_grad;

sq_theta = theta' * theta;

regCost = (lambda / (2 * m)) * sq_theta;

J = unregCost + regCost;




% =============================================================

end
