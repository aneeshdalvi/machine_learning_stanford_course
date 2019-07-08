function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% calculation cost function
h = X * theta;

%error = {the difference between h and y}
err = h - y;

% square each error term
err_sqr = err.^2;

% J = {multiply 1/(2*m) times the sum of the error_sqr vector}
unregJ = 1/(2*m) * sum(err_sqr);

theta(1) = 0;

sq_theta = sum(theta.^2);

regJ = (lambda / (2*m)) * sq_theta;

J = unregJ + regJ;

% calculating grad descent

%grad = (1/m) * X' * (h - y) + (lambda/m) * theta_reg;

UnregGrad = (1/m) * (X' * err);

RegGrad = (lambda / m) * theta;

grad = UnregGrad + RegGrad;

% =========================================================================

grad = grad(:);

end
