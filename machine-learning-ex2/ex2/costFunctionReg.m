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

temp = X*theta
h = sigmoid(temp)
for i=1:m
    J = J+1/m*(sum(-log(h(i))*y(i))-log(1-h(i))*(1-y(i)));
end
J
for i =2:size(theta,1)
    J= J + lambda/2/m*(theta(i)*theta(i));
end
J
updates = X' * (h - y);
grad(1) =  (1/m) * updates(1);

for i =2:size(theta,1)
    grad(i)= (1/m) * updates(i) + lambda/m*theta(i);
end


% =============================================================

end
