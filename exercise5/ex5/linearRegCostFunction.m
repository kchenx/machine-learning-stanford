function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
    %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
    %regression with multiple variables
    %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
    %   cost of using theta as the parameter for linear regression to fit the
    %   data points in X and y. Returns the cost in J and the gradient in grad
    
    m = length(y); % number of training examples
    
    J = 1 / (2 * m) * (X * theta - y)' * (X * theta - y) ...
        + lambda / (2 * m) * theta(2:end)' * theta(2:end);
    
    grad = 1 / m * X' * (X * theta - y) + lambda / m * theta;
    grad(1) = 1 / m * X(:,1)' * (X * theta - y);
    
end
