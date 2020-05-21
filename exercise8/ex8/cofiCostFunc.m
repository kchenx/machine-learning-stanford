function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
    %COFICOSTFUNC Collaborative filtering cost function
    %   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    %   num_features, lambda) returns the cost and gradient for the
    %   collaborative filtering problem.
    %
    % Notes: X - num_movies  x num_features matrix of movie features
    %        Theta - num_users  x num_features matrix of user features
    %        Y - num_movies x num_users matrix of user ratings of movies
    %        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
    %            i-th movie was rated by the j-th user
    %

    % Unfold the U and W matrices from params
    X = reshape(params(1:num_movies*num_features), num_movies, num_features);
    Theta = reshape(params(num_movies*num_features+1:end), ...
                    num_users, num_features);

    % Compute cost function and derivatives
    A = (X * Theta' - Y) .* R;
    J = 1 / 2 * sum(A(:).^2) + lambda / 2 * (sum(Theta(:).^2) + sum(X(:).^2));
    X_grad = A * Theta + lambda * X;
    Theta_grad = A' * X + lambda * Theta;

    % Fold gradient
    grad = [X_grad(:); Theta_grad(:)];

end
