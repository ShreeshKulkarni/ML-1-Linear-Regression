function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2); % number of features
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
	
	theta_new = theta;
	predict_ans = X*theta-y;
	for i=1:n,
		theta_new(i) = theta(i) - (alpha/m) *sum((predict_ans).*X(:,i));
 	end;

	theta = theta_new;

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
