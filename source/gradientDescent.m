function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters,
	g=0;
	k=0;
	t=0;
	for i = 1:m,
		t = theta(1) + theta(2)*X(i,2) - y(i);
		g = g + t;			%summation term for theta1
		k = k + t*X(i,2);		%summation term for theta2
	end
	theta(1) = theta(1) - (alpha/m)*g;
	theta(2) = theta(2) - (alpha/m)*k;
	
	
	%OR
	%predict_ans = X*theta-y;
	%theta(1) = theta(1) - (alpha/m)*sum(predict_ans);
	%theta(2) = theta(2) - (alpha/m)*sum((predict_ans).*X(:,2));

	% Save the cost J in every iteration    
	J_history(iter) = computeCost(X, y, theta);
    
end

%plot([1:1:1500],J_history,'r', 'MarkerSize', 10);
%xlabel('Iterations');
%ylabel('J(0)');
%pause;
end
