function plotDecisionBoundary(theta, X, Y, min_x, max_x, polynomial_degree)
%PLOTDECISIONBOUNDARY Plots the data points X and Y into a new figure with
%the decision boundary defined by theta
%   PLOTDECISIONBOUNDARY(theta, X, Y, min_x, max_x, polynomial_degree) plots the data points with + for the 
%   positive examples and o for the negative examples.
%	Parameters min_x and max_x decide the scale.

% Plot Data
plotData(X(:,2:3), Y,min_x,max_x);
hold on

% Here is the grid range
u = linspace(min_x, max_x, 50);
v = linspace(min_x, max_x, 50);

z = zeros(length(u), length(v));
% Evaluate z = theta*x over the grid
for i = 1:length(u)
	for j = 1:length(v)
		z(i,j) = mapFeature(u(i), v(j),polynomial_degree)*theta;
	end
end
z = z'; % important to transpose z before calling contour

% Plot z = 0
% Need to specify the range [0, 0]
contour(u, v, z, [0, 0], 'LineWidth', 2)

hold off

end
