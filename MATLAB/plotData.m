%Arpan Bag

function plotData(X, Y, min_x, max_x)
%	PLOTDATA Plots the data points X and Y into a new figure 
%   PLOTDATA(x, Y, min_x, max_x) plots the data points with + for the positive examples. Parameters min_x and max_x decide the scale.
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ========================= CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%


% Find Indices of Positive and Negative Examples 
pos = find(Y==1); neg = find(Y == 0);

% Plot Examples 
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(neg, 1), X(neg, 2), 'ko','MarkerFaceColor', 'y', 'MarkerSize', 7);


% Put some labels

% Labels and Legend
xlabel('Feature 1 (x1)')
ylabel('Feature 2 (x2)')

% Specified in plot order
legend('y = 1', 'y = 0')
axis([min_x, max_x, min_x, max_x])
hold off;

% =========================================================================

end
