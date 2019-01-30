%% Initialization
%% Clear and Close Figures
clear ; close all; clc


fprintf('*****Logistic Regression*******\n\n\n');
fprintf('Press enter to select the Input data file.\n');
pause;

%% Load Data

%Open the file selection dialogue
[inputFileName,inputFilePath] = uigetfile({
   '*.txt','Text (*.txt)'; ...
   '*.*',  'All Files (*.*)'}, ...
   'Select the Data file');
   
fprintf('\nSelected file: %s \nPress Enter to load data.\n',inputFileName);
pause;
   
fprintf('Loading data ...\n');
data = load([inputFilePath '\' inputFileName]);	%The file containing the training data. The Last column is the output (Y) and the rest of the columns are input (X) 
X = data(:,1:end-1);	%Inputs
Y = data(:,end);		%Outputs
num_features = size(X,2); %No of features, which is the dimension of X
min_x = min(min(X));
max_x = max(max(X));


% Print out some data points
fprintf('\nData loaded.\nFirst 10 examples from the dataset: \n');
disp([X(1:10,:),Y(1:10,:)])

if num_features == 2 %Check if the input is two dimensional
	%Plot data
	fprintf('\nPress enter to plot data.\n');
	pause;
	plotData(X, Y, min_x, max_x);
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Regularized Logistic Regression ==================
%  In this part, suppose are given a dataset with data points that are not
%  linearly separable. However, we would still like to use logistic
%  regression to classify the data points.
%
%  To do so, introduce more features to use -- in particular, add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
polynomial_degree = 2;
X = mapFeature(X(:,1), X(:,2),polynomial_degree);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, Y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nPress enter to start training.\n');
pause;


% Set Options
num_iters = 400;
options = optimset('GradObj', 'on', 'MaxIter', num_iters);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, Y, lambda)), initial_theta, options);

% Plot Boundary
fprintf('\nTraining complete.\n');
fprintf('\nPress enter to plot decision boundary.\n');
pause;
plotDecisionBoundary(theta, X, Y, min_x, max_x, polynomial_degree);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('Feature 2 (x2)')
ylabel('Feature 2 (x2)')

legend('y = 1', 'y = 0', 'Decision boundary')
hold off;

fprintf('\nPress enter to start evaluation.\n');
pause;


% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == Y)) * 100);
























