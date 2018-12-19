%% Machine Learning Online Class - Exercise 4 Neural Network Learning
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions 
%  in this exericse:
%
%     sigmoidGradient.m
%     randInitializeWeights.m
%     nnCostFunction.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('vowels.mat');
dimensions=size(X, 2);
m = size(X, 1);
input_layer_size  = dimensions; 
num_labels = 2; % 2 labels, from 0 and 1
alpha=1;
hidden_layer_size =round(m/(alpha*(input_layer_size+num_labels)))% 25 hidden units /neurons in hidden layer   
                % (note that we have mapped "0" to label 10)
hidden_layer_size=6;
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%Pca implementation
z=pca_x(X);

%shuffle randomly all row
X=[X y];
X= X(randperm(size(X,1)),:);
y= X(:,dimensions+1);
X=X(:,1:dimensions);
data_map(z,y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%pick 70% of the data as training data and 30% of the data as test data 
[X_train, X_test, y_train, y_test]=divide(X,y);
%%% shuffle x_train and y_train again
X_train=[X_train y_train];
X_train= X_train(randperm(size(X_train,1)),:);
y_train= X_train(:,dimensions+1);
X_train=X_train(:,1:dimensions);
%%%shuffle x_test and y_test again
X_test=[X_test y_test];
X_test= X_test(randperm(size(X_test,1)),:);
y_test= X_test(:,dimensions+1);
X_test=X_test(:,1:dimensions);
%% ================ Part 2: Loading Parameters ================
% In this part of the exercise, we load some pre-initialized 
% neural network parameters.

fprintf('\nLoading Saved Neural Network Parameters ...\n')

% Load the weights into variables Theta1 and Theta2
Theta1=zeros(hidden_layer_size,dimensions+1);
Theta2=zeros(num_labels,hidden_layer_size+1);

% Unroll parameters 
nn_params = [Theta1(:) ; Theta2(:)];

%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
fprintf('\nFeedforward Using Neural Network ...\n')

% Weight regularization parameter (we set this to 0 here).
lambda = 0;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X_train, y_train, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =============== Part 4: Implement Regularization ===============
%  Once your cost function implementation is correct, you should now
%  continue to implement the regularization with the cost.
%

fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

% Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                   num_labels, X_train, y_train, lambda);

fprintf(['Cost at parameters (loaded from ex4weights): %f '...
         '\n'], J);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================ Part 6: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
init =sum(sum(initial_nn_params));
fprintf(['\n\n init: %f ' ...
         ], init);
fprintf('\nProgram paused. Press enter to continue.\n');
pause;
%% =============== Part 7: Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

fprintf('\nChecking Backpropagation (w/ Regularization) ... \n')

%  Check gradients by running checkNNGradients
lambda = 3;
checkNNGradients(lambda);

% Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X_train, y_train, lambda);

fprintf(['\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f ' ...
         '\n(for lambda = 3, \n\n'], lambda, debug_J);

fprintf('Program paused. Press enter to continue.\n');
pause;


%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 200);

%  You should also try different values of lambda
lambda=0.2;
result=zeros(10,3);
i=1;
ideal_lambda=0;
gap=0.1;
while lambda<1
disp(lambda)
% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
%pause;

% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

%disp(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X_train);
check=convert(pred);
%calculation part
outliers_number=nnz(check);%non zero elements
actual_outlier_number=nnz(y_train); %actual outliers from the y matrix 
[hit_rate outlier_accuracy]=calculations(y_train,check,m,outliers_number,actual_outlier_number);
result(i,:)=[lambda hit_rate outlier_accuracy];
fprintf('\n');
lambda=lambda+gap;
i=i+1;
end
%%working
new=abs(result(:,3)-100);
[minval, row] = min(min(new,[],2));
ideal_lambda=result(row,1)
%plot the optimum lambda
fprintf(['results of the data(lambda hit_rate accuracy on train data ): %f '...
         '\n'],result(row,:) );
fprintf('Program paused. Press enter to continue.\n');
pause;
figure(1);
plot(result(:,1),result(:,2), 'rx','MarkerSize', 20)
hold on;
plot(result(:,1),result(:,3),'bx','MarkerSize', 20)
hold on;
legend('hit rate','outlier accuracy','ideal value')
ideal=ones(size(result,1),1)*100;
[row col]=find(result(:,1)==0);
row=row(1);
ideal=ideal(1:row-1,:);
result=result(1:row-1,:);
plot(result(:,1),result(:,2),'-r', 'LineWidth', 2);
hold on;
plot(result(:,1),result(:,3),'-b', 'LineWidth', 2);
hold on;
plot(result(:,1),ideal, '-y', 'LineWidth', 2);
title('hit rate vs outlier detection trade-off wrt lambda');
legend('hit rate','outlier accuracy','ideal value')
xlabel('lambda')
ylabel('accuracies')
fprintf('\nProgram paused. Press enter to continue.\n');
%%%%%%%%%%%%%%%%%%%%%%%run for ideal lambda%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_train, y_train, ideal_lambda);
                                   
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
%pause;
pred = predict(Theta1, Theta2, X_test);
check=convert(pred);
%calculation part
outliers_number=nnz(check);
[hit_rate outlier_accuracy]=calculations(y_test,check,m,outliers_number,actual_outlier_number);
%pause;
#{
%%%%%%%%%%%%%%%%%%%%%%%%cross validation approach%%%%%%%%%%%%%%%%%%%%%%%
%pick the cross_validation set split set for 5
fprintf('\nworking with cross validation.\n');
splits=5;
[part_1,part_2,part_3,part_4,part_5]=cross_validation(X,y,splits);

%split columns into X and y from parts
dimensions_1 = size(part_1, 2);
dimensions_2 = size(part_2, 2);
dimensions_3 = size(part_3, 2);
dimensions_4 = size(part_4, 2);
dimensions_5 = size(part_5, 2);
part_1x=part_1(:,1:dimensions_1-1);
part_2x=part_2(:,1:dimensions_2-1);
part_3x=part_3(:,1:dimensions_3-1);
part_4x=part_4(:,1:dimensions_4-1);
part_5x=part_5(:,1:dimensions_5-1);
part_1y=part_1(:,dimensions_1);
part_2y=part_2(:,dimensions_2);
part_3y=part_3(:,dimensions_3);
part_4y=part_4(:,dimensions_4);
part_5y=part_5(:,dimensions_5);

%%%%%test accuracy on all crossvalidation sets%%%%%%%%
accuracy_cross=crossvalidation_test(part_1x,part_2x,part_3x,part_4x,part_5x,
part_1y,part_2y,part_3y,part_4y,part_5y,splits);
accuracy_cross=mean(accuracy_cross)



