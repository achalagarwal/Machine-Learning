function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01;
sigma = 0.01;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
p=0;q=0;
val = intmax;
for i=0:9
    for j=0:9
        model= svmTrain(X, y, C*(3^i), @(x1, x2) gaussianKernel(x1, x2, sigma*(3^j)));
        predictions = svmPredict(model,Xval);
        if(mean(double(predictions ~= yval))<val)
            p=i;
            q=j;
            val =  mean(double(predictions ~= yval));
        end
    end
end
C = C*(3^p);
sigma = sigma*(3^q);




% =========================================================================

end
