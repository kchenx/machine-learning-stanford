function [C, sigma] = dataset3Params(X, y, Xval, yval)
    %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
    %where you select the optimal (C, sigma) learning parameters to use for SVM
    %with RBF kernel
    %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
    %   sigma. You should complete this function to return the optimal C and 
    %   sigma based on a cross-validation set.
    %

    C = 0;
    sigma = 0;
    minerror = 1;
    
    % find min error among 64 options for C-sigma pairs
    vars = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    for Ci = vars
        for sigmaj = vars
            model = svmTrain(X, y, Ci, @(x1, x2) gaussianKernel(x1, x2, sigmaj));
            predictions = svmPredict(model, Xval);
            error = mean(double(predictions ~= yval));
            if error < minerror
                C = Ci;
                sigma = sigmaj;
                minerror = error;
            end
        end
    end
    
end
