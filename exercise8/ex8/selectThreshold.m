function [bestEpsilon, bestF1] = selectThreshold(yval, pval)
    %SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
    %outliers
    %   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
    %   threshold to use for selecting outliers based on the results from a
    %   validation set (pval) and the ground truth (yval).
    %

    bestF1 = 0;

    stepsize = (max(pval) - min(pval)) / 1000;
    for epsilon = min(pval):stepsize:max(pval)
        predictions = (pval < epsilon);

        truepos = sum((predictions == 1) & (yval == 1));
        falsepos = sum((predictions == 1) & (yval == 0));
        falseneg = sum((predictions == 0) & (yval == 1));

        precision = truepos / (truepos + falsepos);
        recall = truepos / (truepos + falseneg);
        F1 = 2 * precision * recall / (precision + recall);

        if F1 > bestF1
           bestF1 = F1;
           bestEpsilon = epsilon;
        end
    end

end
