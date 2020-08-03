%% Function applyAdaBoost:
%
% When analyzing a sample, each weak classifier that composes the 
% adaboost classifier will attribute to the sample a class 
% according to its own threshold value. The final class is given by the
% sign of the sum of the results of each weak classifier multiplied 
% by their respective weights "alpha".
%
% Parameters:
%
% adaboostClassifier: as in the description of the trainAdaBoost function.
%
% dataFeatures: as in the description of the trainAdaBoost function.
%
% Returns:
%
% predictedClass: a vector containing the class label (1 or -1) of each
% sample, as assigned by the adaboostClassifier.


function predictedClass = ...
    applyAdaBoost(adaboostClassifier, dataFeatures)
    
    % To classify each test sample, first add all the results of each
    % weak classifier weighted by their respective alpha.
    predictedClass = zeros(size(dataFeatures, 1), 1);
    for i=1:length(adaboostClassifier);
        predictedClass = predictedClass + ...
            (adaboostClassifier(i).alpha * ...
            applyWeakClassifier(adaboostClassifier(i), dataFeatures));
    end
    % The final class of each test sample is given by
    % the sign of the previous sum.
    predictedClass = sign(predictedClass);

end