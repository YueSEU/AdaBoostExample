%% Function applyWeakClassifier:
%
% This function uses a weak classifier to assign labels to a sample of data
% based on its features.
%
% Parameters:
%
% weakClassifier: as in the description of the buildWeakClassifier function.
%
% dataFeatures: as in the description of the buildWeakClassifier function.
%
% Returns:
%
% predictedClass: a vector containing the class label (1 or -1) of each
% sample, as assigned by the weak classifier.

function predictedClass = applyWeakClassifier(weakClassifier, dataFeatures)

    % If the weak classifier's direction is equal to "1", then all samples
    % with a value equal to or above the weak classifier's threshold
    % in the weak classifier's feature are assigned the label "1".
    if(weakClassifier.direction == 1)
        predictedClass =  double(dataFeatures(:, weakClassifier.feature) >= ...
            weakClassifier.threshold);
    % If the weak classifier's direction is not "1", then all samples
    % with a value smaller than the weak classifier's threshold in the
    % weak classifier's feature are assigned the label "1".
    else
        predictedClass =  double(dataFeatures(:, weakClassifier.feature) < ...
            weakClassifier.threshold);
    end
    % Replace zeros with the label "-1".
    predictedClass(predictedClass == 0) = -1;

end