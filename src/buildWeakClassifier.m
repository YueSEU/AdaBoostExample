%% Function buildWeakClassifier:
%
% This function creates the weak classifiers.
% Each weak classifier is defined by three numbers:
%
% 1- threshold: threshold value that defines the frontier between two
% classes with respect to one of the data features.
% 2- feature: number that indicates the feature of the data in which
% the threshold should be applied.
% 3- direction: number that indicates where each class is located with
% respect to the threshold value. A direction of value "1" indicates that
% the class labeled as "1" presents a feature value that is equal to or 
% bigger than the threshold value. Otherwise, label "1" is applied to the
% class that presents a feature value smaller than the threshold value.
%
% Parameters:
%
% dataFeatures: a matrix containing the features that characterize the
% samples in the training dataset. Each column represents one feature, 
% each line contains all the feature values that characterize a sample.
%
% dataClass: a vector containing the class label (1 or -1) of each sample
% in the training dataset. Must contain labels for all of the samples in
% the training dataset.
%
% dataWeight: a vector containing the weights of each training sample.
% These weights are adjusted each time a weak classifier is added to the
% AdaBoost classifier to increase the influence of missclassified samples
% in the training process and reduce the influence of samples that the
% AdaBoost classifier can already correctly classify.
% 
% Returns:
%
% weakClassifier: the weak classifier, as defined above, that has the
% smallest weighted error in the training dataset.
%
% predictedClass: the class labels assigned by the weak classifier to the
% training dataset.
%
% error: the classification error of the weak classifier.

function [weakClassifier, predictedClass, error] = ...
    buildWeakClassifier(dataFeatures, dataClass, dataWeight)

    % Find the maximum and minimum values presented by 
    % each feature in the training data.

    maxFeatureValue = max(dataFeatures, [], 1) + realmin;
    minFeatureValue = min(dataFeatures, [], 1) - realmin;
     
    % For each feature, find the threshold that best classifies the 
    % training data and build the weak classifier using the threshold
    % that classifies the training data with the smallest error.
    
    % Initialize variables that will store the values of the error,
    % threshold, feature and direction of the best weak classifier.
    error = realmax;
    weakClassifier.threshold = 0;
    weakClassifier.feature = 1;
    weakClassifier.direction = 0;
    
    % For all features...
    for feature = 1:ndims(dataFeatures)
        % Generate 1000 threshold values.
        thresholdCandidates = linspace(minFeatureValue(feature), ...
            maxFeatureValue(feature), 1000);
        % For each threshold candidate...
        for i = 1:length(thresholdCandidates)
            % Classify training samples using the positive direction.
            estimatedClass(:, 1) = ...
                double(dataFeatures(:, feature) >= thresholdCandidates(i));
            % Classify training samples using the negative direction.
            estimatedClass(:, 2) = ...
                double(dataFeatures(:, feature) < thresholdCandidates(i));
            % Replace zeros with the label "-1".
            estimatedClass(estimatedClass == 0) = -1;
            % Calculate error for this threshold in both directions.
            candidateError(1) = ...
                sum((estimatedClass(:, 1) ~= dataClass) .* dataWeight) / ...
                sum(dataWeight);
            candidateError(2) = ...
                sum((estimatedClass(:, 2) ~= dataClass) .* dataWeight) / ...
                sum(dataWeight);
            % Store minimum error and direction of minimum error
            % for this threshold candidate.
            [thrError(i) thrDir(i)] = min(candidateError);
        end
        % Find the threshold with minimum error for this feature.
        [minError indexCandidateMinError] = min(thrError);
        % If this error is less than the current minimum, then save the
        % threshold value, dimension, direction and error.
        if(error > minError)
            error = minError;
            weakClassifier.threshold = ...
                thresholdCandidates(indexCandidateMinError);
            weakClassifier.feature = feature;
            weakClassifier.direction = thrDir(indexCandidateMinError);
        end
        % Clear variables for next iteration.
        clear thresholdCandidates estimatedClass candidateError thrError;
        clear thrDir minError indexCandidateMinError;
        
    end
    
    % Classify training data using new weak classifier.
    predictedClass = applyWeakClassifier(weakClassifier, dataFeatures);

end