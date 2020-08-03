%% Function trainAdaBoost:
%
% The AdaBoost classifier is composed of a group of weak classifiers
% that are selected using a boosting process.
%
% A weak classifier finds the best threshold in one of the features of
% the data to separate the training examples into two classes, according
% to the current weights of the training samples. For more details, see 
% the description of the buildWeakClassifier function.
%
% The boosting process is used to select, among the pool of possible weak
% classifiers, the one that most contributes to the performance of the
% AdaBoost classifier. It does that by adjusting the weights of the
% training samples.
%
% At each iteration, a weak classifier is selected based on the current set
% of weights attributed to the training samples. After finding the weak
% classifier and attributing to it a weight "alpha", the weights of 
% missclassified examples are increased and the weights of correctly 
% classified examples are decreased. Based on this new set of weights, 
% the function will select another weak classifier.
%
% At the end, the adaBoost classifier will be composed of a group
% of weak classifiers that together behave as a strong classifier.
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
% numberOfIterations: the number of training iterations, which is equal to
% the number of weak classifiers added to the adaboost classifier.
%
% Returns:
%
% adaboostClassifier: a struct containing a group of weak classifiers,
% their weights alpha and errors.
%
% predictedClass: a vector containing the class label (1 or -1) of each
% sample, as assigned by the adaboostClassifier.

function [adaboostClassifier, predictedClass] = ...
    trainAdaBoost(dataFeatures, dataClass, numberOfIterations)

    % Initialize variable to hold the AdaBoost classifier.
    adaboostClassifier = struct;
    
    % Create variable to hold the sum of the predictions of each weak 
    % classifier multiplied by their respective alphas.
    alphaWeakClassPredicSum = zeros(size(dataClass));
    
    % Create variable to hold the weight of each training sample.
    % For the first iteration, each sample is equally important,
    % so all weights are equal.
    dataWeights = ones(length(dataClass), 1) / length(dataClass);
    
    % Execute training iterations
    for i = 1:numberOfIterations
        % Find the best weak classifier to separate the training
        % data in two classes, using the current set of weights
        % assigned to the training data.
        [weakClassifier, weakClassPredic, error] = ...
            buildWeakClassifier(dataFeatures, dataClass, dataWeights);
        
        % Calculate the weight alpha of this weak classifier,
        % based on the current classification error.
        alpha = 1/2 * (log((1 - error) / max(error, eps)));
        
        % Save the model parameters of this weak classifier.
        adaboostClassifier(i).threshold = weakClassifier.threshold;
        adaboostClassifier(i).feature = weakClassifier.feature;
        adaboostClassifier(i).direction = weakClassifier.direction;
        adaboostClassifier(i).alpha = alpha;
        
        % Calculate the current error of the AdaBoost classifier.
        alphaWeakClassPredicSum = alphaWeakClassPredicSum + ...
            adaboostClassifier(i).alpha * weakClassPredic;
        predictedClass = sign(alphaWeakClassPredicSum);
        adaboostClassifier(i).error = ...
            sum(predictedClass ~= dataClass) / length(dataClass);
        
        % Update weights of the training data, so that missclassified
        % examples will have more weight in the next iteration.
        dataWeights = dataWeights .* ...
            exp(-dataClass .* adaboostClassifier(i).alpha .* weakClassPredic);
        dataWeights = dataWeights ./ sum(dataWeights);
        
        % If the error is zero, stop the training process.
        if(adaboostClassifier(i).error == 0)
            break; 
        end
    end

end