% Example for trainAdaBoost.m

% Clear previous data.
clear all;
close all;
clc;


% Create training data with two classes.
% The "blue" class is defined as the group of points inside of
% a square centered at the origin and with side 10.
% The "red" class is defined as the group of points inside of
% a circle centered at the origin, with radius 10,
% surrounding the blue square.
blue = [(-5 + 10 .* rand(200, 1)) (-5 + 10 .* rand(200, 1))];
angle = rand(200,1) * 2 * pi;
radius = rand(200,1) * 5 + 5;
red = [(sin(angle) .* radius) (cos(angle) .* radius)];
clear angle radius;

% Create matrices holding the feature and class
% information of the training data.
datafeatures = [blue; red];
dataclass(1:200) = 1;
dataclass(201:400) = -1;

% Show training data.
figure, subplot(2, 2, 1), hold on, axis equal;
plot(blue(:, 1), blue(:, 2), 'b.');
plot(red(:, 1), red(:, 2), 'r.');
title('Training Data');

% Train the AdaBoost classifier.
[adaboostClassifier, predictedClass] = ...
    trainAdaBoost(datafeatures, dataclass', 20);

% Retrieve training results.
blue = datafeatures(predictedClass == 1, :);
red = datafeatures(predictedClass == -1, :);

% Show training results.
subplot(2, 2, 2), hold on, axis equal;
plot(blue(:, 1), blue(:, 2), 'b.');
plot(red(:, 1), red(:, 2), 'r.');
title('Training Data Classified Using AdaBoost Classifier');

% Show how the classification error changes as
% weak classifiers are added to the model.
error = zeros(1, length(adaboostClassifier)); 
for i = 1:length(adaboostClassifier)
    error(i) = adaboostClassifier(i).error;
end
subplot(2, 2, 3), plot(error), grid on;
title('Classification Error VS. Number of Weak Classifiers');

% Create test data.
angle = rand(300,1) * 2 * pi;
radius = -10 + 20 .* rand(300, 1);
testdata = [(sin(angle) .* radius) (cos(angle) .* radius)];
clear angle radius;

% Classify the testdata with the trained model.
testclass = applyAdaBoost(adaboostClassifier, testdata);

% Retrieve test results.
blue = testdata(testclass == 1, :);
red = testdata(testclass == -1, :);

% Show test results.
subplot(2, 2, 4), hold on, axis equal;
plot(blue(:, 1), blue(:, 2), 'b.');
plot(red(:, 1), red(:, 2), 'r.');
title('Test Data Classified Using AdaBoost Classifier');