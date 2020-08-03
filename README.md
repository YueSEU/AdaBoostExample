# AdaBoost Example

Implementation of the AdaBoost classifier using Matlab.

![Classification Example](https://github.com/gus-c-oliveira/AdaBoostExample/blob/master/Example%20Results.png)

The AdaBoost classifier is composed of a group of weak classifiers that are selected using a boosting process.

A weak classifier finds the best threshold in one of the features of the data to separate the training examples into two classes, according to the current weights of the training samples. For more details, see the description of the buildWeakClassifier function.

The boosting process is used to select, among the pool of possible weak classifiers, the one that most contributes to the performance of the AdaBoost classifier. It does that by adjusting the weights of the training samples.

At each iteration, a weak classifier is selected based on the current set of weights attributed to the training samples. After finding the weak classifier and attributing to it a weight "alpha", the weights of missclassified examples are increased and the weights of correctly classified examples are decreased. Based on this new set of weights, the function will select another weak classifier.

At the end, the adaBoost classifier will be composed of a group of weak classifiers that together behave as a strong classifier.
