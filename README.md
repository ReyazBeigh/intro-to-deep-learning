# Intro To Deep Learning - Learning the use of Tensorflow and Keras

## Single Neuron 
- Deep learning is an approach to machine learning characterized by deep stacks of computation. the depth of computation has enabled deep learning models to break the complex and challenging real world data sets. Single Neuron is a linear unit (building block) that can have some input(s) and outup(s). the formulla `y = wx+b` depicts the working of a neuron. while w in the input x is the qty of input and b is bias value and y is the output. In case of mutliple inputs the formulla will look like this: `y = w0x0 + w1x1 + w2x2 + w3x3 + b`. while w0,w1,w2,w3 are different inputs.  Easiest way to create models in Keras is Keras.Sequential. 

## Deep Neural Networks 
- Building up complex network from simpler functional units. Neural Networks are orgnaise their neurons into layers, the liner units having common inputs are collected together into dense layers. There can be multiple layers and each layer is a transformation that takes us closer to the solution. An activation function is some function that is applied to the layer's outputs(its activations). the most common is rectifier function `max(0,X)`. Rectifier function has a graph that has the -ve values rectified to Zero. When we attach the rectifier to a linear unit, we get a rectified linear unit or ReLU. ReLU application formulla `max(0 , w * x + b )`

## Stochastic Gradient Descent
- We compile model by adding an optimizer and loss function. Optimiser is the function that tells how to solve the problem and example function is `Adam`. Loss function as we know helps in validation and gives us an idea about the accuracy and an exmample is `mae`. There are multiple Optimizer functions and all of them virutally belong to Stochastic Gradient Descent. They peform their job in 3 steps `1 - Sample some traning data and run it though the network to make predictions` `2 - Measure the loss between prediction and true value` `3 - Finally adjust the weights in a direction to make the losses smaller`, then just repeat 1-2 over and over again untill the loss is smaller or won't decrease further.

## Overfitting and Underfitting
- We need to understand the learning curves. Essential part of the curves is Signal and Noise, Single is the part that generalizes and helps our model to predict from new data, while Noise is the random fluctuation that comes from the data in the real world and don't help in making the predictions. The Noise part might sometimes look useful but isn't. `Underfitting the training set is when the loss is not as low as it could be because the model hasn't learned enough signals`, `Overfitting the training set is when the loss is not as low as it could be because the model has learned too much noise`. The trick to train deep learning models is finding the best balance in between the two. `Capacity` referrs to the size and complexity a model is able to learn. For neural networks it depends on how many neurons are there and how they are connected to each other. Capacity of a network can be increased in 2 ways. Making a network `wider` by increasing number of inputs to the existing laryers. Or making a Network `deeper` by adding new hidden layers. wider networks has easier time learning more linear relationships while depper networks prefer more non liner ones

## Dropout and Batch Normalisation 
- Dropout is removing some units from the network layer to correct overfitting, as we know overfitting is caused by the learning too much noise. droppping out some units from the layers prevents learning the noise. We add dropout before a hidden layer and tell the model how many units to remove. Batch Normalisation or `batchnorm` helps correcting the training data that is slow or unstable. Batch Normalisation layers is added after a hidden layer.Most often, batchnorm is added as an aid to the optimization process (though it can sometimes also help prediction performance). It helps models to complete training with few epochs and fix various problems that cause training to "stuck".

## Binary Classification 
- `Classification` is another machine learning problem and the difference is in loss function and final output layer that the model produces.Classification into one of two classes is a common machine learning problem. one might want to predict if a consumer is likely to make a purchase or not. whether or not a credit card transaction was fradulent. In the raw data the class might be represented by a string `Yes` or `No`, `Dog` and `Cat`, before using them we assing a class label. one class will be 0 and another 1, assigning these numerical values makes the nueral network to use it.
Accuracy is one othe many metrics in use for measuring success on a classification problem. `Accuracy = correct_predictions_sum / total_prediction`. A model that always predicted correctly would have an accuracy score of 1.0. Accuracy and some other classical metrics can't be used as loss functions. a substitue used here is class `cross-entopy` function. `Cross-entropy` is a sort of measure for the distance from one probability distribution to another. use cross-entropy for a classification loss; other metrics you might care about (like accuracy) will tend to improve along with it.
The cross-entropy and accuracy functions both require probabilities as inputs, meaning, numbers from 0 to 1. To covert the real-valued outputs produced by a dense layer into probabilities, we attach a new kind of activation function, the `sigmoid activation`.