# five techniques to prevent overfitting while training neural networks.

1. Simplifying The Model
 
The first step when dealing with overfitting is to decrease the complexity of the model. To decrease the complexity, we can simply remove layers or reduce the number of neurons to make the network smaller. While doing this, it is important to calculate the input and output dimensions of the various layers involved in the neural network. There is no general rule on how much to remove or how large your network should be. But, if your neural network is overfitting, try making it smaller.

 

2. Early Stopping
 
Early stopping is a form of regularization while training a model with an iterative method, such as gradient descent. Since all the neural networks learn exclusively by using gradient descent, early stopping is a technique applicable to all the problems. This method update the model so as to make it better fit the training data with each iteration. Up to a point, this improves the model’s performance on data on the test set. Past that point however, improving the model’s fit to the training data leads to increased generalization error. Early stopping rules provide guidance as to how many iterations can be run before the model begins to overfit.

## 3. Data-Augmentation
data augmentation simply means increasing size of the data that is increasing the number of images present in the dataset. Some of the popular image augmentation techniques are flipping, translation, rotation, scaling, changing brightness, adding noise etcetera. using data augmentation a lot of similar images can be generated. This helps in increasing the dataset size and thus reduce overfitting. The reason is that, as we add more data, the model is unable to overfit all the samples, and is forced to generalize.

4. Use Regularization

Regularization is a technique to reduce the complexity of the model. It does so by adding a penalty term to the loss function. The most common techniques are known as L1 and L2 regularization:

The L1 penalty aims to minimize the absolute value of the weights. This is mathematically shown in the below formula.

The L2 penalty aims to minimize the squared magnitude of the weights. This is mathematically shown in the below formula.

So which technique is better at avoiding overfitting? The answer is — it depends. If the data is too complex to be modelled accurately then L2 is a better choice as it is able to learn inherent patterns present in the data. While L1 is better if the data is simple enough to be modelled accurately. For most of the computer vision problems that I have encountered, L2 regularization almost always gives better results. However, L1 has an added advantage of being robust to outliers. So the correct choice of regularization depends on the problem that we are trying to solve.

5. Use Dropouts
 
Dropout is a regularization technique that prevents neural networks from overfitting. Regularization methods like L1 and L2 reduce overfitting by modifying the cost function. Dropout on the other hand, modify the network itself. It randomly drops neurons from the neural network during training in each iteration. When we drop different sets of neurons, it’s equivalent to training different neural networks. The different networks will overfit in different ways, so the net effect of dropout will be to reduce overfitting.

used to randomly remove neurons while training of the neural network. This technique has proven to reduce overfitting to a variety of problems involving image classification, image segmentation, word embeddings, semantic matching etcetera.

6. Batch Normallization

Batch normalization provides an elegant way of reparametrizing almost any deep network. The reparametrization significantly reduces the problem of coordinating updates across many layers
technique to help coordinate the update of multiple layers in the model.

Batch normalization acts to standardize only the mean and variance of each unit in order to stabilize learning, but allows the relationships between units and the nonlinear statistics of a single unit to change.

We adopt batch normalization (BN) right after each convolution and before activation …

## tips for Using Batch Normalization
This section provides tips and suggestions for using batch normalization with your own neural networks.

Use With Different Network Types
Batch normalization is a general technique that can be used to normalize the inputs to a layer.

It can be used with most network types, such as Multilayer Perceptrons, Convolutional Neural Networks and Recurrent Neural Networks.

Probably Use Before the Activation
Batch normalization may be used on the inputs to the layer before or after the activation function in the previous layer.

It may be more appropriate after the activation function if for s-shaped functions like the hyperbolic tangent and logistic function.

It may be appropriate before the activation function for activations that may result in non-Gaussian distributions like the rectified linear activation function, the modern default for most network types.

## Don’t Use With Dropout
Batch normalization offers some regularization effect, reducing generalization error, perhaps no longer requiring the use of dropout for regularization.

Removing Dropout from Modified BN-Inception speeds up training, without increasing overfitting.

— Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015.

Further, it may not be a good idea to use batch normalization and dropout in the same network.

The reason is that the statistics used to normalize the activations of the prior layer may become noisy given the random dropping out of nodes during the dropout procedure.

Batch normalization also sometimes reduces generalization error and allows dropout to be omitted, due to the noise in the estimate of the statistics used to normalize each variable.


## How to use Deep Learning when you have Limited Data


https://nanonets.com/blog/data-augmentation-how-to-use-deep-learning-when-you-have-limited-data-part-2/
