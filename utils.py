import os, pickle
import numpy as np
import cv2

class Softmax(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
          # lazily initialize W
          self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            self.W = self.W - learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        
        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = scores.argmax(axis=1)
        return y_pred
  
    def loss(self, X, y, reg):
        """
        Compute the softmax loss function and its derivative. 
        Inputs:
        - X: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)

        # Compute the softmax loss and its gradient.
        # Store the loss in loss and the gradient in dW.
        num_train = X.shape[0]
        scores = X.dot(self.W)
        scores = scores - np.max(scores, axis=1, keepdims=True)

        # Softmax Loss
        sum_exp_scores = np.exp(scores).sum(axis=1, keepdims=True)
        softmax_matrix = np.exp(scores)/sum_exp_scores
        loss = np.sum(-np.log(softmax_matrix[np.arange(num_train), y]) )

        # Weight Gradient
        softmax_matrix[np.arange(num_train),y] -= 1
        dW = X.T.dot(softmax_matrix)

        # Average
        loss /= num_train
        dW /= num_train

        # Regularization
        loss += reg * np.sum(self.W * self.W)
        dW += reg * 2 * self.W

        return loss, dW

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, "rb") as f:
        datadict = pickle.load(f, encoding="latin1")
        X = datadict["data"]; Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
        return X, Y
    
def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []; ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X); ys.append(Y)
    Xtr = np.concatenate(xs); Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    return Xtr, Ytr, Xte, Yte

def get_CIFAR10_data(cifar10_dir='data/cifar-10-batches-py', num_training=45000, num_validation=5000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.  
    """
    
    # Cleaning up variables to prevent loading data multiple times (which may cause memory issue)
    try:
        del X_train, y_train
        del X_val, y_val
        del X_test, y_test
        print('Clear previously loaded data.')
    except:
        pass

    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
    
    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Print shapes
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape) 
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    

    return X_train, y_train, X_val, y_val, X_test, y_test


def train(X_train, y_train, X_val, y_val, X_test, y_test, regularization_strengths):

    # Train the classifier
    results = {}
    best_val = -1
    best_softmax = None

    for reg in regularization_strengths:
        # Create a new Softmax instance
        softmax_model = Softmax()
        # Train the model with current parameters
        softmax_model.train(X_train, y_train, learning_rate=2e-6, reg=reg, num_iters=1000)
        # Predict values for training set
        y_train_pred = softmax_model.predict(X_train)
        # Calculate accuracy
        train_accuracy = np.mean(y_train_pred == y_train)
        # Predict values for validation set
        y_val_pred = softmax_model.predict(X_val)
        # Calculate accuracy
        val_accuracy = np.mean(y_val_pred == y_val)
        # Save results
        results[reg] = (train_accuracy, val_accuracy)
        if best_val < val_accuracy:
            best_val = val_accuracy
            best_softmax = softmax_model

    # Print out results
    for reg in sorted(results):
        train_accuracy, val_accuracy = results[reg]
        print('reg %e train accuracy: %f val accuracy: %f' % (
                    reg, train_accuracy, val_accuracy))
    print('\nbest validation accuracy achieved during training: %f' % best_val)

    # Evaluate the best softmax on test set
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('\nfinal test set accuracy: %f' % test_accuracy)
    
    # Return the best trained classifier
    return best_softmax

