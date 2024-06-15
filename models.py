__author__ = 'Raymond Chia'
'''
Contains basic descriptions of linear and non-linear classification models. 
All models can be reached via the 'get_model' function. Scikit-learn models 
may also be wrapped in a GridSearch hyperparameter optimisation function, see 
'set_gridsearch' and 'get_best_gridsearch'. Confident users may adapt neural 
network architectures as desired.
'''
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_decomposition import CCA

from sklearn.model_selection import GridSearchCV

import tensorflow as tf

from configs import N_CLASSES

class NNClass():
    def __init__(self,
                 n_classes=N_CLASSES,
                 batch_size=32,
                 lr=0.001,
                 epochs=10,
                 loss='categorical_crossentropy',
                 optimizer='Adam',
                 metrics=['categorical_accuracy'],
                 verbose=1,
                ):

        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = lr
        self.epochs = epochs
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.history = []
        self.model = None

        if optimizer.lower() == 'adagrad':
            self.optimizer = tf.keras.optimizers.Adagrad(self.learning_rate)
        elif optimizer.lower() == 'adam':
            self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        elif optimizer.lower() == 'adamw':
            self.optimizer = tf.keras.optimizers.AdamW(self.learning_rate)
        elif optimizer.lower() == 'rmsprop':
            self.optimizer = tf.keras.optimizers.RMSprop(self.learning_rate)
        elif optimizer.lower() == 'sgd':
            self.optimizer = tf.keras.optimizers.SGD(self.learning_rate)
        else:
            raise NotImplementedError

    def compile_model(self):
        return self.model.compile(optimizer=self.optimizer, loss=self.loss,
                             metrics=self.metrics)

    def fit_model(self, x, y, **kwargs):
        history = self.model.fit(x, y, batch_size=self.batch_size, 
                                 epochs=self.epochs, verbose=self.verbose, 
                                 **kwargs)
        self.history = history
        return self.model

class FNN(NNClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_model()
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(32),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        return self.model

class CNN1D(NNClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_model()
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv1D(16, 3, activation='gelu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(32, 2, activation='gelu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 2, activation='relu'),
            tf.keras.layers.GlobalMaxPool1D(),
            tf.keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        return self.model

class CNN2D(NNClass):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.create_model()
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, 2, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(128, 2, activation='relu'),
            tf.keras.layers.GlobalMaxPool2D(),
            tf.keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        return self.model

class LSTM(NNClass):
    def __init__(self, is_bidirectional=False, stateful=False, **kwargs):
        super().__init__(**kwargs)
        self.is_bidirectional = is_bidirectional
        self.stateful = stateful
        self.dropout = 0.4
        self.create_model()

    def LSTMCell(self, rnn_units, return_sequences=True, **kwargs):
        '''
            The requirements to use the cuDNN implementation are:
            activation == tanh
            recurrent_activation == sigmoid
            recurrent_dropout == 0
            unroll is False
            use_bias is True
            Inputs, if use masking, are strictly right-padded.
            Eager execution is enabled in the outermost context.
            internal dropout are recurrent output dropouts and input dropouts are
            input connection dropouts
        '''
        # Stateful runs the end of the current batch onto the next batch
        cell = tf.keras.layers.LSTM(
            rnn_units,
            return_sequences=return_sequences,
            recurrent_initializer='glorot_uniform',
            stateful=self.stateful,
            **kwargs
        )
        if self.is_bidirectional:
            return tf.keras.layers.Bidirectional(cell)
        else:
            return cell
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            self.LSTMCell(32),
            tf.keras.layers.Dropout(self.dropout),
            self.LSTMCell(64),
            tf.keras.layers.Dropout(self.dropout),
            self.LSTMCell(128, return_sequences=False),
            tf.keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        return self.model

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.4):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x

class ATTN(NNClass):
    def __init__(self, patch_size=25, **kwargs):
        super().__init__(**kwargs)
        # Ensure the stride and kernel size is of suitable size
        self.patch_size = patch_size

        self.create_model()
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            # Patch layer
            tf.keras.layers.Conv1D(32, self.patch_size,
                                   strides=self.patch_size,
                                   activation='relu'),
            # Linear Projection layer
            tf.keras.layers.Conv1D(32, 1),
            # Attention Encoding
            BaseAttention(num_heads=8, key_dim=128),
            FeedForward(32, 128),
            tf.keras.layers.Flatten(),
            # Output layer
            tf.keras.layers.Dense(self.n_classes, activation='softmax'),
        ])
        return self.model

def get_model(mdl_str, n_classes=N_CLASSES, **kwargs):
    '''
    Returns the requested model.
    Model selection can be as follows:
        svm : Support Vector Machine
        cca : Canonical Correlation Analysis
        lda : Linear Dicriminant Analysis
        tree: Decision Tree
        nb : Categorical Naive Bayes
        knn : K Nearest Neighours
        fnn : Fully connected Neural Network
        cnn1d : 1-D Convolutional Neural Network
        cnn2d : 2-D Convolutional Neural Network
        lstm : Long Short-Term Memory Neural Network
        attn : Patch input to Attention Encoder Neural Network
    '''
    if mdl_str == 'svm':
        return LinearSVC(**kwargs)
    elif mdl_str == 'cca':
        return CCA(n_components=n_classes, **kwargs)
    elif mdl_str == 'lda':
        return LinearDiscriminantAnalysis(n_components=n_classes-1, 
                                          **kwargs)
    elif mdl_str == 'tree':
        return DecisionTreeClassifier(**kwargs)
    elif mdl_str == 'nb':
        return CategoricalNB(**kwargs)
    elif mdl_str == 'knn':
        return KNeighborsClassifier(n_neighbors=n_classes, **kwargs)
    elif mdl_str == 'fnn':
        return FNN(n_classes=n_classes, **kwargs)
    elif mdl_str == 'cnn1d':
        return CNN1D(n_classes=n_classes, **kwargs)
    elif mdl_str == 'cnn2d':
        return CNN2D(n_classes=n_classes, **kwargs)
    elif mdl_str == 'lstm':
        return LSTM(n_classes=n_classes, **kwargs)
    elif mdl_str == 'attn':
        return ATTN(n_classes=n_classes, **kwargs)
    else:
        raise NotImplementedError

def set_gridsearch(model, params, scoring='f1_micro', cv=3):
    return GridSearchCV(estimator=model,
                        param_grid=params,
                        scoring=scoring,
                        cv=cv,)

def get_best_gridsearch(model):
    print("selected model score: ", model.best_score_)
    print("selected model params: ", model.best_params_)
    return model.best_estimator_
