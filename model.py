import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
import tensorflow_probability as tfp
from absl import flags
import tensorflow.keras.backend as K
FLAGS = flags.FLAGS


class CustomLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomLossLayer, self).__init__(**kwargs)
        self.eps1 = 1e-6

    def call(self, inputs):
        # Print the structure of inputs to debug
        print(f"Received inputs: {inputs}")
        
        # Check if inputs are a list or tuple
        if isinstance(inputs, list):
            print("Inputs are a list")
        elif isinstance(inputs, tuple):
            print("Inputs are a tuple")
        else:
            print(f"Unexpected input type: {type(inputs)}")

        # Unpack the inputs
        input_label, sample_r, r_mu, r_sqrt_sigma = inputs[0], inputs[1], inputs[2], inputs[3]
        
        # Ensure all inputs are tf.Tensor types
        input_label = tf.convert_to_tensor(input_label, dtype=tf.float32)
        sample_r = tf.convert_to_tensor(sample_r, dtype=tf.float32)
        r_mu = tf.convert_to_tensor(r_mu, dtype=tf.float32)
        r_sqrt_sigma = tf.convert_to_tensor(r_sqrt_sigma, dtype=tf.float32)
        
        # Check the shapes and broadcast r_sqrt_sigma if needed
        if r_sqrt_sigma.shape[-1] == 20:
            r_sqrt_sigma = tf.reshape(r_sqrt_sigma, [1, 1, 404, 20])
            r_sqrt_sigma = tf.tile(r_sqrt_sigma, [tf.shape(input_label)[0], tf.shape(input_label)[1], 1, 1])  # broadcast across batch

        # Compute the normal distribution for sampling
        norm = tfp.distributions.Normal(loc=0.0, scale=1.0)
        E = norm.cdf(sample_r) * (1 - self.eps1) + self.eps1 * 0.5

        # Compute negative log-likelihood (nll)
        sample_nll = -(tf.math.log(E) * input_label + tf.math.log(1 - E) * (1 - input_label))
       
        logprob = -tf.reduce_sum(sample_nll, axis=2)

        # Avoid overflow
        maxlogprob = tf.reduce_max(logprob, axis=0)
        Eprob = tf.reduce_mean(tf.exp(logprob - maxlogprob), axis=0)
        nll_loss = tf.reduce_mean(-tf.math.log(Eprob + 1e-10) - maxlogprob)

        # Compute marginal loss
        indiv_prob = tf.reduce_mean(E, axis=0, keepdims=True, name='individual_prob')

        # Compute cross-entropy with numerical stability
        cross_entropy = (
            tf.math.log(tf.clip_by_value(indiv_prob, 1e-8, 1.0)) * input_label + 
            tf.math.log(tf.clip_by_value(1.0 - indiv_prob, 1e-8, 1.0)) * (1 - input_label)
        )

        # Compute marginal loss
        marginal_loss = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))

        # Compute L2 regularization loss
        # Add L2 regularization losses from all layers
        l2_loss = marginal_loss
        # Sum of all regularization losses defined in layers

        # Compute total loss
        total_loss = nll_loss + marginal_loss + l2_loss  # Add marginal_loss to the total loss as well
        # Debug print to see what is being returned
        # Debug print to check values and shapes
        
    

        
        # Return nll_loss, l2_loss, and total_loss as a tuple
        return nll_loss, l2_loss, total_loss

    




class MODEL(tf.keras.Model):
    def __init__(self, is_training):
        super(MODEL, self).__init__()

        # Set the random seed for reproducibility
        tf.random.set_seed(19950420)

        r_dim = FLAGS.r_dim

        # Define the input layers
        self.input_nlcd = tf.keras.Input(shape=(None, FLAGS.nlcd_dim), dtype=tf.float32, name='input_nlcd')
        self.input_label = tf.keras.Input(shape=(None, FLAGS.r_dim), dtype=tf.float32, name='input_label')
        self.keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name='keep_prob') #keep probability for the dropout
        print(f"input_label in call method: {self.input_label}")

        # Regularizer for weights
        weights_regularizer = tf.keras.regularizers.l2(FLAGS.weight_decay)

        # Build the model layers
        self.fc_1 = tf.keras.layers.Dense(256, kernel_regularizer=weights_regularizer)(self.input_nlcd)
        self.fc_2 = tf.keras.layers.Dense(512, kernel_regularizer=weights_regularizer)(self.fc_1)
        #self.fc_3 = tf.keras.layers.Dense(units=256, kernel_regularizer=weights_regularizer, activation=tf.nn.relu, name='extractor_fc_3')

        # Define the feature from the layers
       # self.feature = self.fc_3(self.fc_2(self.fc_1(self.input_X)))
        feature1=self.fc_2
        ############## compute mu & sigma ###############
        #compute the mean of the normal random variables 
        self.r_mu = tf.keras.layers.Dense(units=r_dim, activation=None, kernel_regularizer=weights_regularizer, name='r_mu')(feature1)
        
        # Initialize the square root of the residual covariance matrix 
        self.r_sqrt_sigma = tf.Variable(
            np.random.uniform(-np.sqrt(6.0 / (r_dim + FLAGS.z_dim)),
                              np.sqrt(6.0 / (r_dim + FLAGS.z_dim)),
                              (r_dim, FLAGS.z_dim)), 
            dtype=tf.float32, name='r_sqrt_sigma')

        # Compute the residual covariance matrix, which is guaranteed to be semi-positive definite
        self.sigma = tf.matmul(self.r_sqrt_sigma, self.r_sqrt_sigma, transpose_b=True)
        self.covariance = self.sigma + tf.eye(r_dim)
        self.cov_diag = tf.linalg.diag_part(self.covariance)


        # Residual means
        #self.r_mu = self.normalized_mu * tf.sqrt(self.cov_diag)

        #compute the loss for each sample point
        self.loss_layer = CustomLossLayer()

    def call(self, is_training=False):
        # This is where KerasTensor (self.r_mu) is properly handled
        n_batch = FLAGS.batch_size  # Evaluate KerasTensor here in the call() method

        # Print the value of self.r_mu
        tf.print("self.r_mu:", self.r_mu, summarize=-1)

        ############## Sample_r ###############
        # Generate the noise tensor based on the training flag
        n_sample = FLAGS.n_train_sample if is_training else FLAGS.n_test_sample
        print ("n_sample=",n_sample)

        self.noise = tf.random.normal(shape=[n_sample, n_batch, FLAGS.z_dim])

        # B matrix transpose
        self.B = tf.transpose(self.r_sqrt_sigma)#*self.sqrt_diag

        # Sample the r variables
        self.sample_r = tf.tensordot(self.noise, self.B, axes=1) + self.r_mu  #tensor: n_sample*n_batch*r_dim
        print("input_label shape:", self.input_label.shape)
        print("sample_r shape:", self.sample_r.shape)
        print("r_mu:", self.r_mu)
        print("r_sqrt_sigma shape:", self.r_sqrt_sigma.shape)

        
        # Pass the relevant inputs to the loss layer
        nll_loss, l2_loss, total_loss = self.loss_layer(
            [self.input_label, self.sample_r, self.r_mu, self.r_sqrt_sigma]
        )
        
   
        return nll_loss, l2_loss, total_loss
        