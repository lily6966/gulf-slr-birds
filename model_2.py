import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import tensorflow_probability as tfp
from absl import flags
import tensorflow.keras.backend as K
FLAGS = flags.FLAGS
from keras import Model, layers, regularizers, initializers, activations
from tensorflow_probability import distributions
from tensorflow import (
    shape,
    eye,
    sqrt,
    matmul,
    reduce_sum,
    exp,
    constant,
    cast,
    transpose,
    tensordot,
    reduce_max,
    reduce_mean,
    random,
    linalg,
    math,
    float32,
    float32,
)


class MODEL(tf.keras.Model):
    def __init__(self, is_training):
        super(MODEL, self).__init__()

        # Set the random seed for reproducibility
        tf.random.set_seed(19950420)

        self.r_dim = FLAGS.r_dim
        self.batch_size = FLAGS.batch_size
        self.is_training = is_training

        # Define the input layers (as part of the model's architecture)
        self.input_nlcd = tf.keras.Input(shape=(None, FLAGS.nlcd_dim), dtype=tf.float32, name='input_nlcd')
        self.input_label = tf.keras.Input(shape=(None, FLAGS.r_dim), dtype=tf.float32, name='input_label')
        self.keep_prob = tf.keras.Input(shape=(), dtype=tf.float32, name='keep_prob')  # for dropout (if used)

        # Regularizer for weights
        weights_regularizer = tf.keras.regularizers.l2(FLAGS.weight_decay)

        # Define the model layers
        self.fc_1 = tf.keras.layers.Dense(256, kernel_regularizer=weights_regularizer)
        self.fc_2 = tf.keras.layers.Dense(512, kernel_regularizer=weights_regularizer)
        self.fc_3 = tf.keras.layers.Dense(512, kernel_regularizer=weights_regularizer)

        # Define the residual mean (mu) normalized mu
        self.r_mu = tf.keras.layers.Dense(self.r_dim, activation=None, kernel_regularizer=weights_regularizer, name='r_mu')
        
        # Learnable weight for the residual covariance
        # self.r_sqrt_sigma = self.add_weight(
        #     name="r_sqrt_sigma",
        #     shape=(self.r_dim, FLAGS.z_dim),
        #     initializer=tf.keras.initializers.RandomUniform(
        #         -tf.sqrt(6.0 / (self.r_dim + FLAGS.z_dim)),
        #         tf.sqrt(6.0 / (self.r_dim + FLAGS.z_dim))
        #     ),
        #     trainable=True
        # )
        
        # Initialize the square root of the residual covariance matrix
        initializer = initializers.RandomUniform(
            minval=-np.sqrt(6.0 / (self.r_dim + FLAGS.z_dim)),
            maxval=np.sqrt(6.0 / (self.r_dim + FLAGS.z_dim)),
        )

        self.r_sqrt_sigma = self.add_weight(
            shape=(self.r_dim, FLAGS.z_dim),
            initializer=initializer,
            dtype=float32,
            trainable=True,
        )

        
        """ self.r_sqrt_sigma = tf.Variable(
            np.random.uniform(-np.sqrt(6.0/(self.r_dim+FLAGS.z_dim)), np.sqrt(6.0/(self.r_dim+FLAGS.z_dim)), (self.r_dim, FLAGS.z_dim)),
            dtype=tf.float32,
            name='r_sqrt_sigma',
            trainable=True  # Ensure this variable is trainable
        ) """

        # Compute the covariance matrix, which is guaranteed to be semi-positive definite

        # self.sigma = tf.matmul(self.r_sqrt_sigma, tf.transpose(self.r_sqrt_sigma))
        # self.sqrt_diag=1.0/tf.sqrt(tf.linalg.diag_part(self.sigma))
        # self.covariance = self.sigma + tf.eye(self.r_dim)
        # self.cov_diag = tf.linalg.diag_part(self.covariance)

        # Initialize constants
        self.eps2 = tf.constant(1e-6 * 2.0**(-100), dtype="float64")
        self.eps1 = tf.constant(1e-6, dtype="float32")
        self.eps3 = 1e-30

        # Define distribution for sampling
        self.norm = random.normal

    

        # Transpose of the residual covariance matrix
        self.B = tf.transpose(self.r_sqrt_sigma)#*self.sqrt_diag

    def call(self, inputs, is_training=False):
        # Generate noise tensor based on whether we're in training or testing
        n_sample = FLAGS.n_train_sample if self.is_training else FLAGS.n_test_sample

        input_nlcd, input_label = inputs
        
        # Apply the model layers to input_nlcd
        #self.dropout = tf.keras.layers.Dropout(rate=0.5)
        

        feature1 = self.fc_1(input_nlcd)
        #feature1 = self.dropout(feature1, training=is_training)
        feature2 = self.fc_2(feature1)
        feature = self.fc_3(feature2)

        # Compute sigma (covariance matrix)
        sigma = matmul(self.r_sqrt_sigma, transpose(self.r_sqrt_sigma))
        covariance = sigma + eye(self.r_dim, dtype=float32)
        cov_diag = linalg.diag_part(covariance)

        # Compute normalized_mu and r_mu
        normalized_mu = self.r_mu(feature)
        r_mu = normalized_mu * sqrt(cov_diag)

        
        noise = tf.random.normal(
            shape=[n_sample, shape(r_mu)[0], FLAGS.z_dim], dtype=float32
        )
        # Sample residuals (r) from a multivariate normal distribution
        sample_r = tf.tensordot(noise, self.B, axes=1) + r_mu

        # Normal distribution to sample residuals
        norm = tfp.distributions.Normal(loc=0.0, scale=1.0)
        E = (
            cast(norm.cdf(cast(sample_r, float32)), float32) * (1 - self.eps1)
            + self.eps1 * 0.5
        )

        # Compute negative log-likelihood (nll) for the residuals
        input_label = tf.cast(input_label, dtype=tf.float32)  # Convert int64 -> float32
        sample_nll = -(tf.math.log(E) * input_label + tf.math.log(1 - E) * (1 - input_label))
        logprob = -tf.reduce_sum(sample_nll, axis=2)

        # Numerical stability: avoid overflow by adjusting probabilities
        maxlogprob = tf.reduce_max(logprob, axis=0)
        Eprob = tf.reduce_mean(tf.exp(logprob - maxlogprob), axis=0)
        nll_loss = tf.reduce_mean(-tf.math.log(Eprob + 1e-10) - maxlogprob)

        # Compute individual probabilities
        norm_dist = layers.Lambda(
            lambda x: cast(activations.sigmoid(x * 1.70169), float32)
        )  # Approximation of normal CDF

        indiv_prob = norm_dist(normalized_mu) * (1 - self.eps1) + 0.5 * self.eps1

        # Compute marginal loss (cross-entropy)
        cross_entropy = (
            tf.math.log(tf.clip_by_value(indiv_prob, 1e-8, 1.0)) * input_label +
            tf.math.log(tf.clip_by_value(1.0 - indiv_prob, 1e-8, 1.0)) * (1 - input_label)
        )
        marginal_loss = -tf.reduce_mean(tf.reduce_sum(cross_entropy, axis=1))

        # Compute L2 regularization loss (from model layers)
        l2_loss = tf.add_n(self.losses) if self.losses else 0.0

        # Compute the total loss
        total_loss = nll_loss + l2_loss

        return indiv_prob, nll_loss, marginal_loss, l2_loss, total_loss
