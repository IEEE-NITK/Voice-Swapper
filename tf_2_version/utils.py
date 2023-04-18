import tensorflow as tf
import os
import random
import numpy as np
# tf.compat.v1.enable_eager_execution()

def l1_loss(y, y_hat):
    y_hat = tf.cast(y_hat,dtype=tf.dtypes.float64)
    loss = tf.math.reduce_mean(tf.math.abs(y - y_hat))
    
    return loss

def l2_loss(y, y_hat):
    y_hat = tf.cast(y_hat,dtype=tf.dtypes.float64)
    y = tf.cast(y,dtype=tf.dtypes.float64)
    loss = tf.math.reduce_mean(tf.math.square(y - y_hat))    

    return loss

def cross_entropy_loss(logits, labels):
    return tf.math.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels),dtype=tf.dtypes.float32)

def gradient_penalty(discriminator, real_samples, fake_samples, batch_size, lambda_gp):
    alpha = tf.random.uniform(shape=[batch_size, 1, 1], minval=0., maxval=1.)
    interpolated_samples = real_samples + alpha * (fake_samples - real_samples)
    with tf.GradientTape() as tape:
        tape.watch(interpolated_samples)
        d_interpolated = discriminator(interpolated_samples, training=True)
    grads = tape.gradient(d_interpolated, interpolated_samples)
    grads_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
    gp = lambda_gp * tf.reduce_mean(tf.square(grads_norm - 1.))
    return gp
