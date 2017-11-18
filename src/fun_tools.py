# coding: utf-8
import tensorflow as tf


def cosine_distance(x1, x2):
    """
    :param x1: batch x dim
    :param x2: batch x dim
    :return: batch x 1
    """
    with tf.name_scope('cosine_distance'):
        # cosine = x*y / (|x||y|)
        # |x| = sqrt(x1^2+x2^2+...+xn^2)

        # shape: batch x 1
        x1_norm = tf.sqrt(tf.reduce_sum(tf.square(x1), axis=1))
        x2_norm = tf.sqrt(tf.reduce_sum(tf.square(x2), axis=1))
        # shape: batch x 1
        d = tf.reduce_sum(tf.multiply(x1, x2), axis=1) / tf.multiply(x1_norm, x2_norm)
        return d


def l1_distance(x1, x2):
    """
    :param x1: batch x dim
    :param x2: batch x dim
    :return: batch x 1
    """
    return tf.reduce_sum(x1 - x2, axis=1)


def l2_distance(x1, x2):
    """
    :param x1: batch x dim
    :param x2: batch x dim
    :return: batch x 1
    """
    return tf.sqrt(tf.reduce_sum(tf.square(x1 - x2), axis=1))
