# coding: utf-8
import tensorflow as tf


def center_cost(in_x, in_y, centers):
    """
    :param in_x: tensor [batch x dim]
    :param in_y: tensor [batch]
    :param centers: tensor [class_num x dim]
    :return:
    """
    # center cost
    # diff_to_center = centers[in_y] - in_x  # shape: (batch x h)
    diff_to_center = tf.gather(centers, in_y) - in_x
    d = tf.reduce_mean(tf.square(diff_to_center))
    center_loss = tf.clip_by_value(0.1 * d, 1e-7, 10)
    return center_loss


def center_update(in_x, in_y, centers_list):
    centers_diff = []
    for c_id in range(len(centers_list)):
        idx = tf.where(tf.equal(in_y, c_id))
        c_point = tf.gather(in_x, idx)
        new_c = tf.transpose(tf.reduce_mean(c_point, axis=0))
        result = tf.cond(tf.cast(idx.get_shape()[0].value is None, tf.bool),
                         lambda: new_c,
                         lambda: centers_list[c_id] - new_c)
        centers_diff.append(result)
    return centers_diff


def attention_center(in_x, in_len, in_dim, h_dim, center_num):
    """
        :param in_x: tensor [batch x time_step x dim]
        :param in_len: tensor [batch]
        :param in_dim: int32 -1 dim of in_x
        :param h_dim: int32 dim of hidden
        :param center_num: int32
        :return:
        """
    in_shape = tf.shape(in_x)
    outputs = tf.reshape(in_x, [in_shape[0]*in_shape[1], in_shape[2]])    # [(batch*time_step) x hidden]
    # w_x = tf.get_variable('att_w_x', [dim, dim], initializer=tf.random_normal_initializer(stddev=0.01))
    with tf.variable_scope('center'):
        w_x = tf.get_variable('att_w_x', [in_dim, h_dim])
    b_x = tf.get_variable('att_b_x', [h_dim], initializer=tf.zeros_initializer)
    u = tf.tanh(tf.nn.xw_plus_b(outputs, w_x, b_x))  # [(batch*time_step) x hidden]
    centers_att = []

    for i in range(center_num):
        w_u = tf.get_variable('center_%d' % i, [h_dim, 1], initializer=tf.random_normal_initializer(stddev=0.01))
        att = tf.matmul(u, w_u)  # [(batch*time_step) x 1]
        att = tf.reshape(att, [in_shape[0], in_shape[1]])   # [batch x time_step]
        att = tf.exp(att) * tf.sequence_mask(in_len, maxlen=in_shape[1], dtype=tf.float32)  # [batch x time_step]
        att_sum = tf.reduce_sum(att, axis=1, keep_dims=True)  # [batch x 1]
        att_prob = att / att_sum  # [batch x time_step]
        att_prob = tf.expand_dims(att_prob, -1)
        att = tf.reduce_sum(in_x * att_prob, axis=1)  # [batch x hidden]
        centers_att.append(att)
    return tf.concat(centers_att, 1)
