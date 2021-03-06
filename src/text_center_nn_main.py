# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf

from text_center_nn import Model
from dataset import DataSet
from config import ConfigRnn as Config

import numpy as np
import tools

my_config = Config()
my_data = DataSet(my_config, True)
my_config.we = my_data.we
my_model = Model(my_config)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


def train():
    summary_writer = tf.summary.FileWriter(my_config.log_dir, sess.graph)
    my_model.train_dev_test(sess, [my_data.train_x, my_data.train_y, my_data.train_seq_len],
                            test_xy=[my_data.test_x, my_data.test_y, my_data.test_seq_len],
                            save_model=True,
                            summary_writer=summary_writer)


def get_repr(fname):
    samples_v = my_model.get_represent(sess, [my_data.train_x, my_data.train_y, my_data.train_seq_len])
    samples_v = np.array(samples_v)
    print ("samples_vector: {}".format(samples_v.shape))
    tools.save_params([samples_v, my_data.train_y], my_config.log_dir+"/{}.pkl".format(fname))

if __name__ == '__main__':
    train()
    # get_repr("center_divide_samples_vector")
