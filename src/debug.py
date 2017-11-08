# coding: utf-8
from __future__ import print_function
from __future__ import division
import tensorflow as tf

from text_center_nn import Model
from dataset import DataSet
from config import ConfigRnn as Config

my_config = Config()
my_data = DataSet(my_config, True)
my_config.we = my_data.we
my_model = Model(my_config)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

d = my_model.model_debug(sess)
print (d.shape)
print (d[1, :])

