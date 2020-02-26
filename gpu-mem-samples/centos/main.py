#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = None

def  train():

	a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
	b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
	c = tf.matmul(a, b)
    
	sess = tf.Session()
	# Runs the op.
	while True:
		sess.run(c)


if __name__ == '__main__':
    train()