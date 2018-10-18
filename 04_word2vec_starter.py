""" The mo frills implementation of word2vec skip-gram model using NCE loss. 
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 20000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss

class SkipGramModel:
    def __init__(self,):
        self.VOCAB_SIZE = 50000
        self.BATCH_SIZE = 128
        self.EMBED_SIZE = 128 # dimension of the word embedding vectors
        self.SKIP_WINDOW = 1 # the context window
        self.NUM_SAMPLED = 64    # Number of negative examples to sample.
        self.LEARNING_RATE = 1.0
        self.NUM_TRAIN_STEPS = 20000
        self.SKIP_STEP = 2000 # how many steps to skip before reporting the loss

    def _import_data(self):
        return process_data(self.VOCAB_SIZE, self.BATCH_SIZE, self.SKIP_WINDOW)

    def _create_embedding(self):
        with tf.name_scope('data'):
            self.center_words = tf.placeholder(dtype=tf.int32, name="center_words", shape=[self.BATCH_SIZE])
            self.target_words = tf.placeholder(dtype=tf.int32, name="target_words", shape=[self.BATCH_SIZE,1])
        # Step 2: define weights. In word2vec, it's actually the weights that we care about
        # vocab size x embed size
        # initialized to random uniform -1 to 1
        with tf.name_scope('embed'):
            INIT_matrix = tf.random_uniform( [self.VOCAB_SIZE, self.EMBED_SIZE], minval=-1, maxval=1, dtype=tf.float32 )
            self.embed_matrix = tf.get_variable("embed_matrix", dtype=tf.float32, initializer=INIT_matrix)

    def _create_loss(self):

        # Step 3: define the inference
        # get the embed of input words using tf.nn.embedding_lookup
        # embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')
        with tf.name_scope('loss'):
            self.embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name="embed")
            tf.reshape(self.embed, [self.BATCH_SIZE,self.EMBED_SIZE,1])
        # Step 4: construct variables for NCE loss
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
        # bias: vocab size, initialized to 0
            INIT_nce_w = tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], stddev=1.0/EMBED_SIZE**0.5 )
            INIT_nce_b = tf.zeros([VOCAB_SIZE], dtype=tf.float32)
            self.nce_weight = tf.get_variable("nce_weight", dtype=tf.float32, initializer= INIT_nce_w)
            self.nce_b = tf.get_variable("nce_bias", dtype=tf.float32, initializer= INIT_nce_b)

        # define loss function to be NCE loss function
        # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
        # need to get the mean accross the batch
        # note: you should use embedding of center words for inputs, not center words themselves
        # TO DO
            loss = tf.nn.sampled_softmax_loss(weights=self.nce_weight, biases=self.nce_b,
                                  labels=self.target_words, inputs=self.embed,
                                  num_sampled=NUM_SAMPLED, num_classes=VOCAB_SIZE)
            self.loss = tf.reduce_mean( loss )
            return self.loss
    def _create_optimizer(self):
        self.gloabl_step = tf.Variable(0,dtype=tf.int32, trainable=False, name='global_step')
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(self.loss,global_step=self.gloabl_step)
        return optimizer

    def _create_summary(self):
        tf.summary.scalar("loss",self.loss)
        tf.summary.histogram("histogram loss", self.loss)
        summary_op = tf.summary.merge_all()
        return summary_op

def main():
    # batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    # word2vec(batch_gen)
    w2v = SkipGramModel()
    batch_gen = w2v._import_data()
    w2v._create_embedding()
    loss = w2v._create_loss()
    optimizer = w2v._create_optimizer()
    summary_op = w2v._create_summary()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # TO DO: initialize variables
        sess.run( tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter('./graphs/no_frills/', sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets = next(batch_gen)
            # TO DO: create feed_dict, run optimizer, fetch loss_batch
            loss_batch, _ ,summary = sess.run([loss, optimizer,summary_op], feed_dict={w2v.center_words:centers, w2v.target_words: targets})
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                writer.add_summary(summary, global_step=index)
                saver.save(sess, 'checkpoint_dict/SkipGramModel', global_step=w2v.gloabl_step)
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0

        writer.close()

if __name__ == '__main__':
    main()
