__author__ = 'Iris'

"""logistic regression to predict heart disease : 9 features, 1 label
label=0 : negative
label=1 : positive
"""

import numpy as np
import tensorflow as tf
import time
import shutil
import utils

# Define paramaters for the model
learning_rate = 0.003
batch_size = 30
n_epochs = 40
n_train = 362
n_test = 100
n_features = 9
keep_prob = tf.placeholder(tf.float32)

# Step 1: Read in data
fpath = ['data/heart.csv']
filenames = tf.train.string_input_producer(fpath)
reader = tf.TextLineReader(skip_header_lines=1)
_, value = reader.read(filenames)

record_default = [ [1.0] for i in range(n_features)]
record_default[4] = ['']
record_default.append([1])
content = tf.decode_csv(value, record_default=record_default)

# Step 2: Create datasets and iterator
# create training Dataset and batch it
train_data = tf.data.Dataset.from_tensor_slices((features[:n_train],labels[:n_train]))
train_data = train_data.shuffle(10)
train_data = train_data.batch(batch_size)

# create testing Dataset and batch it
test_data = tf.data.Dataset.from_tensor_slices((features[n_train:],labels[n_train:]))
test_data = test_data.shuffle(10)
test_data = test_data.batch(batch_size)

# create one iterator and initialize it with different datasets
iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
feature, label = iter.get_next()
train_init = iter.make_initializer(train_data)
test_init = iter.make_initializer(test_data)

# Step 3: create weights and bias
x_dim = feature.get_shape().as_list()[1]
y_dim = label.get_shape().as_list()[1]

w_INIT = tf.random_normal([x_dim, y_dim], mean=0.0, stddev=0.01, dtype=tf.float32 )
b_INIT = tf.zeros([1,y_dim],dtype=tf.float32 )
w = tf.get_variable("weight", dtype=tf.float32, initializer=w_INIT)
b = tf.get_variable("bias", dtype=tf.float32, initializer=b_INIT)

# Step 4: build model
logits = tf.matmul(feature, w) + b
logits = tf.nn.dropout(logits,keep_prob)

# Step 5: define loss function
# use cross entropy of softmax of logits as the loss function
entropy = tf.sigmoid(logits)
loss = tf.losses.mean_squared_error(labels=label,predictions=entropy)

# Step 6: define optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Step 7: calculate accuracy with test set
pred = tf.cast(tf.greater(entropy,0.5), tf.float32)
correct_pred = tf.equal( pred ,label)
accuracy = tf.reduce_sum(tf.cast(correct_pred, tf.float32))

writer = tf.summary.FileWriter('./graphs/logreg', tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range( n_epochs):
        sess.run(train_init)
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l = sess.run([optimizer, loss], feed_dict ={keep_prob:0.5})
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        print("Average loss epoch {0}:{1}".format(i, total_loss/n_batches))

    # test the model
    sess.run(test_init)
    total_corrected_preds = 0
    try:
        while True:
            accuracy_batch = sess.run(accuracy, feed_dict={keep_prob:1})
            total_corrected_preds += accuracy_batch
    except tf.errors.OutOfRangeError :
        pass

    print("Accuracy {0}".format(total_corrected_preds/n_test))