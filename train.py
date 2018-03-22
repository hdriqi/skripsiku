import tensorflow as tf
import numpy as np
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global config variables
num_steps = 3
day_predict = 2
data_length = 100
input_dimension = 14
output_dimension = 2
batch_size = 128
num_classes = 2
state_size = 20
learning_rate = 0.01

def prepare_train_data():
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_input = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(day_predict) + "days.csv").as_matrix()
    train_input = []
    train_output = []
    for i in range(0, data_length):
        train_input.append(df_input[i:i+num_steps])
        train_output.append(df_output[i:i+num_steps])

    return np.asarray(train_input), np.asarray(train_output)

def prepare_test_data():
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_input = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(day_predict) + "days.csv").as_matrix()
    test_input = []
    test_output = []
    for i in range(data_length, len(df_input) - 10):
        test_input.append(df_input[i:i+num_steps])
        test_output.append(df_output[i:i+num_steps])

    return np.asarray(test_input), np.asarray(test_output)

train_input, train_output = prepare_train_data()
test_input, test_output = prepare_test_data()

x = tf.placeholder(tf.float32, [None, num_steps, input_dimension], name='input_placeholder')
y = tf.placeholder(tf.float32, [None, num_steps, output_dimension], name='labels_placeholder')

# x = tf.placeholder(tf.float32, [None, None, input_dimension], name='input_placeholder')
# y = tf.placeholder(tf.float32, [None, None, output_dimension], name='labels_placeholder')

cell = tf.contrib.rnn.BasicRNNCell(state_size, activation=tf.nn.relu)
val, state = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
tf.Print(val.get_shape()[0], [val], "Acoy: ")
# rnn_last = tf.gather(val, int(val.shape[0]) - 1)
rnn_last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([state_size, num_classes]))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))

logits = tf.matmul(rnn_last, weight) + bias
prediction = tf.nn.softmax(logits)

y_last = tf.transpose(y, [1, 0, 2])
# y_last = tf.gather(y_last, int(y_last.shape[0]) - 1)
y_last = tf.gather(y_last, int(y_last.get_shape()[0]) - 1)

losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_last, logits=logits)
# losses = -tf.reduce_sum(y_last * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

total_loss =  tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)

# mistakes = tf.not_equal(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

correct = tf.equal(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def train_network():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        no_of_batches = int(len(train_input)/batch_size)
        train_loss = 0
        epoch = 1
        for i in range(epoch):
            step = i + 1
            ptr = 0
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
                ptr+=batch_size
                training_step, training_loss, training_state = sess.run([train_step, total_loss, state], feed_dict={
                    x: inp,
                    y: out
                })
                train_loss += training_loss
            if step % 10 == 0 and step != 0:
                train_loss = train_loss/10
                train_loss = 0
        accuracy_, correct_, y_last_, prediction_ = sess.run([accuracy, correct, y_last, prediction], 
            feed_dict={
                x: test_input, 
                y: test_output
            })
            # for i in range(len(y_last_)):
            #     print(y_last_[i], prediction_[i], correct_[i])
        print('Day to predict:', day_predict, 'Epoch:', step, "Hidden Node:", state_size, "Timesteps:", num_steps, "Accuracy", accuracy_ * 100)


num_steps = 3
train_network()

for i in range(2, 61, 2):
    day_predict = i

    # train_input, train_output = prepare_train_data()
    # test_input, test_output = prepare_test_data()

    num_steps = 3
    train_network()

    num_steps = 5
    train_network()

    num_steps = 7
    train_network()

    print()

# for i in range(20, 40, 2):
#     state_size = i
#     num_steps = 3
#     train_network()

#     num_steps = 5
#     train_network()

#     num_steps = 7
#     train_network()

#     print()