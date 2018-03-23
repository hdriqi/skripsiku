import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global config variables
file_name = "./model"
num_steps = 3
epoch = 40
days_predict = 2
data_length = 1000
input_dimension = 14
output_dimension = 2
batch_size = 128
num_classes = 2
state_size = 40
learning_rate = 0.01

def prepare_train_data():
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(days_predict) + "days.csv").as_matrix()
    train_input = []
    train_output = []
    for i in range(0, data_length):
        train_input.append(df_norm[i:i+num_steps])
        train_output.append(df_output[i:i+num_steps])

    return np.asarray(train_input), np.asarray(train_output)

def prepare_test_data():
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(days_predict) + "days.csv").as_matrix()
    test_input = []
    test_output = []
    for i in range(data_length, len(df_norm) - 10):
        test_input.append(df_norm[i:i+num_steps])
        test_output.append(df_output[i:i+num_steps])

    return np.asarray(test_input), np.asarray(test_output)

train_input, train_output = prepare_train_data()
test_input, test_output = prepare_test_data()

x = tf.placeholder(tf.float32, [None, None, input_dimension], name='input_placeholder')
y = tf.placeholder(tf.float32, [None, None, output_dimension], name='labels_placeholder')
init_state = tf.placeholder(tf.float32, [None, state_size], name='state_placeholder')

cell = tf.contrib.rnn.BasicRNNCell(state_size, activation=tf.nn.relu)
cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
val, state = tf.nn.dynamic_rnn(cell, x, initial_state=init_state, dtype=tf.float32)

val = tf.transpose(val, [1, 0, 2])
rnn_last = tf.gather(val, num_steps - 1)

weight = tf.Variable(tf.truncated_normal([state_size, num_classes]))
bias = tf.Variable(tf.zeros(num_classes))

logits = tf.matmul(rnn_last, weight) + bias
prediction = tf.nn.softmax(logits)

y_last = tf.transpose(y, [1, 0, 2])
y_last = tf.gather(y_last, num_steps - 1)

losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_last, logits=logits)

avg_losses =  tf.reduce_mean(losses)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(avg_losses)
# train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(avg_losses)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(avg_losses)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(avg_losses)

correct = tf.equal(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def train_network():
    train_input, train_output = prepare_train_data()
    test_input, test_output = prepare_test_data()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        no_of_batches = int(len(train_input)/batch_size)
        train_loss = []
        train_losses = []
        test_accuracy = []
        test_losses = []
        for i in range(epoch):
            step = i + 1
            ptr = 0
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
                ptr+=batch_size
                rnn_init_weight = np.eye(batch_size, state_size)
                training_step, training_loss, training_state = sess.run([train_step, avg_losses, state], 
                    feed_dict={
                        x: inp,
                        y: out,
                        init_state: rnn_init_weight
                    })
                train_loss.append(training_loss)

            train_losses.append(np.mean(train_loss))
            train_loss = []

            rnn_init_weight = np.eye(len(test_input), state_size)
            accuracy_, correct_, y_last_, prediction_, avg_losses_ = sess.run([accuracy, correct, y_last, prediction, avg_losses], 
                feed_dict={
                    x: test_input, 
                    y: test_output,
                    init_state: rnn_init_weight
                })
            
            test_losses.append(avg_losses_)
        # print('Days to predict:', days_predict, 'Epoch:', step, "Hidden Node:", state_size, "Timesteps:", num_steps, "Accuracy", accuracy_ * 100)
        return accuracy_ * 100, train_losses, test_losses

total_acc = []
total_train_losses = []
total_valid_losses = []

# TIMESTEP_LOOP
for z in range(1):
    for i in range(1, 2):
        num_steps = 2 * i + 1
        # total_acc.append([])
        total_acc = []
        total_train_losses = []
        total_valid_losses = []
        # DAYS_LOOP
        for j in range(2, 61, 2):
            days_predict = j
            acc, train_loss, val_loss = train_network()
            total_acc.append(acc)
            total_train_losses.append(train_loss)
            total_valid_losses.append(val_loss)

    # PRINT TRAIN LOSS
    # for i, val in enumerate(total_train_losses):
    #     plt.plot(val, label="Train Losses Index " + str(i))

    print("Average Loss", str(np.mean(total_valid_losses)))
    total_valid_losses = np.array(total_valid_losses)
    best_loss = np.argmin(total_valid_losses.transpose()[-1])
    worst_loss = np.argmax(total_valid_losses.transpose()[-1])
    print("Best Loss", str(best_loss * 2), "->", str(total_valid_losses.transpose()[-1][best_loss]))
    print("Worst Loss", str(worst_loss * 2), "->", str(total_valid_losses.transpose()[-1][worst_loss]))
    # PRINT VALID LOSS
    # for i, val in enumerate(total_valid_losses):
    #     plt.plot(val, label="Valid Losses Index " + str(i))
    print()

    print("Average Accuracy", str(np.mean(total_acc)))
    total_acc = np.array(total_acc)
    best_acc = np.argmax(total_acc)
    worst_acc = np.argmin(total_acc)
    print("Best Accuracy", str(best_acc * 2), "->", str(total_acc.flatten()[best_acc]))
    print("Worst Accuracy", str(worst_acc *2), "->", str(total_acc.flatten()[worst_acc]))
    # PRINT ACCURACY
    # for i, val in enumerate(total_acc):
    plt.plot([i for i in range(2, 61, 2)], total_acc, label="Accuracy - Timesteps -> " + str(2 * i + 1))

    plt.yticks([i for i in range(0, 101, 10)])
    plt.ylabel("Accuracy")
    plt.xticks([i for i in range(2, 61, 2)])
    plt.xlabel("Time Window (days)")
    plt.legend()
    plt.show()

    print('------------------------')