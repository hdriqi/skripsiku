import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Global config variables
file_name = "./model"
feature_dimension = 14
num_steps = 3
input_dimension = feature_dimension * num_steps
output_dimension = 2
num_classes = 2

epoch = 40
days_predict = 2
data_length = 1000
batch_size = 128
state_size = 40
learning_rate = 0.01
opt_optimizer = "adam"
k_fold = KFold(n_splits=5, shuffle=True)

def prepare_train_data(day):
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(day) + "days.csv").as_matrix()
    train_input = []
    train_output = []
    for i in range(0, data_length):
        train_input.append(df_norm[i:i+num_steps])
        train_output.append(df_output[i:i+num_steps])

    return np.asarray(train_input), np.asarray(train_output)

def prepare_test_data(day):
    df_input = pd.read_csv("dataset.csv").drop('timestamp', axis=1).as_matrix()[1:]
    df_norm = (df_input - df_input.mean()) / (df_input.max() - df_input.min())
    df_output = pd.read_csv("labels/" + str(day) + "days.csv").as_matrix()
    test_input = []
    test_output = []
    for i in range(data_length, len(df_norm) - num_steps):
        test_input.append(df_norm[i:i+num_steps])
        test_output.append(df_output[i:i+num_steps])

    return np.asarray(test_input), np.asarray(test_output)

x = tf.placeholder(tf.float32, [None, input_dimension], name='input_placeholder')
y = tf.placeholder(tf.float32, [None, None, output_dimension], name='labels_placeholder')

hidden_1 = tf.layers.dense(inputs=x, units=1024, activation=tf.nn.relu)
logits = tf.layers.dense(inputs=hidden_1, units=num_classes)

prediction = tf.nn.softmax(logits)

y_last = tf.transpose(y, [1, 0, 2])
y_last = tf.gather(y_last, num_steps - 1)

losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_last, logits=logits)

cost =  tf.reduce_mean(losses)
if(opt_optimizer == "adam"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
elif(opt_optimizer == "adadelta"):
    train_step = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)
elif(opt_optimizer == "rmsprop"):
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
else:
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

correct = tf.equal(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
recall = tf.metrics.recall(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
precision = tf.metrics.precision(tf.argmax(y_last, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

def train_network(train_input, train_output, valid_input, valid_output):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        no_of_batches = int(len(train_input)/batch_size)
        train_loss = []
        train_losses = []
        test_accuracy = []
        test_losses = []
        test_precision = []
        test_recall = []
        for i in range(epoch):
            step = i + 1
            ptr = 0
            for j in range(no_of_batches):
                inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
                inp = inp.reshape(batch_size, input_dimension)
                ptr+=batch_size
                training_step, training_cost = sess.run([train_step, cost], 
                    feed_dict={
                        x: inp,
                        y: out
                    })
                train_loss.append(training_cost)
            
            # RESULT PER EPOCH
            valid_input = valid_input.reshape(200, input_dimension)
            accuracy_, precision_, recall_, prediction_, cost_ = sess.run([accuracy, precision, recall, prediction, cost], 
                feed_dict={
                    x: valid_input, 
                    y: valid_output
                })
            # print(prediction_)
            train_losses.append(np.mean(train_loss))
            train_loss = []
            test_accuracy.append(accuracy_*100)
            test_losses.append(cost_)
            test_precision.append(precision_[0])
            test_recall.append(recall_[0])

        # RETURN array of value per epoch
        test_acc = np.asarray(test_accuracy)
        # print("Day", days_predict, "Epoch", np.argmax(test_acc), np.argmin(test_acc))
        return np.asarray(test_accuracy), np.asarray(train_losses), np.asarray(test_losses), np.asarray(test_precision), np.asarray(test_recall)

def train_network_kfold(day=2, step=3, verbose=True):
    days_predict = day
    num_steps = step
    train_input, train_output = prepare_train_data(days_predict)
    test_input, test_output = prepare_test_data(days_predict)

    total_acc = []
    total_train_losses = []
    total_val_losses = []
    total_val_precision = []
    total_val_recall = []

    train_losses_by_epoch = []
    val_losses_by_epoch = []
    i = 1
    for train_indices, val_indices in k_fold.split(train_input, train_output):
        train_input_kfold = train_input[train_indices]
        train_output_kfold = train_output[train_indices]

        val_input_kfold = train_input[val_indices]
        val_output_kfold = train_output[val_indices]
        
        val_acc_, train_losses_, val_losses_, val_precision_, val_recall_ = train_network(train_input_kfold, train_output_kfold, val_input_kfold, val_output_kfold)

        total_acc.append(val_acc_[-1])
        total_train_losses.append(train_losses_[-1])
        total_val_losses.append(val_losses_[-1])
        total_val_precision.append(val_precision_[-1])
        total_val_recall.append(val_recall_[-1])

        train_losses_by_epoch.append(train_losses_)
        val_losses_by_epoch.append(val_losses_)

        i+=1

    train_losses_by_epoch = np.asarray(train_losses_by_epoch)
    train_losses_by_epoch = np.mean(train_losses_by_epoch.T, axis=1)

    val_losses_by_epoch = np.asarray(val_losses_by_epoch)
    val_losses_by_epoch = np.mean(val_losses_by_epoch.T, axis=1)

    # plt.plot(train_losses_by_epoch, label="Train Loss")
    # plt.plot(val_losses_by_epoch, label="Val Loss")
    # print("Best Train Loss", np.argmin(train_losses_by_epoch))
    # print("Best Val Loss", np.argmin(val_losses_by_epoch))

    # set as numpy array
    total_acc = np.asarray(total_acc)
    total_train_losses = np.asarray(total_train_losses)
    total_val_losses = np.asarray(total_val_losses)
    total_val_precision = np.asarray(total_val_precision)
    total_val_recall = np.asarray(total_val_recall)

    # print avg k-fold
    if(verbose):
        print("------------ Day", day, "• Timestep", step, "------------")
        print("Average Acc", round(np.mean(total_acc)), "Best Acc", round(np.amax(total_acc)), np.argmax(total_acc),"Worst Acc", round(np.amin(total_acc), np.argmin(total_acc)))
        print("Average Validation Precision", np.mean(total_val_precision))
        print("Average Validation Recall", np.mean(total_val_recall[-1]))

    # return per fold
    return total_acc, total_train_losses, total_val_losses

top_acc = 0
# top_acc_details = ""

total_acc = []
for i in range(10,11):
    step = 2*i+1
    total_acc = []
    for j in range(2, 61, 2):
        opt_optimizer = "adam"
        acc, train_losses, val_losses = train_network_kfold(j, step=step, verbose=True)
        total_acc.append(np.mean(acc))
        if(np.mean(acc) > top_acc):
            top_acc = np.mean(acc)
            top_acc_details = "Days " + str(j) + " Timesteps " + str(step)
    plt.plot([z for z in range(2, 61, 2)], total_acc, label="Accuracy - Timesteps " + str(step))

print(top_acc, top_acc_details)
plt.yticks([i for i in range(0, 101, 10)])
plt.ylabel("Accuracy")
plt.xticks([i for i in range(2, 61, 2)])
plt.xlabel("Time Window (days)")
plt.legend()
plt.show()