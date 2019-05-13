import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dfply import *


# To convert the sentences
# The method is the universal sentence encoder: https://arxiv.org/abs/1803.11175
module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/3"
# Import the Universal Sentence Encoder's TF Hub module
embed = hub.Module(module_url)

# one hot encoding function
def onehot_encoder(data):
    YY = np.zeros((len(data), 2))
    YY[np.arange(len(data)), data] = 1
    return YY


# the universal sentence encoder function
def USencoder(l):
    with tf.Session() as session:
        session.run([tf.global_variables_initializer(), tf.tables_initializer()])
        message_embeddings = session.run(embed(l))
        return message_embeddings

# load the data
df = pd.read_csv('Eluvio_DS_Challenge.csv')
# only keep the column up_votes and the title
df = df>> select(X.up_votes,X.title)

# class the data into two groups: with threshold as the 80% of the up_votes
# if up_votes number > threshold is an attractive news; o.w. it is not an attractive news
threshold = np.quantile( df['up_votes'], 0.8) # corresponding value is 24

df = df >> mutate(category = (X.up_votes>threshold)*1 )

print('The number of the attractive news', sum(df['category']), '; the number of the non-attractive news', len(df)-sum(df['category']),'.')
print('The raito of attractive ~ non-attractive is', sum(df['category'])/(len(df)-sum(df['category'])),'.')

np.random.seed(seed=1)
score = pd.DataFrame(np.random.randn(df.shape[0], 1))
msk = np.random.rand(len(score)) < 0.99

train = df[msk]
test = df[~msk]

print('The number of the training data is ', len(train), '; the number of the testing data', len(test),'.')

# # convert the testing data and save them
# test_text = test['title'].values.tolist()
#
# with tf.Session() as session:
#   session.run([tf.global_variables_initializer(), tf.tables_initializer()])
#   message_embeddings = session.run(embed(test_text))
#   np.savetxt('covariate.csv',message_embeddings, delimiter=',')

# test_x = message_embeddings
test_x = np.loadtxt('covariate.csv',delimiter = ',')
test_y = test['category'].values.tolist()
print('The proportion of the attractive news in testing data is ', sum(test_y)/len(test_y))
test_y = onehot_encoder(test_y)





def dnn(x):
    with tf.name_scope('layer_1'):
        W_1 = weight_variable([512,512])
        b_1 = bias_variable([512])
        h_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)

    with tf.name_scope('layer_2'):
        W_2 = weight_variable([512,128])
        b_2 = bias_variable([128])
        h_2 = tf.nn.relu(tf.matmul(h_1, W_2) + b_2)

    with tf.name_scope('layer_3'):
        W_3 = weight_variable([128,64])
        b_3 = bias_variable([64])
        h_3 = tf.nn.relu(tf.matmul(h_2, W_3) + b_3)

    with tf.name_scope('dropout'):
        rate = tf.placeholder(tf.float32)
        h_3_drop = tf.nn.dropout(h_3, rate = rate)

    with tf.name_scope('full_connected'):
        W_4 = weight_variable([64, 2])
        b_4 = bias_variable([2])
        y_pred = tf.nn.relu(tf.matmul(h_3_drop, W_4) + b_4)

    return y_pred, rate

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev= 0.1)
    return tf.Variable(initial, name = 'W')

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial, name = 'b')

def main():
    tf.set_random_seed(1)

    x = tf.placeholder(tf.float32, [None, 512])  # our covariate is 512 dimensional
    y = tf.placeholder(tf.float32, [None, 2])  # 2 classes

    y_pred , rate = dnn(x)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels= y, logits= y_pred)
        cross_entropy = tf.reduce_mean(cross_entropy)

    with tf.name_scope('adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-2).minimize(cross_entropy)

    with tf.name_scope('Average_Accuracy'):
        correct_prediction = tf.cast(tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1)), tf.float32)
        accuracy = tf.reduce_mean(correct_prediction)

    trainwriter = tf.summary.FileWriter('./dnn/train')
    testwriter = tf.summary.FileWriter('./dnn/test')
    trainwriter.add_graph(tf.get_default_graph())
    tf.summary.scalar('Prediction_Accuracy', accuracy)
    tf.summary.scalar('Objective_Loss', cross_entropy)
    s = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(10):
            batch_train_df = train.sample(100)
            batch_ys = onehot_encoder(batch_train_df['category'].values.tolist())
            batch_xs = USencoder(batch_train_df['title'].values.tolist())

            train_step.run(feed_dict={x: batch_xs, y: batch_ys, rate: 0.5})
            train_summary = sess.run(s, feed_dict={x: batch_xs, y: batch_ys, rate: 0})
            trainwriter.add_summary(train_summary, epoch)

            test_summary = sess.run(s, feed_dict={x: test_x, y: test_y, rate: 0})
            testwriter.add_summary(test_summary, epoch)

            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y: batch_ys, rate: 0})
            print('step %d, training accuracy %g' % (epoch, train_accuracy))


        trainwriter.close()
        testwriter.close()

if __name__ == '__main__':
    main()

