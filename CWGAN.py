# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 21:33:14 2018

@author: WZW
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
from collections import defaultdict
from re import compile,findall,split
import random

sess = tf.InteractiveSession()



with open("ratings.txt") as f:
    ratings = f.readlines()
order = ('0 1 2').strip().split()
trainingData = defaultdict(dict)
trainingSet_i = defaultdict(dict)
user_num = 0
item_num = 0
item_most = 0
for lineNo, line in enumerate(ratings):
    items = split(' |,|\t', line.strip())
    userId = items[int(order[0])]
    itemId = items[int(order[1])]
    if int(itemId) > item_most:
        item_most = int(itemId)
    rating  = items[int(order[2])]
    trainingData[userId][itemId]=float(rating)/float(5)
    
#print(trainingData)
for i,user in enumerate(trainingData):
    for item in trainingData[user]:
        trainingSet_i[item][user] = trainingData[user][item]
#print(trainingSet_i)
print(len(trainingData),item_most)
x = np.zeros((len(trainingData),item_most))
Y = np.zeros((len(trainingData),10))

for user in trainingData:
    for item in trainingData[user]:
        x[int(user)-1][int(item)-1] = trainingData[user][item]


filename = 'ratingsgan100percentactive.txt'
with open(filename,'w') as f:
    f.writelines(ratings)
    
with open("labels.txt") as f:
    times = f.readlines()

for i in range(len(times)):
    line = times[i].strip('\n').split(' ')
    Y[int(line[0])-1][int(line[1])] = 1
print(x.shape,Y.shape)
print(x[0],Y[0])

mb_size = 100
Z_dim = 100
X_dim = item_most
y_dim = Y.shape[1]
h_dim = 128


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, item_most])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, 1]))
D_b2 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_b1, D_b2]


def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    out = tf.matmul(D_h1, D_W2) + D_b2
    return out



""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_b1, G_b2]


def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
    G_prob = tf.nn.sigmoid(G_log_prob)

    return G_prob



G_sample = generator(Z, y)
D_real = discriminator(X, y)
D_fake = discriminator(G_sample, y)

D_loss = tf.reduce_mean(D_real) - tf.reduce_mean(D_fake)
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(-D_loss, var_list=theta_D))
G_solver = (tf.train.RMSPropOptimizer(learning_rate=1e-4)
            .minimize(G_loss, var_list=theta_G))

clip_D = [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in theta_D]

def sample_Z(m, n):
    '''Uniform prior for G(Z)'''
    return np.random.uniform(-1., 1., size=[m, n])


sess.run(tf.global_variables_initializer())

i = 0
has_user = 0
user_num = len(trainingData)
for it in range(1000000):
    if it % 1000 == 0:
        n_sample = int(len(trainingData)*1)

        Z_sample = sample_Z(n_sample, Z_dim)
        y_sample = np.zeros(shape=[n_sample, y_dim])
        y_sample[:, 7] = 1

        samples = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})

        print(samples)
        
    choose_user = random.sample(range(1,len(trainingData)), mb_size)
    #print(choose_user)
    X_mb = []
    y_mb = []
    for i in range(mb_size):
        X_mb.append(x[choose_user[i]])
        y_mb.append(Y[choose_user[i]])
    Z_sample = sample_Z(mb_size, Z_dim)
    _, D_loss_curr,_ = sess.run([D_solver, D_loss, clip_D], feed_dict={X: np.array(X_mb), Z: Z_sample, y:np.array(y_mb)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:np.array(y_mb)})
    if abs(G_loss_curr) < 1e-3 and abs(D_loss_curr) < 1e-3:
        filename = 'ratingscwgan100percentactive.txt'
        print(it)
        print(G_loss_curr,D_loss_curr)
        print(samples)
        with open(filename,'a') as f:
            rating_gan = []
            for i in samples:
                if np.sum(i > 0.2) > 100:
                    has_user += 1
                    user_num += 1
                    print(has_user)
                    for m,n in enumerate(list(i)):
                        if n >= 0.2:
                            rating_gan.append(str(user_num)+' '+str(m+1)+' '+str(round(n*5))+'\n')
                            
            f.writelines(rating_gan)
        if has_user == 1*int(len(trainingData)):
            break
    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()