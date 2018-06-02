# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 20:45:07 2018

@author: admin
"""
import tensorflow as tf
import cv2
import sys
sys.path.append('game/')
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

GAME = 'bird'
ACTIONS = 2
GAMMA = 0.99
OBSERVER = 10000.
EXPLORE = 2000000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000
BATCH = 32
FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01,shape = shape)
    return tf.Variable(initial)
def conv2d(x,W,stride):
    return tf.nn.conv2d(x,W,strides = [1,stride,stride,1],padding = "SAME")
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
def createNetwork():
    W_conv1 = weight_variable([8,8,4,32])
    b_conv1 = bias_variable([32])
    
    W_conv2 = weight_variable([4,4,32,64])
    b_conv2 = bias_variable([64])
    
    W_conv3 = weight_variable([3,3,64,64])
    b_conv3 = bias_variable([64])
    
    W_fc1 = weight_variable([1600,512])
    b_fc1 = bias_variable([512])
    
    W_fc2 = weight_variable([512,ACTIONS])
    b_fc2 = bias_variable([ACTIONS])
    
    #输入层
    s = tf.placeholder("float",[None,80,80,4])
    
    #隐藏层
    h_conv1 = tf.nn.relu(conv2d(s,W_conv1,4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2,2) + b_conv2)
    
    h_conv3 = tf.nn.relu(conv2d(h_conv2,W_conv3,1) + b_conv3)
    
    h_conv3_flat = tf.reshape(h_conv3,[-1,1600])
    
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
    
    #输出层
    
    readout = tf.matmul(h_fc1,W_fc2) + b_fc2
    
    return s, readout, h_fc1

def trainNetwork(s,readout,h_fc1,sess):
    #定义损失函数
    a = tf.placeholder("float",[None,ACTIONS])
    y = tf.placeholder("float",[None])
    readout_action = tf.reduce_sum(tf.multiply(readout,a),
                                   reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)
    
    #开启游戏模拟器，打开模拟器窗口，实时显示游戏信息
    game_state = game.GameState()
    
    #创建双端队列用于存放replay memory
    D = deque()
    
    #获取游戏的初始状态，设置动作为不执行跳跃，并将初始状态修改成80*80*4大小
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x-t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t,(80,80)),cv2.COLOR_BGR2GRAY)
    ret,x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t,x_t,x_t,x_t),axis=2)
    
    #用于加载或保存网络参数
    
    save = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess,checkpoint.model_checkpoint_path)
        print("Successfully loaded:",checkpoint.model_checkpoint_path)
    else:
        print('could not find old network weights')
        
    epsilon = INITIAL_EPSILON
    t = 0
    while 'flappy bird' != 'angry bird':
        #使用epsilon贪心策略选择一个动作
        readout_t = readout.eval(feed_dict={s:[s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t%FRAME_PER_ACTION == 0:
            #执行一个随机动作
            if random.random() <= epsilon:
                print('----------------Random Action-------------')
                action_index = random.randrange(ACTIONS)
                a_t[random.randrange[ACTIONS]] = 1
            #由神经网络计算的Q(s,a)值选择对应的动作
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1 #不执行跳跃动作
    if epsilon > FINAL_EPSILON and t > OBSERVER:
        epsilon -= (INITIAL_EPSILON-FINAL_EPSILON) / EXPLORE
    
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored,(80,80)),cv2.COLOR_BGR2GRAY)
    ret,x_t1 = cv2.threshold(x_t1,1,255,cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1,(80,80,1))
    s_t1 = np.append(x_t1,s_t[:,:,:3],axis=2)
    
    #将状态转移过程存储到D中，用于更新参数时采样
    D.append((s_t,a_t,r_t,s_t1,terminal))
    if len(D) > REPLAY_MEMORY:
        D.popleft()
        
    if t>OBSERVER:
        #从D中随机采样，用于参数更新
        minibatch = random.sample(D,BATCH)
        
        #分别将当前状态，采取的动作，获得的回报，下一状态分组存放
        s_j_batch = [d[0] for d  in minibatch]
        a_batch = [d[1] for d  in minibatch]
        r_batch = [d[2] for d  in minibatch]
        s_j1_batch = [d[3] for d  in minibatch]
        
        #计算Q(s,a)的新值
        y_batch = []
        readout_j1_batch = readout.eval(feed_dict={s:s_j1_batch})
        for i in range(0,len(minibatch)):

            
        