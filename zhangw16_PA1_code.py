import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.image as mpimg
import pickle

# load the data set
rootdir = 'E:/4/Deep Learning/data_prog2(1)/train_data'
file_list = os.listdir(rootdir)
train_label = np.loadtxt('E:/4/Deep Learning/data_prog2(1)/labels/train_label.txt')

# create the label vector
def y_label(label):
    length = len(label)
    y = np.zeros([5,length])
    for i in range(length):
        y[int(label[i]-1),i] = 1
    return y

# create the label in 1 in 5 form
train_label = np.float32(y_label(train_label))
# laod the training data and pre-process
length_train = len(file_list)
train_data = np.ones([length_train,785])
for i in range(length_train):
    im = mpimg.imread(rootdir + '/' + file_list[i])
    im = np.reshape(im,[1,784])
    train_data[i:i+1,0:784] = im / 255
train_data = np.float32(train_data)

# laod the data set and the testing data and pre-process
rootdir = 'E:/4/Deep Learning/data_prog2(1)/test_data'
file_list = os.listdir(rootdir)
test_label = np.loadtxt('E:/4/Deep Learning/data_prog2(1)/labels/test_label.txt')
test_label = np.float32(y_label(test_label))
length_test = len(file_list)
test_data = np.ones([length_test,785])
for i in range(length_test):
    im = mpimg.imread(rootdir + '/' + file_list[i])
    im = np.reshape(im,[1,784])
    test_data[i:i+1,0:784] = im / 255

test_data = np.float32(test_data)
#hyperparameter settings
lamb = 0.01
batch_size = length_train
learning_rate = 0.05
training_epochs = 5001
display_step = 100
#construct models
x = tf.placeholder('float32',[785,None])
y = tf.placeholder('float32',[5,None])
theta = tf.Variable(tf.zeros([785,5],dtype='float32')+0.001)
x_next = tf.matmul(theta,x,transpose_a=True)

sig = tf.exp(tf.matmul(theta,x,transpose_a=True))
# softmax function
softmax = tf.divide(sig,tf.reduce_sum(sig,0))
#standard gradient descent method
grad = tf.divide(tf.add(-tf.matmul(x,tf.subtract(y,softmax),transpose_b=True),tf.multiply(theta,lamb)),batch_size)
# new theta/weight
theta_next = tf.subtract(theta,learning_rate*grad)
# update theta/weight
theta_update = tf.assign(theta,theta_next)

y2 = tf.argmax(sig,0)
y3 = tf.argmax(y,0)
score = tf.reduce_mean(tf.cast(tf.equal(y2,y3),'float32'))
# check the wrong number
def Error_num(y2,test_label):
    out = np.zeros(5)
    for i in range(len(test_label)):
        if test_label[i] != y2[i]:
            out[test_label[i]] += 1
    score = np.zeros(5)
    for i in range(5):
        number = len(test_label) - np.count_nonzero(test_label-i)
        score[i] = out[i] / number
    
    return score
#run session
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    train_error = []
    test_error = []
    for epoch in range(training_epochs):
        ind = 0
        batch_x = train_data[ind:ind+batch_size,:]
        batch_x = batch_x.transpose()
        batch_y = train_label[:,ind:ind+batch_size]
        test = test_data.transpose()
        train = train_data.transpose()
        
        sess.run(theta_update, feed_dict={x:batch_x,y:batch_y})
        
        y2_1 = sess.run(y2,feed_dict={x:train,y:train_label})
        train_error.append(Error_num(y2_1,np.argmax(train_label,0)))
            
        y2_2 = sess.run(y2,feed_dict={x:test,y:test_label})
        test_error.append(Error_num(y2_2,np.argmax(test_label,0)))
        
        if (epoch+1) % display_step == 0:
            #classification error
            score1 = sess.run(score,feed_dict={x:test,y:test_label})
            print('epoch:%d' % epoch, 'score:%f' % score1)
    Y = sess.run(y2,feed_dict={x:test,y:test_label})
    weight = sess.run(theta)
    
    #saving of the trained weights
    filehandler = open('multiclass_parameters.txt','wb')
    pickle.dump(weight,filehandler)
    filehandler.close()
#%% plot the weights
for i in range(5):
    image = weight[0:784,i:i+1]
    image= image.reshape(28,28)
    plt.subplot(1,5,i+1)
    plt.imshow(image)
plt.colorbar()
plt.show()
#%% plot the training error
train_error = np.array(train_error)
for i in range(5):
    plt.plot(train_error[:,i])
    plt.xlabel('iteration')
    plt.ylabel('training error for digit ' + str(i))
    plt.show()
#%% plot the testing error
test_error = np.array(test_error)
for i in range(5):
    plt.plot(test_error[:,i])
    plt.xlabel('iteration')
    plt.ylabel('testing error for digit ' + str(i))
    plt.show()