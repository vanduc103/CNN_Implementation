import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_dense = labels_dense.astype(int)
    labels_one_hot[np.arange(num_labels), labels_dense] = 1
    
    return labels_one_hot

def preproc(unclean_batch_x):
    """Convert values to range 0-1"""
    temp_batch = unclean_batch_x / unclean_batch_x.max()
    
    return temp_batch

def batch_creator(batch_size, dataset_length):
    """Create batch with random samples and return appropriate format"""
    batch_mask = rng.choice(dataset_length, batch_size)
    
    batch_x = train_x[[batch_mask]]
    batch_x = preproc(batch_x)

    batch_y = train_y[[batch_mask]]    
    batch_y = dense_to_one_hot(batch_y)
    
    return batch_x, batch_y

# random number
seed = 128
rng = np.random.RandomState(seed)

#load all training batches
X_train = np.empty([50000, 3072])
Y_train = np.empty([50000])
for b in range(1,6):
  batch = unpickle(os.path.join('../file/cifar-10-batches-py', 'data_batch_%d' % b))
  i = b - 1
  X_train[i*10000:(i+1)*10000,:] = batch['data']
  Y_train[i*10000:(i+1)*10000] = np.array(batch['labels'])
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
Y_train = Y_train.astype("uint8")

# split to train set and validate set
split_size = int(X_train.shape[0]*0.8)

train_x, val_x = X_train[:split_size], X_train[split_size:]
train_y, val_y = Y_train[:split_size], Y_train[split_size:]
#print(train_x.shape)
#print(val_x.shape)
#print(train_y.shape)

# test set
batch_test = unpickle('../file/cifar-10-batches-py/test_batch')
X_test = batch_test['data']
Y_test = np.array(batch_test['labels'])
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float32")
#print(X_test.shape)
#print(Y_test.shape)
print('Read data done !')

### define the layers
image_size = 32
filter_num = 32
filter_size = 5
fc3_size = 384
fc4_size = 192
output_classes = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, image_size, image_size, 3])
y = tf.placeholder(tf.float32, [None, output_classes])

# set remaining variables
epochs = 10
batch_size = 100
learning_rate = 0.1

### weight initialization
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


### define model
# convolution-pooling layer define
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# convolution-pooling layer #1
W_conv1 = weight_variable([filter_size, filter_size, 3, filter_num])
b_conv1 = bias_variable([filter_num])
conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
pool1 = max_pool_2x2(conv1)

# convolution-pooling layer #2
W_conv2 = weight_variable([filter_size, filter_size, 32, filter_num])
b_conv2 = bias_variable([filter_num])
conv2 = tf.nn.relu(conv2d(pool1, W_conv2) + b_conv2)
pool2 = max_pool_2x2(conv2)

# fully connected #3
pool2_reshape = tf.reshape(pool2, [-1, 8*8*32])
pool2_dim = pool2_reshape.get_shape()[1].value
W_fc3 = tf.get_variable(name='W_fc3', shape=[pool2_dim, fc3_size], 
		initializer=tf.contrib.layers.xavier_initializer())
b_fc3 = tf.Variable(tf.zeros(fc3_size))
fc3 = tf.nn.relu(tf.add(tf.matmul(pool2_reshape, W_fc3), b_fc3))

# fully connected #4
W_fc4 = tf.get_variable(name='W_fc4', shape=[fc3_size, fc4_size],   
                initializer=tf.contrib.layers.xavier_initializer())
b_fc4 = tf.Variable(tf.zeros(fc4_size))
fc4 = tf.nn.relu(tf.add(tf.matmul(fc3, W_fc4), b_fc4))

# output layer
W_output = tf.get_variable(name='W_output', shape=[fc4_size, output_classes],   
                initializer=tf.contrib.layers.xavier_initializer())
b_output = tf.Variable(tf.zeros(output_classes))
output_layer = tf.add(tf.matmul(fc4, W_output), b_output)

### loss function - cross entropy with softmax
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))

## optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.initialize_all_variables()

# create session and run neural network on session
print('\n Training start ...')
with tf.Session() as sess:
  sess.run(init)
  
  ### for each epoch, do:
    ###   for each batch, do:
    ###     create pre-processed batch
    ###     run optimizer by feeding batch
    ###     find cost and reiterate to minimize
  graph_x = list()
  graph_y = list()
  for epoch in range(epochs):
    avg_cost = 0
    total_batch = int(train_x.shape[0]/batch_size)
    for i in range(total_batch):
      batch_x, batch_y = batch_creator(batch_size, train_x.shape[0])
      _, c = sess.run([optimizer, loss], feed_dict = {x: batch_x, y: batch_y})
            
      avg_cost += c / total_batch
    
    print "Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost)
    # compute accuracy on validate set
    prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
    print "Validation accuracy:", accuracy.eval({x: preproc(val_x), y: dense_to_one_hot(val_y)})
    graph_x.append(epoch+1)
    graph_y.append(avg_cost)

  print "\nTraining complete!"

  # show graph of loss value
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Graph of loss value')
  plt.grid(True)
  plt.plot(graph_x, graph_y)
  plt.show()

  # compute accuracy on test set
  prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(prediction, "float"))
  print "Test Accuracy:", accuracy.eval({x: preproc(X_test), y: dense_to_one_hot(Y_test)})
    


