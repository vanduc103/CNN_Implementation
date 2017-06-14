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
    
    batch_x = train_x[[batch_mask]].reshape(-1, input_num_units)
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
X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
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
X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
#print(X_test.shape)
#print(Y_test.shape)
print('Read data done !')

### set all variables

# number of neurons in each layer
input_num_units = 32*32*3
hidden_num_units = [1024, 384, 192, 64]
output_num_units = 10

# define placeholders
x = tf.placeholder(tf.float32, [None, input_num_units])
y = tf.placeholder(tf.float32, [None, output_num_units])

# set training parameters
epochs = 15
batch_size = 100
learning_rate = 0.15

### define weights and biases
weights = {
  'hidden1': tf.get_variable('w_hidden1', shape=[input_num_units, hidden_num_units[0]], 
		initializer=tf.contrib.layers.xavier_initializer()),
  'hidden2': tf.get_variable('w_hidden2', shape=[hidden_num_units[0], hidden_num_units[1]],
                initializer=tf.contrib.layers.xavier_initializer()),
  'hidden3': tf.get_variable('w_hidden3', shape=[hidden_num_units[1], hidden_num_units[2]],
                initializer=tf.contrib.layers.xavier_initializer()),
  'hidden4': tf.get_variable('w_hidden4', shape=[hidden_num_units[2], hidden_num_units[3]],
                initializer=tf.contrib.layers.xavier_initializer()),
  'output': tf.get_variable('w_output', shape=[hidden_num_units[3], output_num_units],
                initializer=tf.contrib.layers.xavier_initializer())
}

biases = {
  'hidden1': tf.Variable(tf.zeros(hidden_num_units[0]), name='b_hidden1'),
  'hidden2': tf.Variable(tf.zeros(hidden_num_units[1]), name='b_hidden2'),
  'hidden3': tf.Variable(tf.zeros(hidden_num_units[2]), name='b_hidden3'),
  'hidden4': tf.Variable(tf.zeros(hidden_num_units[3]), name='b_hidden4'),
  'output' : tf.Variable(tf.zeros(output_num_units), name='b_output')
}

### create our neural networks computational graph
hidden_layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1']))

hidden_layer2 = tf.nn.relu(tf.add(tf.matmul(hidden_layer1, weights['hidden2']), biases['hidden2']))

hidden_layer3 = tf.nn.relu(tf.add(tf.matmul(hidden_layer2, weights['hidden3']), biases['hidden3']))

hidden_layer4 = tf.nn.relu(tf.add(tf.matmul(hidden_layer3, weights['hidden4']), biases['hidden4']))

output_layer = tf.matmul(hidden_layer4, weights['output']) + biases['output']
output_layer = tf.nn.softmax(output_layer)

### loss function - cross entropy with softmax
#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(output_layer), reduction_indices=[1]))

## optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

# initialize all variables
init = tf.initialize_all_variables()

# create session and run neural network on session
print('\nTraining start ...')
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
    print "Validation accuracy:", accuracy.eval({x: preproc(val_x.reshape(-1, input_num_units)), y: dense_to_one_hot(val_y)})
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
  print "Test Accuracy:", accuracy.eval({x: preproc(X_test.reshape(-1, input_num_units)), y: dense_to_one_hot(Y_test)})
    

