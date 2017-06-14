import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

#load batch_1
batch_1 = unpickle('feedforward/file/cifar-10-batches-py/data_batch_1')
X = batch_1['data']
Y = batch_1['labels']
X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")
#X = X.reshape(10000, 32, 32, 3).astype("uint8")
print(X[0:1][0].shape)
Y = np.array(Y)

#load meta
meta = unpickle('feedforward/file/cifar-10-batches-py/batches.meta')
labels = meta['label_names']

#Visualizing CIFAR 10
fig = plt.figure()
i = np.random.choice(range(len(X)))
ax = fig.add_subplot(121)
ax.set_title(labels[Y[i]])
ax.imshow(X[i:i+1][0])

i = np.random.choice(range(len(X)))
ax = fig.add_subplot(122)
ax.set_title(labels[Y[i]])
ax.imshow(X[i:i+1][0])

plt.show()

