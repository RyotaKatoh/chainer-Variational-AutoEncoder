import gzip
import os

import numpy as np
import cPickle as pickle
import six
from six.moves.urllib import request
import scipy
from scipy import io
from sklearn import decomposition

parent = 'http://yann.lecun.com/exdb/mnist'
train_images = 'train-images-idx3-ubyte.gz'
train_labels = 'train-labels-idx1-ubyte.gz'
test_images = 't10k-images-idx3-ubyte.gz'
test_labels = 't10k-labels-idx1-ubyte.gz'
num_train = 60000
num_test = 10000
dim = 784

'''

BVH

'''
def load_bvh_data(file_path):

    frames = 0
    frame_time = 0.0


    with open(file_path, "rb") as f:
        lines = f.readlines()

        n = 0
        while lines[n].find('MOTION') < 0:
            n += 1

            assert n < len(lines)

        # frames
        n += 1
        frames = int(lines[n].split(" ")[-1].replace('\n', ''))

        # frame time
        n += 1
        frame_time = float(lines[n].split(" ")[-1].replace('\n', ''))

        # motion data
        n += 1
        for i in range(frames):
            motion = lines[n + i].split(' ')

            if i == 0:
                dim = len(motion)
                global motion_data
                motion_data = np.zeros(frames * dim, dtype=np.float32).reshape((frames, dim))

            for j in range(dim):
                motion_data[i, j] = float(motion[j].replace('\n', ''))

    return frames, frame_time, motion_data




'''

MNIST

'''

def load_mnist(images, labels, num):
    data = np.zeros(num * dim, dtype=np.uint8).reshape((num, dim))
    target = np.zeros(num, dtype=np.uint8).reshape((num, ))

    with gzip.open(images, 'rb') as f_images,\
            gzip.open(labels, 'rb') as f_labels:
        f_images.read(16)
        f_labels.read(8)
        for i in six.moves.range(num):
            target[i] = ord(f_labels.read(1))
            for j in six.moves.range(dim):
                data[i, j] = ord(f_images.read(1))

    return data, target


def download_mnist_data(data_dir):
    print('Downloading {:s}...'.format(train_images))
    request.urlretrieve('{:s}/{:s}'.format(parent, train_images), train_images)
    print('Done')
    print('Downloading {:s}...'.format(train_labels))
    request.urlretrieve('{:s}/{:s}'.format(parent, train_labels), train_labels)
    print('Done')
    print('Downloading {:s}...'.format(test_images))
    request.urlretrieve('{:s}/{:s}'.format(parent, test_images), test_images)
    print('Done')
    print('Downloading {:s}...'.format(test_labels))
    request.urlretrieve('{:s}/{:s}'.format(parent, test_labels), test_labels)
    print('Done')

    print('Converting training data...')
    data_train, target_train = load_mnist(train_images, train_labels,
                                          num_train)
    print('Done')
    print('Converting test data...')
    data_test, target_test = load_mnist(test_images, test_labels, num_test)
    mnist = {}
    mnist['data'] = np.append(data_train, data_test, axis=0)
    mnist['target'] = np.append(target_train, target_test, axis=0)

    print('Done')
    print('Save output...')
    with open('%s/mnist/mnist.pkl' % data_dir, 'wb') as output:
        six.moves.cPickle.dump(mnist, output, -1)
    print('Done')
    print('Convert completed')


def load_mnist_data(data_dir):
    if not os.path.exists('%s/mnist/mnist.pkl' % data_dir):
        download_mnist_data(data_dir)
    with open('%s/mnist/mnist.pkl' % data_dir, 'rb') as mnist_pickle:
        mnist = six.moves.cPickle.load(mnist_pickle)
    return mnist

'''
SVHN

'''

def svhn_pickle_checker(data_dir):
    if os.path.exists(data_dir+'/SVHN/train_x.pkl') and os.path.exists(data_dir+'/SVHN/train_y.pkl') \
        and os.path.exists(data_dir+'/SVHN/test_x.pkl') and os.path.exists(data_dir+'/SVHN/test_y.pkl'):
        return 1
    else:
        return 0

def load_svhn(data_dir, toFloat=True, binarize_y=True, dtype=np.float32, pca=True, n_components=1000):

    # if svhn_pickle_checker(data_dir) == 1:
    #     print "load from pickle file."
    #     train_x = pickle.load(open(data_dir+'/SVHN/train_x.pkl'))
    #     train_y = pickle.load(open(data_dir+'/SVHN/train_y.pkl'))
    #     test_x  = pickle.load(open(data_dir+'/SVHN/test_x.pkl'))
    #     test_y  = pickle.load(open(data_dir+'/SVHN/test_y.pkl'))
    #
    #     return train_x, train_y, test_x, test_y


    train = scipy.io.loadmat(data_dir+'/SVHN/train_32x32.mat')
    train_x = train['X'].swapaxes(0,1).T.reshape((train['X'].shape[3], -1))
    train_y = train['y'].reshape((-1)) - 1
    test = scipy.io.loadmat(data_dir+'/SVHN/test_32x32.mat')
    test_x = test['X'].swapaxes(0,1).T.reshape((test['X'].shape[3], -1))
    test_y = test['y'].reshape((-1)) - 1
    if toFloat:
        train_x = train_x.astype(dtype)/256.
        test_x = test_x.astype(dtype)/256.
    if binarize_y:
        train_y = binarize_labels(train_y)
        test_y = binarize_labels(test_y)

    # if pca:
    #     x_stack = np.vstack([train_x, test_x])
    #     pca = decomposition.PCA(n_components=n_components)
    #     pca.whiten=True
    #     # pca.fit(x_stack)
    #     # x_pca = pca.transform(x_stack)
    #     x_pca = pca.fit_transform(x_stack)
    #     train_x = x_pca[:train_x.shape[0], :]
    #     test_x = x_pca[train_x.shape[0]:, :]
    #
    #     with open('%s/SVHN/pca.pkl' % data_dir, "wb") as f:
    #         pickle.dump(pca, f)
    #     with open('%s/SVHN/train_x.pkl' % data_dir, "wb") as f:
    #         pickle.dump(train_x, f)
    #     with open('%s/SVHN/train_y.pkl' % data_dir, "wb") as f:
    #         pickle.dump(train_y, f)
    #     with open('%s/SVHN/test_x.pkl' % data_dir, "wb") as f:
    #         pickle.dump(test_x, f)
    #     with open('%s/SVHN/test_y.pkl' % data_dir, "wb") as f:
    #         pickle.dump(test_y, f)

    return train_x, train_y, test_x, test_y

def binarize_labels(y, n_classes=10):
    new_y = np.zeros((y.shape[0], n_classes))
    for i in range(y.shape[0]):
        new_y[i, y[i]] = 1
    return new_y.astype(np.float32)



'''

Shakespeare

'''
def load_shakespeare(data_dir):
    vocab = {}
    words = open('%s/tinyshakespeare/input.txt' % data_dir, 'rb').read()
    words = list(words)
    dataset = np.ndarray((len(words), ), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]

    return dataset, words, vocab
