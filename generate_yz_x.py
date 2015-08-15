import os
import time
import numpy as np
import argparse
import cPickle as pickle
from scipy import misc
import dataset
from sklearn import decomposition

from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   required=True)
parser.add_argument('--data_dir',       type=str,   default="dataset")
parser.add_argument('--output_dir',     type=str,   default="generated_yz_x")
parser.add_argument('--dataset',        type=str,   default="mnist")
parser.add_argument('--gpu',            type=int,   default=-1)
parser.add_argument('--n_samples',      type=int,   default=10)

args = parser.parse_args()

if args.dataset == 'mnist':
    im_size = (28, 28)
    data_dir = args.data_dir
    mnist = dataset.load_mnist_data(data_dir)
    all_x = np.array(mnist['data'], dtype=np.float32) /255
    all_y_tmp = np.array(mnist['target'], dtype=np.float32)
    all_y = np.zeros((all_x.shape[0], (np.max(all_y_tmp) + 1.0)), dtype=np.float32)
    for i in range(all_y_tmp.shape[0]):
        all_y[i][all_y_tmp[i]] = 1.

    train_x = all_x[:50000]
    train_y = all_y[:50000]
    valid_x = all_x[50000:60000]
    valid_y = all_y[50000:60000]
    test_x  = all_x[60000:]
    test_y  = all_y[60000:]

    size    = 28
    n_x     = size*size
    n_hidden= [500, 500]
    n_z     = 50
    n_y     = 10
    output_f= 'sigmoid'
    output_image = np.zeros((size, size*(n_y+1)))

if args.dataset == 'svhn':
    size = 32
    im_size = (size, size, 3)
    train_x, train_y, test_x, test_y = dataset.load_svhn(args.data_dir, binarize_y=True)
    pca = pickle.load(open(args.data_dir+"/SVHN/pca.pkl"))
    n_x = train_x.shape[1]
    n_hidden = [500, 500]
    n_z = 300
    n_y = 10
    output_f = 'sigmoid'
    output_image = np.zeros((size, size*(n_y+1), 3))


model = pickle.load(open(args.model, "rb"))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

indexes = np.random.permutation(test_x.shape[0])

n_sample = args.n_samples
sample_x = test_x[indexes[0:n_sample]]
sample_y = test_y[indexes[0:n_sample]]

n_layers_recog = n_layers_gen = len(n_hidden)


for i in range(n_sample):
    generated_output = model.generate(sample_x[i].reshape((1, sample_x.shape[1])), sample_y[i].reshape((1, sample_y.shape[1])), n_layers_recog, n_layers_gen, nonlinear_q='relu', nonlinear_p='relu', output_f=output_f)

    if args.dataset == 'mnist':
        im = sample_x[i].reshape(im_size)
        output_image[:,0:im_size[1]] = im
    elif args.dataset == 'svhn':
        im = sample_x[i].reshape((3, size, size)).T.swapaxes(0,1)
        output_image[:, 0:im_size[1], :] = im


    for j in range(sample_y.shape[1]):
        if args.dataset == 'mnist':
            im = generated_output[j].reshape(im_size)
            output_image[:, (j+1)*im_size[1]:(j+2)*im_size[1]] = im
        elif args.dataset == 'svhn':
            im = generated_output[j].reshape((3, size, size)).T.swapaxes(0,1)
            output_image[:, (j+1)*im_size[1]:(j+2)*im_size[1], :] = im

    misc.imsave('%s/%d.jpg' % (args.output_dir, i) , output_image)
