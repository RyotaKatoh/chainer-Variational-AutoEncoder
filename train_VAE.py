#%%
import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import six

import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F
from VAE import VAE

import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default="dataset")
parser.add_argument('--output_dir',     type=str,   default="model")
parser.add_argument('--dataset',        type=str,   default="mnist")
parser.add_argument('--log_dir',        type=str,   default="log")
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

np.random.seed(123)

if args.dataset == 'mnist':
    data_dir = args.data_dir
    mnist = dataset.load_mnist_data(data_dir)
    all_x = np.array(mnist['data'], dtype=np.float32) /255
    all_y = np.array(mnist['target'], dtype=np.float32)
    train_x = all_x[:60000]
    train_y = all_y[:60000]
    test_x  = all_x[60000:]
    test_y  = all_y[60000:]

    size    = 28
    n_x     = size*size
    n_hidden= [500, 500]
    n_z     = 50
    n_y     = 10
    n_batch = 1000
    n_epochs    = 100

if args.dataset == 'faces':
    data_dir = args.data_dir
    faces = pickle.load(open("dataset/freyfaces/freyfaces.pkl", "rb"))
    train_x = faces.astype(np.float32)

    n_x = 20*28
    n_hidden = [500, 500]
    n_z = 50
    n_y = 10
    n_batch = 27
    n_epochs = 100


n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)

layers = {}

# Recognition model.
rec_layer_sizes = [(train_x.shape[1], n_hidden_recog[0])]
rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])
rec_layer_sizes += [(n_hidden_recog[-1], n_z)]

for i, (n_incoming, n_outgoing) in enumerate(rec_layer_sizes):
    layers['recog_%i' % i] = F.Linear(n_incoming, n_outgoing)

layers['log_sigma'] = F.Linear(n_hidden_recog[-1], n_z)

# Generating model.
gen_layer_sizes = [(n_z, n_hidden_gen[0])]
gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])
gen_layer_sizes += [(n_hidden_gen[-1], train_x.shape[1])]

for i, (n_incoming, n_outgoing) in enumerate(gen_layer_sizes):
    layers['gen_%i' % i] = F.Linear(n_incoming, n_outgoing)

model = VAE(**layers)

if args.gpu >= 0:
    cuda.init(args.gpu)
    model.to_gpu()


# use Adam
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())

total_losses = np.zeros(n_epochs, dtype=np.float32)

for epoch in xrange(1, n_epochs + 1):
    print('epoch', epoch)

    t1 = time.time()
    indexes = np.random.permutation(train_x.shape[0])
    total_rec_loss = 0.0
    total_kl_loss = 0.0
    total_loss = 0.0
    for i in xrange(0, train_x.shape[0], n_batch):
        x_batch = train_x[indexes[i : i + n_batch]]
        y_batch = x_batch

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()

        rec_loss, kl_loss, output = model.forward_one_step(x_batch, y_batch, n_layers_recog, n_layers_gen, 'relu', 'relu', gpu=args.gpu)
        loss = rec_loss + kl_loss
        total_loss += loss
        total_losses[epoch-1] = total_loss.data
        loss.backward()
        optimizer.update()


    rec_loss, kl_loss, _ = model.forward_one_step(x_batch, y_batch, n_layers_recog, n_layers_gen, 'relu', 'relu', gpu=args.gpu)
    print rec_loss.data, kl_loss.data
    print total_loss.data
    print "time:", time.time()-t1

    if epoch % 100 == 0:
        model_path = "%s/%s_%d.pkl" % (args.output_dir, args.dataset, epoch)
        with open(model_path, "w") as f:
            pickle.dump(copy.deepcopy(model).to_cpu(), f)

loss_path = "%s/%s_VAE_loss.pkl" % (args.output_dir, args.dataset)
with open(loss_path, "wb") as f:
    pickle.dump(total_losses, f)
