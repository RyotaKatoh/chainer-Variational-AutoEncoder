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
from VAE_YZ_X import VAE_YZ_X

import dataset

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',       type=str,   default="dataset")
parser.add_argument('--output_dir',     type=str,   default="model")
parser.add_argument('--dataset',        type=str,   default="mnist")
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

np.random.seed(123)


print 'VAE p(x|y, z) start'

if args.dataset == 'mnist':
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
    n_batch = 1000
    n_epochs    = 1000
    output_f = 'sigmoid'

elif args.dataset == "svhn":
    size = 32
    train_x, train_y, test_x, test_y = dataset.load_svhn(args.data_dir, binarize_y=True)
    n_x = train_x.shape[1]
    n_hidden = [500, 500]
    n_z = 300
    n_y = 10
    n_batch = 1000
    n_epochs = 1000
    output_f = 'sigmoid'

    #
    # x = {'x': np.hstack((train_x, extra_x)), 'y':np.hstack((train_y, extra_y))}
    # ndict.shuffleCols(x)

    #f_enc, f_dec, (x_sd, x_mean) = pp.preprocess_normalize01(train_x, True)
    # f_enc, f_dec, pca_params = pp.PCA(x['x'][:,:10000], cutoff=1000, toFloat=True)
    # ndict.savez(pca_params, logdir+'pca_params')




# initialize model
n_hidden_q = n_hidden
n_hidden_p = n_hidden


n_hidden_recog = n_hidden
n_hidden_gen   = n_hidden
n_layers_recog = len(n_hidden_recog)
n_layers_gen   = len(n_hidden_gen)

layers = {}


rec_layer_sizes = []
rec_layer_sizes += zip(n_hidden_recog[:-1], n_hidden_recog[1:])

layers['recog_x'] = F.Linear(train_x.shape[1], n_hidden_recog[0], nobias=True)
layers['recog_y'] = F.Linear(train_y.shape[1], n_hidden_recog[0])

for i, (n_incoming, n_outgoing) in enumerate(rec_layer_sizes):
    layers['recog_%i' % i] = F.Linear(n_incoming, n_outgoing)

layers['recog_mean']= F.Linear(n_hidden_recog[-1], n_z)
layers['recog_log'] = F.Linear(n_hidden_recog[-1], n_z)

# Generating model.
gen_layer_sizes = []
gen_layer_sizes += zip(n_hidden_gen[:-1], n_hidden_gen[1:])

layers['gen_y'] = F.Linear(train_y.shape[1], n_hidden_gen[0])
layers['gen_z'] = F.Linear(n_z, n_hidden_gen[0], nobias=True)

for i, (n_incoming, n_outgoing) in enumerate(gen_layer_sizes):
    layers['gen_%i' % i] = F.Linear(n_incoming, n_outgoing)

layers['gen_out'] = F.Linear(n_hidden_gen[-1], train_x.shape[1])

model = VAE_YZ_X(**layers)

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
        y_batch = train_y[indexes[i : i + n_batch]]

        if args.gpu >= 0:
            x_batch = cuda.to_gpu(x_batch)
            y_batch = cuda.to_gpu(y_batch)

        optimizer.zero_grads()


        rec_loss, kl_loss, output = model.forward_one_step(x_batch, y_batch, n_layers_recog, n_layers_gen, 'relu', 'relu', output_f, gpu=args.gpu)
        loss = rec_loss + kl_loss
        total_loss += loss
        loss.backward()
        optimizer.update()

    if args.gpu >= 0:
        total_losses[epoch-1] = cuda.to_cpu(total_loss.data)
    else:
        total_losses[epoch-1] = total_loss.data
    print rec_loss.data, kl_loss.data
    print total_loss.data
    print "time:", time.time()-t1

    if epoch % 100 == 0:
        model_path = "%s/%s_VAE_YZ_X_%d.pkl" % (args.output_dir, args.dataset, epoch)
        with open(model_path, "w") as f:
            pickle.dump(copy.deepcopy(model).to_cpu(), f)

loss_path = "%s/%s_VAE_YZ_X_loss.pkl" % (args.output_dir, args.dataset)
with open(loss_path, "wb") as f:
    pickle.dump(total_losses, f)
