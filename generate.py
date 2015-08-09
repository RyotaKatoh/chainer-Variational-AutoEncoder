import os
import time
import numpy as np
import argparse
import cPickle as pickle
from scipy import misc

from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

parser = argparse.ArgumentParser()
parser.add_argument('--model',          type=str,   required=True)
parser.add_argument('--output_dir',     type=str,   default="generated")
parser.add_argument('--dataset',        type=str,   default="faces")
parser.add_argument('--gpu',            type=int,   default=-1)

args = parser.parse_args()

if args.dataset == 'faces':
    im_size = (28,20)

elif args.dataset == 'mnist':
    im_size = (28, 28)

model = pickle.load(open(args.model, "rb"))

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

n_sample = 100
n_z = model.log_sigma.W.shape[0]

sampleN = np.random.standard_normal((100, n_z)).astype(np.float32)
n_layers_gen = 2

generated_output = model.generate(sampleN, n_layers_gen, 'relu')


for i in range(n_sample):
    im = np.ones(im_size).astype(np.float32) - generated_output.data[i].reshape(im_size)
    misc.imsave('%s/%d.jpg'% (args.output_dir, i), im)
