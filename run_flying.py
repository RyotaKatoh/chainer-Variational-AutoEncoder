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
parser.add_argument('--dataset',        type=str,   default="svhn")
parser.add_argument('--gpu',            type=int,   default=-1)
parser.add_argument('--output_file',    type=str,   default="flying.mp4")
parser.add_argument('--output_dir',       type=str,   default="flying")

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
    n_batch_w = 7
    x_size  = size*size
    output_f= 'sigmoid'
    n_layers_recog = n_layers_gen = len(n_hidden)
    output_image = np.zeros((n_batch_w*imsize[0], n_y*im_size[1]))

if args.dataset == 'svhn':
    size = 32
    im_size = (size, size, 3)
    train_x, train_y, test_x, test_y = dataset.load_svhn(args.data_dir, binarize_y=True)
    # pca = pickle.load(open(args.data_dir+"/SVHN/pca.pkl"))
    n_x = train_x.shape[1]
    n_hidden = [500, 500]
    n_z = 300
    n_y = 10
    n_batch_w = 7
    x_size = size*size*3
    output_f = 'sigmoid'
    n_layers_recog = n_layers_gen = len(n_hidden)
    output_image = np.zeros((n_batch_w*im_size[0], n_y*im_size[1], im_size[2]))

model = pickle.load(open(args.model, "rb"))

output_dir = "%s_%s" % (args.output_dir, args.dataset)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# set y
y = np.zeros((n_batch_w*n_y, n_y))
for i in range(n_y):
    y[i::n_y, i] = 1

# test model
print "Test model"
z = np.random.standard_normal((n_batch_w, n_z))
zsmooth = z.copy()
smoothingfactor = 0.1
noise_var = 0.06


for i in range(2000):
    z = np.sqrt(1-noise_var)*z + np.sqrt(noise_var)*np.random.standard_normal(z.shape)
    zsmooth += smoothingfactor*(z - zsmooth)
    _z = np.repeat(zsmooth, n_y, axis=0)


    for j in range(n_batch_w*n_y):
        sample_z = _z[j].astype(np.float32)
        sample_y = y[j].astype(np.float32)

        output1 = model.generate_z_x(x_size, sample_z, sample_y, n_layers_recog, n_layers_gen, nonlinear_q='relu', nonlinear_p='relu', output_f='sigmoid', gpu=-1)

        if args.dataset == 'mnist':
            im = output1.reshape((size, size))
            output_image[im_size[0]*(j / n_y):im_size[0]*((j / n_y)+1), im_size[1]*(j % n_y):im_size[1]*((j % n_y)+1)] = im

        elif args.dataset == 'svhn':
            im = output1.reshape((3, size, size)).T.swapaxes(0,1)
            output_image[im_size[0]*(j / n_y):im_size[0]*((j / n_y)+1), im_size[1]*(j % n_y):im_size[1]*((j % n_y)+1), :] = im

    misc.imsave('%s/%d_gen.png' % (output_dir, i) , output_image)


os.system("ffmpeg -start_number 0 -i "+output_dir+"/%d_gen.png -c:v libx264 -pix_fmt yuv420p -r 30 "+output_dir+"/"+args.output_file)
print "Saved to "+args.output_file
print "Done."
