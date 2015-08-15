# chainer-Variational-AutoEncoder

Variational Auto Encoder implemented by Chainer

## Requirement

* Chainer

## M1 model

### Train

Start training the model using train_VAE.py, for example
```
$python train_VAE.py
```

### Generate data

You can generate data by giving a latent space vector.
For example,
```
$python generated.py --model [model/created_model.pkl]
```

## M2 model

### Train

Start training the model using train_VAE_yz_x.py, for example
```
$python train_VAE_yz_x.py
```

### Generate data set giving 1 sample input.

You can generate data set by giving a sample input.
For example,
```
$python generated_yz_x.py --model [model/created_model.pkl]
```

### Flying through latent space of M2 model

To generate movies of flying through latent-space of the M2 model, run:

```
$python run_flying.py --dataset [dataset] --model [model/created_model.pkl] --output_file [output file name]
```
where dataset is 'mnist' or 'svhn', and output_file is the filename to save the movie file to.
NOTE: This script requires ffmpeg to be installed.

## ToDo

* GPU implementation

## Reference

* Justin Bayer's Chainer based Variational Auto Encoder
http://nbviewer.ipython.org/gist/duschendestroyer/a41fcab5f7f9ffa45387
* http://papers.nips.cc/paper/5352-semi-supervised-learning-with-deep-generative-models.pdf
* http://www.slideshare.net/beam2d/semisupervised-learning-with-deep-generative-models
