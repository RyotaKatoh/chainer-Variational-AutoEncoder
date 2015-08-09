# chainer-Variational-AutoEncoder

Variational Auto Encoder implemented by Chainer

## Requirement

* Chainer

## Train

Start training the model using train_VAE.py, for example
```
$python train_VAE.py
```

## Generate data

You can generate data by giving a latent space vector.
For example,
```
$python generated.py --model [model/created_model.pkl]
```

## ToDO

* GPU implementation
