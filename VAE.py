import os
import time
import numpy as np


from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

class VAE(FunctionSet):

    def __init__(self, **layers):
        super(VAE, self).__init__(**layers)

    def softplus(self, x):
        return F.log(F.exp(x) + 1)

    def forward_one_step(self, x_data, y_data, n_layers_recog, n_layers_gen, nonlinear_q='softplus', nonlinear_p='softplus', gpu=False):
        inputs = Variable(x_data)
        y = Variable(y_data)

        # set non-linear function
        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        chain = [inputs]

        # compute q(z|x, y)
        for i in range(n_layers_recog):
            chain.append(nonlinear_f_q(getattr(self, 'recog_%i' % i)(chain[-1])))

        recog_out = getattr(self, 'recog_%i' % n_layers_recog)(chain[-1])

        log_sigma_out = 0.5 * (getattr(self, 'log_sigma')(chain[-1]))

        eps = np.random.normal(0, 1, (inputs.data.shape[0], log_sigma_out.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)
        eps = Variable(eps)
        z   = recog_out + F.exp(log_sigma_out) * eps

        chain  += [recog_out, z]

        for i in range(n_layers_gen):
            chain.append(nonlinear_f_p(getattr(self, 'gen_%i' % i)(chain[-1])))

        chain.append(F.sigmoid(getattr(self, 'gen_%i' % (n_layers_gen))(chain[-1])))
        output = chain[-1]


        rec_loss = F.mean_squared_error(output, y)
        KLD = -0.5 * F.sum(1 + log_sigma_out - recog_out**2 - F.exp(log_sigma_out)) / (x_data.shape[0]*x_data.shape[1])

        return rec_loss, KLD, output


    def generate(self, latent_data, n_layers_gen, nonlinear_p='softplus'):
        latent = Variable(latent_data)
        chain = [latent]

        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_p = nonlinear[nonlinear_p]

        for i in range(n_layers_gen):
            chain.append(nonlinear_f_p(getattr(self, 'gen_%i' % i)(chain[-1])))

        chain.append(F.sigmoid(getattr(self, 'gen_%i' % (n_layers_gen))(chain[-1])))

        output = chain[-1]

        return output
