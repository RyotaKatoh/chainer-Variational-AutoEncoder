import os
import time
import numpy as np


from chainer import cuda, Variable, function, FunctionSet, optimizers
from chainer import functions as F

class VAE_YZ_X(FunctionSet):

    def __init__(self, **layers):
        super(VAE_YZ_X, self).__init__(**layers)

    def softplus(self, x):
        return F.log(F.exp(x) + 1)
    def identity(self, x):
        return x

    def forward_one_step(self, x_data, y_data, n_layers_recog, n_layers_gen, nonlinear_q='softplus', nonlinear_p='softplus', output_f = 'sigmoid', type_qx='gaussian', type_px='gaussian', gpu=-1):
        x = Variable(x_data)
        y = Variable(y_data)

        # set non-linear function
        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_activation = {'sigmoid': F.sigmoid, 'identity': self.identity, 'tanh': F.tanh}
        output_a_f = output_activation[output_f]

        hidden_q = [ nonlinear_f_q( self.recog_x( x ) + self.recog_y( y ) ) ]

        # compute q(z|x, y)

        for i in range(n_layers_recog-1):
            hidden_q.append(nonlinear_f_q(getattr(self, 'recog_%i' % i)(hidden_q[-1])))


        q_mean = getattr(self, 'recog_mean')(hidden_q[-1])
        q_log_sigma = 0.5 * getattr(self, 'recog_log')(hidden_q[-1])

        eps = np.random.normal(0, 1, (x.data.shape[0], q_log_sigma.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)

        eps = Variable(eps)
        z   = q_mean + F.exp(q_log_sigma) * eps

        # compute q(x |y, z)
        hidden_p = [ nonlinear_f_p( self.gen_y( y ) + self.gen_z( z ) ) ]

        for i in range(n_layers_gen-1):
            hidden_p.append(nonlinear_f_p(getattr(self, 'gen_%i' % i)(hidden_p[-1])))

        hidden_p.append(output_a_f(getattr(self, 'gen_out')(hidden_p[-1])))
        output = hidden_p[-1]

        rec_loss = F.mean_squared_error(output, x)
        KLD = -0.5 * F.sum(1 + q_log_sigma - q_mean**2 - F.exp(q_log_sigma)) / (x_data.shape[0]*x_data.shape[1])

        return rec_loss, KLD, output


    def generate(self, sample_x, sample_y, n_layers_recog, n_layers_gen, nonlinear_q='relu', nonlinear_p='relu', output_f='sigmoid', gpu=-1):
        x = Variable(sample_x)
        y = Variable(sample_y)

        # set non-linear function
        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_activation = {'sigmoid': F.sigmoid, 'identity': self.identity, 'tanh': F.tanh}
        output_a_f = output_activation[output_f]

        # compute q(z|x, y)
        hidden_q = [ nonlinear_f_q( self.recog_x( x ) + self.recog_y( y ) ) ]

        for i in range(n_layers_recog-1):
            hidden_q.append(nonlinear_f_q(getattr(self, 'recog_%i' % i)(hidden_q[-1])))


        q_mean = getattr(self, 'recog_mean')(hidden_q[-1])
        q_log_sigma = 0.5 * getattr(self, 'recog_log')(hidden_q[-1])

        eps = np.random.normal(0, 1, (x.data.shape[0], q_log_sigma.data.shape[1])).astype('float32')
        if gpu >= 0:
            eps = cuda.to_gpu(eps)

        eps = Variable(eps)
        z   = q_mean + F.exp(q_log_sigma) * eps

        outputs = np.zeros((sample_y.shape[1], sample_x.shape[1]), dtype=np.float32)

        for label in range(sample_y.shape[1]):
            sample_y = np.zeros((1, sample_y.shape[1]), dtype=np.float32)
            sample_y[0][label] = 1.

            # compute q(x |y, z)
            hidden_p = [ nonlinear_f_p( self.gen_y( Variable(sample_y) ) + self.gen_z( z ) ) ]

            for i in range(n_layers_gen-1):
                hidden_p.append(nonlinear_f_p(getattr(self, 'gen_%i' % i)(hidden_p[-1])))

            hidden_p.append(output_a_f(getattr(self, 'gen_out')(hidden_p[-1])))
            output = hidden_p[-1]

            outputs[label] = output.data

        return outputs

    def generate_z_x(self, x_size, sample_z, sample_y, n_layers_recog, n_layers_gen, nonlinear_q='relu', nonlinear_p='relu', output_f='sigmoid', gpu=-1):

        # set non-linear function
        nonlinear = {'sigmoid': F.sigmoid, 'tanh': F.tanh, 'softplus': self.softplus, 'relu': F.relu}
        nonlinear_f_q = nonlinear[nonlinear_q]
        nonlinear_f_p = nonlinear[nonlinear_p]

        output_activation = {'sigmoid': F.sigmoid, 'identity': self.identity, 'tanh': F.tanh}
        output_a_f = output_activation[output_f]

        # input variables
        z = Variable(sample_z.reshape((1, sample_z.shape[0])))
        y = Variable(sample_y.reshape((1, sample_y.shape[0])))

        outputs = np.zeros((1, x_size), dtype=np.float32)

        # compute q(x |y, z)
        hidden_p = [ nonlinear_f_p( self.gen_y( y ) + self.gen_z( z ) ) ]

        for i in range(n_layers_gen-1):
            hidden_p.append(nonlinear_f_p(getattr(self, 'gen_%i' % i)(hidden_p[-1])))

        hidden_p.append(output_a_f(getattr(self, 'gen_out')(hidden_p[-1])))
        output = hidden_p[-1]

        outputs = output.data

        return outputs
