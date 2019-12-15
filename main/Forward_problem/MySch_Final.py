"""
@Written by Hanxun Jin for CS1470 Final Project

Reference: 
Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. 
"Physics-informed neural networks: A deep learning framework for solving forward 
and inverse problems involving nonlinear partial differential equations." 
Journal of Computational Physics 378 (2019): 686-707.
"""

import tensorflow as tf
import numpy as np
from pyDOE import lhs
from matplotlib import pyplot as plt
from scipy.interpolate import griddata
import scipy.io
from keras import initializers
import os

class PDENET(tf.keras.Model):
    def __init__(self):
        super(PDENET, self).__init__()

        # Initialize all hyperparameters
        self.learning_rate = 0.001
        self.hidden_layer = 30
        self.epochN = 100000
		
		#Weights for fully connected layer
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer='glorot_uniform',activation='tanh')
        self.dense2 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer='glorot_uniform',activation='tanh')
        self.dense3 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer='glorot_uniform',activation='tanh')
        self.dense4 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer='glorot_uniform',activation='tanh')
        self.dense5 = tf.keras.layers.Dense(2)

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)


    #@tf.function
    def network_f(self,inputs):
 
        x = inputs[:,0:1]
        t = inputs[:,1:2]

        x = tf.convert_to_tensor(x)
        t = tf.convert_to_tensor(t)

        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(t)
            X = tf.concat([x,t],1)

            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(t)

                X = tf.concat([x,t],1)
                layer1 = self.dense1(X) 
                layer2 = self.dense2(layer1)
                layer3 = self.dense3(layer2)
                layer4 = self.dense4(layer3)
                layer5 = self.dense5(layer4)

                u = layer5[:,0:1]
                v = layer5[:,1:2]
                
            u_x = t2.gradient(u, x)
            u_t = t2.gradient(u, t)
            v_x = t2.gradient(v, x)
            v_t = t2.gradient(v, t)
        
        u_xx = t1.gradient(u_x, x)
        v_xx = t1.gradient(v_x, x)


        u = tf.dtypes.cast(u, dtype = tf.float64)
        v = tf.dtypes.cast(v, dtype = tf.float64)


        #f_u = -v_t + 0.5*u_xx + (u**2 + v**2)*u

        #f_v = u_t + 0.5*v_xx + (u**2 + v**2)*v

        f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u 


        return f_u,f_v,u,v,u_x,v_x


    #@tf.function
    def call(self,X0,X_lb,X_ub,X_f):

        _,_,u0_pred,v0_pred, _ , _= self.network_f(X0)
        _,_,u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred = self.network_f(X_lb)
        _,_,u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred = self.network_f(X_ub)
        f_u_pred,f_v_pred, u_pred , v_pred,_,_ = self.network_f(X_f)


        return u0_pred,v0_pred,u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred,u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred,f_u_pred,f_v_pred,u_pred , v_pred

    def loss_function(self,u0_pred,v0_pred,u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred,u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred,f_u_pred,f_v_pred,u0_true,v0_true):
        
        Loss = tf.reduce_mean(tf.square(u0_true - u0_pred)) + \
                    tf.reduce_mean(tf.square(v0_true - v0_pred)) + \
                    tf.reduce_mean(tf.square(u_lb_pred - u_ub_pred)) + \
                    tf.reduce_mean(tf.square(v_lb_pred - v_ub_pred)) + \
                    tf.reduce_mean(tf.square(u_x_lb_pred - u_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(v_x_lb_pred - v_x_ub_pred)) + \
                    tf.reduce_mean(tf.square(f_u_pred)) + \
                    tf.reduce_mean(tf.square(f_v_pred))

        return Loss


def train(model,X0,X_lb,X_ub,X_f,u0_true,v0_true):
    with tf.GradientTape() as tape:
        u0_pred,v0_pred,u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred,u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred,f_u_pred,f_v_pred,_,_ = model.call(X0,X_lb,X_ub,X_f)
        Loss = model.loss_function(u0_pred,v0_pred,u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred,u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred,f_u_pred,f_v_pred,u0_true,v0_true)
    gradients = tape.gradient(Loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return Loss

def test(model,X0,X_lb,X_ub,Xstar):
    
    _,_,_,_,_,_,_,_,_,_,_,_, u_pred,v_pred = model.call(X0,X_lb,X_ub,Xstar)

    return u_pred,v_pred


def main():
    
    data = scipy.io.loadmat('data/NLS.mat') # the data set contains the exact solution
    os.mkdir("Output/")
   
    # Doman bounds
    lb = np.array([-5.0, 0.0]) #lower bound
    ub = np.array([5.0, np.pi/2]) #upper bound
    N0 = 50 # number of initial value data
    Nb = 50 # number of boundary value data
    Nf = 20000 # number of data inside domain

    t = data['tt'].flatten()[:,None]
    x = data['x'].flatten()[:,None]
    Exact = data['uu']
    Exact_u = np.real(Exact)
    Exact_v = np.imag(Exact)
    Exact_h = np.sqrt(Exact_u**2 + Exact_v**2)
    
    X, T = np.meshgrid(x,t)

    X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
    u_star = Exact_u.T.flatten()[:,None]
    v_star = Exact_v.T.flatten()[:,None]
    h_star = Exact_h.T.flatten()[:,None]
    

    # select the initial value data
    idx_x = np.random.choice(x.shape[0], N0, replace=False)
    x0 = x[idx_x,:]
    u0 = Exact_u[idx_x,0:1]
    v0 = Exact_v[idx_x,0:1]
    
    #print("x0",x0.shape)
    # select the boundary value data
    idx_t = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[idx_t,:]
    # select the data inside domain pairs, it contains Nf pairs of (x,t)
    X_f = lb + (ub-lb)*lhs(2, Nf)

    #print("u_star",u_star.shape)
    #Concatenate all data into (x,t) pairs
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

    model = PDENET()
	#create the model
    for i in range (0,model.epochN):
        rLoss = train(model,X0,X_lb,X_ub,X_f,u0,v0)
        if i % 1000 == 0:
            print("The {}th epoch is finished, the loss is:{}".format(i+1,rLoss.numpy()))
            u_value,v_value= test(model,X0,X_lb,X_ub,X_star)
        
            h_value = np.sqrt(u_value**2 + v_value**2)

            u_value = u_value.numpy()
            v_value = v_value.numpy()
            #f_u_value = f_u_value.numpy()
            #f_v_value = f_v_value.numpy()
            #print("u_value",u_value.shape)

            error_h = np.linalg.norm(h_star-h_value,2)/np.linalg.norm(h_star,2)
            error_u = np.linalg.norm(u_star-u_value,2)/np.linalg.norm(u_star,2)
            error_v = np.linalg.norm(v_star-v_value,2)/np.linalg.norm(v_star,2)
            
            #print('Error u: %e' % (error_u))
            #print('Error v: %e' % (error_v))
            print('Error h: %e' % (error_h))
            #U_pred = griddata(X_star, u_value.flatten(), (X, T), method='cubic')
            #V_pred = griddata(X_star, v_value.flatten(), (X, T), method='cubic')
            H_pred = griddata(X_star, h_value.flatten(), (X, T), method='cubic')

            #FU_pred = griddata(X_star, f_u_value.flatten(), (X, T), method='cubic')
            #FV_pred = griddata(X_star, f_v_value.flatten(), (X, T), method='cubic') 

    
            plt.plot(x,Exact_h[:,75],'b-', linewidth = 2, label = 'Exact')
            plt.plot(x,H_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
            plt.ylabel('|h(x,t)|')
            plt.xlabel('x')
            plt.title('Schrodinger Equation at t=75')
            plt.show(block=False)
            path = "Output/" + str(i) + ".png"
            plt.savefig(path)
            plt.pause(1)
            plt.close()

if __name__ == '__main__':
	main()