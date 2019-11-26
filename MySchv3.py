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


class PDENET(tf.keras.Model):
    def __init__(self):
        super(PDENET, self).__init__()

        # Initialize all hyperparameters
        self.learning_rate = 0.05
        self.hidden_layer = 200
        self.epochN = 10000
		
		#Weights for fully connected layer
        self.dense1 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer=initializers.random_normal(stddev=0.1))
        self.dense2 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer=initializers.random_normal(stddev=0.1))
        self.dense3 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer=initializers.random_normal(stddev=0.1))
        self.dense4 = tf.keras.layers.Dense(self.hidden_layer,kernel_initializer=initializers.random_normal(stddev=0.1))
        self.dense5 = tf.keras.layers.Dense(2,kernel_initializer=initializers.random_normal(stddev=0.1))

        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

    def network_uv(self,inputs):

        x = inputs[:,0:1]
        t = inputs[:,1:2]

        x = tf.convert_to_tensor(x)
        t = tf.convert_to_tensor(t)
        
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            tape.watch(t)
            X = tf.concat([x,t],1)
            X = tf.dtypes.cast(X, dtype = tf.float64)
            layer1 = tf.nn.tanh(self.dense1(X)) 
            layer2 = tf.nn.tanh(self.dense2(layer1)) 
            layer3 = tf.nn.tanh(self.dense3(layer2)) 
            layer4 = tf.nn.tanh(self.dense4(layer3)) 
            layer5 = self.dense5(layer4)           
            u = layer5[:,0:1]
            v = layer5[:,1:2]

        u_x = tape.gradient(u, x)
        v_x = tape.gradient(v, x)

        u_x = tf.dtypes.cast(u_x, dtype = tf.float32)
        v_x = tf.dtypes.cast(v_x, dtype = tf.float32)
        return u,v,u_x,v_x


    def network_f(self,inputs):
 
        x = inputs[:,0:1]
        t = inputs[:,1:2]

        x = tf.convert_to_tensor(x)
        t = tf.convert_to_tensor(t)

        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            t1.watch(t)
            with tf.GradientTape(persistent=True) as t2:
                t2.watch(x)
                t2.watch(t)

                X = tf.concat([x,t],1)
                X = tf.dtypes.cast(X, dtype = tf.float32)

                layer1 = tf.nn.tanh(self.dense1(X)) 
                layer2 = tf.nn.tanh(self.dense2(layer1)) 
                layer3 = tf.nn.tanh(self.dense3(layer2)) 
                layer4 = tf.nn.tanh(self.dense4(layer3)) 
                layer5 = self.dense5(layer4)

                u = layer5[:,0:1]
                v = layer5[:,1:2]
                
            u_x = t2.gradient(u, x)
            u_t = t2.gradient(u, t)
            v_x = t2.gradient(v, x)
            v_t = t2.gradient(v, t)
        
        u_xx = t1.gradient(u_x, x)
        v_xx = t1.gradient(v_x, x)


        #print("u_xx",type(u_xx))
        #print("v_xx",type(v_xx))
        #print("u",type(u))
        #print("v",type(v))
        #print("u_t",type(u_t))
        #print("v_t",type(v_t))
        

        u_xx = tf.dtypes.cast(u_xx, dtype = tf.float32)
        v_xx = tf.dtypes.cast(v_xx, dtype = tf.float32)
        u_t = tf.dtypes.cast(u_t, dtype = tf.float32)
        v_t = tf.dtypes.cast(v_t, dtype = tf.float32)


        #print(u)
        #print ((u**2 + v**2)*v)
        #print(0.5*v_xx)
        #print(u_t)


        #print("u_xx", u_xx)
        #print("v_xx", v_xx)



        f_u = -v_t + 0.5*u_xx + (u**2 + v**2)*u

        f_v = u_t + 0.5*v_xx + (u**2 + v**2)*v

        #f_u = u_t + 0.5*v_xx + (u**2 + v**2)*v
        #f_v = v_t - 0.5*u_xx - (u**2 + v**2)*u   

        #print("v_t",tf.reduce_mean(v_t))
        #print("u_xx",tf.reduce_mean(u_xx))
        #print("u",tf.reduce_mean(u))
        #print("v",tf.reduce_mean(v))



        return f_u,f_v,u,v



    def call(self,X0,X_lb,X_ub,X_f,u0_true,v0_true):

        u0_pred,v0_pred, _ , _= self.network_uv(X0)
        u_lb_pred,v_lb_pred, u_x_lb_pred , v_x_lb_pred = self.network_uv(X_lb)
        u_ub_pred,v_ub_pred, u_x_ub_pred , v_x_ub_pred = self.network_uv(X_ub)
        f_u_pred,f_v_pred, _ , _ = self.network_f(X_f)

        #print("f_u",tf.reduce_mean(tf.square(f_u_pred)))
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
        Loss = model.call(X0,X_lb,X_ub,X_f,u0_true,v0_true)
    gradients = tape.gradient(Loss, model.trainable_variables)
    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return Loss

def test(model,X_star):
    
    f_u_star,f_v_star,u_star,v_star = model.network_f(X_star)

    return u_star,v_star,f_u_star,f_v_star


def main():
    
    data = scipy.io.loadmat('NLS.mat') # the data set contains the exact solution
   
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
    u0 = Exact_u[idx_x,0]
    v0 = Exact_v[idx_x,0]
    
    #print("x0",x0.shape)
    # select the boundary value data
    idx_t = np.random.choice(t.shape[0], Nb, replace=False)
    tb = t[idx_t,:]
    # select the data inside domain pairs, it contains Nf pairs of (x,t)
    X_f = lb + (ub-lb)*lhs(2, Nf)

    
    #Concatenate all data into (x,t) pairs
    X0 = np.concatenate((x0, 0*x0), 1) # (x0, 0)
    X_lb = np.concatenate((0*tb + lb[0], tb), 1) # (lb[0], tb)
    X_ub = np.concatenate((0*tb + ub[0], tb), 1) # (ub[0], tb)

    model = PDENET()
	#create the model
    for i in range (0,model.epochN):
        rLoss = train(model,X0,X_lb,X_ub,X_f,u0,v0)
        if i % 100 == 0:
            print("The {}th epoch is finished, the loss is:{}".format(i+1,rLoss.numpy()))
            u_value,v_value,f_u_value,f_v_value = test(model,X_star)
            h_value = np.sqrt(u_value**2 + v_value**2)

            u_value = u_value.numpy()
            v_value = v_value.numpy()
            f_u_value = f_u_value.numpy()
            f_v_value = f_v_value.numpy()


            U_pred = griddata(X_star, u_value.flatten(), (X, T), method='cubic')
            V_pred = griddata(X_star, v_value.flatten(), (X, T), method='cubic')
            H_pred = griddata(X_star, h_value.flatten(), (X, T), method='cubic')

            FU_pred = griddata(X_star, f_u_value.flatten(), (X, T), method='cubic')
            FV_pred = griddata(X_star, f_v_value.flatten(), (X, T), method='cubic') 

    
            plt.plot(x,Exact_h[:,100],'b-', linewidth = 2, label = 'Exact')
            plt.plot(x,H_pred[100,:], 'r--', linewidth = 2, label = 'Prediction')
            plt.ylabel('|h(x,t)|')
            plt.xlabel('x')
            plt.title('Schrodinger Equation at t=100')
            plt.show(block=False)
            plt.savefig('Step {}.png'.format(i+1))
            plt.pause(1)
            plt.close()

if __name__ == '__main__':
	main()