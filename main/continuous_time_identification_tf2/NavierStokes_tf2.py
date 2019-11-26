import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
import time
from itertools import product, combinations
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

np.random.seed(1234)
tf.random.set_seed(1234)

class PhysicsInformedNN(tf.keras.Model):
    # Initialize the class
    def __init__(self):
    
        super(PhysicsInformedNN, self).__init__()
        self.optimizer=tf.keras.optimizers.Nadam(learning_rate=0.001)
        # Initialize layers  layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
        
        self.dense1=tf.keras.layers.Dense(3,activation='tanh')
        self.dense2=tf.keras.layers.Dense(20,activation='tanh')
        self.dense3=tf.keras.layers.Dense(20,activation='tanh')
        self.dense4=tf.keras.layers.Dense(20,activation='tanh')
        self.dense5=tf.keras.layers.Dense(20,activation='tanh')
        self.dense6=tf.keras.layers.Dense(20,activation='tanh')
        self.dense7=tf.keras.layers.Dense(20,activation='tanh')
        self.dense8=tf.keras.layers.Dense(2,activation='tanh')
   
        
        # Initialize parameters
        self.lambda_1 = tf.Variable([0.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([0.0], dtype=tf.float32)
                                 
    def call(self, X):
    
        x = tf.Variable(X[:,0:1])
        y = tf.Variable(X[:,1:2])
        t = tf.Variable(X[:,2:3])   
        X_call=tf.concat([x, y, t], 1)
        lb = tf.reduce_min(x)
        ub = tf.reduce_max(x)
        
        with tf.GradientTape(persistent=True) as t:
            t.watch(X_call)
            with tf.GradientTape(persistent=True) as tt:
                tt.watch(X_call)
                with tf.GradientTape(persistent=True) as ttt:
                    ttt.watch(X_call)
                    H = 2.0*(X_call - lb)/(ub - lb) - 1.0  # centralize the data 
                    psi_and_p=self.dense8(self.dense7(self.dense6(self.dense5(self.dense4(self.dense3(self.dense2(self.dense1(H))))))))
                    psi = psi_and_p[:,0:1]
                    p = psi_and_p[:,1:2] 
        
                u=ttt.gradient(psi, X_call)[:,1]
                v=-ttt.gradient(psi, X_call)[:,0]
                p_x = ttt.gradient(p, X_call)[:,0]
                p_y = ttt.gradient(p, X_call)[:,1]                

            u_t = tt.gradient(u, X_call)[:,2]
            u_x = tt.gradient(u, X_call)[:,0]
            u_y = tt.gradient(u, X_call)[:,1]
  
            v_t = tt.gradient(v, X_call)[:,2]
            v_x = tt.gradient(v, X_call)[:,0]
            v_y = tt.gradient(v, X_call)[:,1]
            
        u_xx = t.gradient(u_x, X_call)[:,0]
        u_yy = t.gradient(u_y, X_call)[:,1]
        v_xx = t.gradient(v_x, X_call)[:,0]
        v_yy = t.gradient(v_y, X_call)[:,1]



        f_u = u_t + self.lambda_1*(u*u_x + v*u_y) + p_x - self.lambda_2*(u_xx + u_yy) 
        f_v = v_t + self.lambda_1*(u*v_x + v*v_y) + p_y - self.lambda_2*(v_xx + v_yy)

        return u, v, p, f_u, f_v
    
    def loss_function(self,u_p, v_p, p_p, f_u_p, f_v_p, u,v):
        
        loss = tf.reduce_sum(tf.square(u - u_p)) + \
                    tf.reduce_sum(tf.square(v - v_p)) + \
                    tf.reduce_sum(tf.square(f_u_p)) + \
                    tf.reduce_sum(tf.square(f_v_p))
        return loss

      
def train(model, nIter, X, u, v): 
    optimizer = model.optimizer
    start_time = time.time()
    for it in range(nIter):
        with tf.GradientTape() as tape:
            u_p, v_p, p_p, f_u_p, f_v_p=model(X)
            loss=model.loss_function(u_p, v_p, p_p, f_u_p, f_v_p,u,v)
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, l1: %.3f, l2: %.5f, Time: %.2f' % (it, loss.numpy(),model.lambda_1.numpy() ,model.lambda_2.numpy(), elapsed))
                start_time = time.time()            
                
        gradients = tape.gradient(loss, model.trainable_variables)
        #print(model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))     
            

    pass
            
def test(model, X, u, v, p):

    # Prediction
    u_pred, v_pred, p_pred,f_u_pred, f_v_pred = model(X)
    
    lambda_1_value = model.lambda_1
    lambda_2_value = model.lambda_2
    
    # Error
    error_u = np.linalg.norm(u-u_pred,2)/np.linalg.norm(u,2)
    error_v = np.linalg.norm(v-v_pred,2)/np.linalg.norm(v,2)
    error_p = np.linalg.norm(p-p_pred,2)/np.linalg.norm(p,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))     

    pass
    

def plot_solution(X_star, u_star, index):
    
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    U_star = griddata(X_star, u_star.flatten(), (X, Y), method='cubic')
    
    plt.figure(index)
    plt.pcolor(X,Y,U_star, cmap = 'jet')
    plt.colorbar()
    
    
def axisEqual3D(ax):
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/4
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
        
if __name__ == "__main__": 
      
    N_train = 5000
    
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    data = scipy.io.loadmat('../Data/cylinder_nektar_wake.mat')
           
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2
    
    N = X_star.shape[0]
    T = t_star.shape[0]
    
    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]
    
    

    # Training
    X_train = np.concatenate([x_train, y_train, t_train], 1)
    print('finish preparing training data')
    model = PhysicsInformedNN()
    train(model, 1000, X_train, u_train, v_train)
    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    X_test = np.concatenate([x_star, y_star, t_star], 1)   
    test(model, X_test, u_star, v_star, p_star)             
    
    # Plot Results
#    plot_solution(X_star, u_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    u_pred, v_pred, p_pred,f_u_pred, f_v_pred = model(X_test)    
    UU_star = griddata(X_star, np.ndarray.flatten(u_pred.numpy()), (X, Y), method='cubic')
    VV_star = griddata(X_star, np.ndarray.flatten(v_pred.numpy()), (X, Y), method='cubic')
    PP_star = griddata(X_star, np.ndarray.flatten(p_pred.numpy()), (X, Y), method='cubic')
    P_exact = griddata(X_star, np.ndarray.flatten(p_star), (X, Y), method='cubic')
    
    
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01        
    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
    v_train = v_train + noise*np.std(v_train)*np.random.randn(v_train.shape[0], v_train.shape[1])    

    # Training
    model2 = PhysicsInformedNN()
    #train(model2, 1000, X_train, u_train, v_train)
        
    lambda_1_value_noisy = model2.lambda_1
    lambda_2_value_noisy = model2.lambda_2
      
    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - 0.01)/0.01 * 100
        
    print('Error l1: %.5f%%' % (error_lambda_1_noisy))                             
    print('Error l2: %.5f%%' % (error_lambda_2_noisy))     

             
    
    ######################################################################
    ############################# Plotting ###############################
    ######################################################################    
     # Load Data
    data_vort = scipy.io.loadmat('../Data/cylinder_nektar_t0_vorticity.mat')
           
    x_vort = data_vort['x'] 
    y_vort = data_vort['y'] 
    w_vort = data_vort['w'] 
    modes = np.asscalar(data_vort['modes'])
    nel = np.asscalar(data_vort['nel'])    
    
    xx_vort = np.reshape(x_vort, (modes+1,modes+1,nel), order = 'F')
    yy_vort = np.reshape(y_vort, (modes+1,modes+1,nel), order = 'F')
    ww_vort = np.reshape(w_vort, (modes+1,modes+1,nel), order = 'F')
    
    box_lb = np.array([1.0, -2.0])
    box_ub = np.array([8.0, 2.0])
    
    fig, ax = newfig(1.0, 1.2)
    ax.axis('off')
    
    ####### Row 0: Vorticity ##################    
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1-0.06, bottom=1-2/4 + 0.12, left=0.0, right=1.0, wspace=0)
    ax = plt.subplot(gs0[:, :])
    
    for i in range(0, nel):
        h = ax.pcolormesh(xx_vort[:,:,i], yy_vort[:,:,i], ww_vort[:,:,i], cmap='seismic',shading='gouraud',  vmin=-3, vmax=3) 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)
    
    ax.plot([box_lb[0],box_lb[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_ub[0],box_ub[0]],[box_lb[1],box_ub[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_lb[1],box_lb[1]],'k',linewidth = 1)
    ax.plot([box_lb[0],box_ub[0]],[box_ub[1],box_ub[1]],'k',linewidth = 1)
    
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title('Vorticity', fontsize = 10)
    
    
    ####### Row 1: Training data ##################
    ########      u(t,x,y)     ###################        
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(top=1-2/4, bottom=0.0, left=0.01, right=0.99, wspace=0)
    ax = plt.subplot(gs1[:, 0],  projection='3d')
    ax.axis('off')

    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,UU_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$u(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    ########      v(t,x,y)     ###################        
    ax = plt.subplot(gs1[:, 1],  projection='3d')
    ax.axis('off')
    
    r1 = [x_star.min(), x_star.max()]
    r2 = [data['t'].min(), data['t'].max()]       
    r3 = [y_star.min(), y_star.max()]
    
    for s, e in combinations(np.array(list(product(r1,r2,r3))), 2):
        if np.sum(np.abs(s-e)) == r1[1]-r1[0] or np.sum(np.abs(s-e)) == r2[1]-r2[0] or np.sum(np.abs(s-e)) == r3[1]-r3[0]:
            ax.plot3D(*zip(s,e), color="k", linewidth = 0.5)   

    ax.scatter(x_train, t_train, y_train, s = 0.1)
    ax.contourf(X,VV_star,Y, zdir = 'y', offset = t_star.mean(), cmap='rainbow', alpha = 0.8)
              
    ax.text(x_star.mean(), data['t'].min() - 1, y_star.min() - 1, '$x$')
    ax.text(x_star.max()+1, data['t'].mean(), y_star.min() - 1, '$t$')
    ax.text(x_star.min()-1, data['t'].min() - 0.5, y_star.mean(), '$y$')
    ax.text(x_star.min()-3, data['t'].mean(), y_star.max() + 1, '$v(t,x,y)$')    
    ax.set_xlim3d(r1)
    ax.set_ylim3d(r2)
    ax.set_zlim3d(r3)
    axisEqual3D(ax)
    
    savefig('./figures/NavierStokes_data') 

    
    fig, ax = newfig(1.015, 0.8)
    ax.axis('off')
    
    ######## Row 2: Pressure #######################
    ########      Predicted p(t,x,y)     ########### 
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1-1/2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(PP_star, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Predicted pressure', fontsize = 10)
    
    ########     Exact p(t,x,y)     ########### 
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(P_exact, interpolation='nearest', cmap='rainbow', 
                  extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()], 
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal', 'box')
    ax.set_title('Exact pressure', fontsize = 10)
    
    
    ######## Row 3: Table #######################
    # gs3 = gridspec.GridSpec(1, 2)
    # gs3.update(top=1-1/2, bottom=0.0, left=0.0, right=1.0, wspace=0)
    # ax = plt.subplot(gs3[:, :])
    # ax.axis('off')
    
    # s = r'$\begin{tabular}{|c|c|}';
    # s = s + r' \hline'
    # s = s + r' Correct PDE & $\begin{array}{c}'
    # s = s + r' u_t + (u u_x + v u_y) = -p_x + 0.01 (u_{xx} + u_{yy})\\'
    # s = s + r' v_t + (u v_x + v v_y) = -p_y + 0.01 (v_{xx} + v_{yy})'
    # s = s + r' \end{array}$ \\ '
    # s = s + r' \hline'
    # s = s + r' Identified PDE (clean data) & $\begin{array}{c}'
    # s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (model.lambda_1.numpy(), model.lambda_2.numpy())
    # s = s + r' \\'
    # s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (model.lambda_1.numpy(), model.lambda_2.numpy())
    # s = s + r' \end{array}$ \\ '
    # s = s + r' \hline'
    # s = s + r' Identified PDE (1\% noise) & $\begin{array}{c}'
    # s = s + r' u_t + %.3f (u u_x + v u_y) = -p_x + %.5f (u_{xx} + u_{yy})' % (lambda_1_value_noisy.numpy(), lambda_2_value_noisy.numpy())
    # s = s + r' \\'
    # s = s + r' v_t + %.3f (u v_x + v v_y) = -p_y + %.5f (v_{xx} + v_{yy})' % (lambda_1_value_noisy.numpy(), lambda_2_value_noisy.numpy())
    # s = s + r' \end{array}$ \\ '
    # s = s + r' \hline'
    # s = s + r' \end{tabular}$'
 
    # ax.text(0.015,0.0,s)
    
    savefig('./figures/NavierStokes_prediction') 

