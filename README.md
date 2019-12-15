# deep-learning-framework-for-PDEs
In this project, we have implemented a physics-informed neural network framework capable of solving the forward and inverse problem for non-linear partial differential equations [1].


Directory

-Utilities (contains useful plotting function) <br />
-Main<br />
    -forward_problem <br />
        -MySch_Final.py (TensorFlow 2.0 implementation for forward problem)<br />
        -Data (training data for the forward problem)<br />
        -Output (outputs from the neural network model)<br />
    -inverse_problem<br />
        -NavierStokes_tf2.py (TensorFlow 2.0 implementation for inverse problem)<br />
        -Data (training data for the inverse problem)<br />
        -Output (outputs from the neural network model)<br />

Reference:
[1] Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
