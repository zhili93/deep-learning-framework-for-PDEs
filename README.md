# deep-learning-framework-for-PDEs
In this project, we have implemented a physics-informed neural network framework capable of solving the forward and inverse problem for non-linear partial differential equations [1].


Directory

--Utilities (contains useful plotting function)
--Main
  --forward_problem
      --MySch_Final.py (TensorFlow 2.0 implementation for forward problem)
      --Data (training data for the forward problem)
      --Output (outputs from the neural network model)
  --inverse_problem
      --NavierStokes_tf2.py (TensorFlow 2.0 implementation for inverse problem)
      --Data (training data for the inverse problem)
      --Output (outputs from the neural network model)



Reference:
[1] Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations." Journal of Computational Physics 378 (2019): 686-707.
