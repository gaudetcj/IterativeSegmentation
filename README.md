# IterativeSegmentation
Keras  implementation of 'Learning Iterative Processes with Recurrent Neural Networks to Correct Satellite Image Classification Maps'

This paper uses a 2 stage process of training where th results from a CNN are used to train a RNN for refinement. The RNN is analogous to a diffusion process in the form of a PDE. The RNN takes advantage of a MLP being a universal function approximator to solve the system in a x_{t + 1} = x_{t} + \delta x_{t}.

This implementation uses a U-Net like model to generate heatmaps and then a RNN is built by stacking layers that share weights.
