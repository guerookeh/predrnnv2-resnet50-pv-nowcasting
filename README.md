# PredRNNv2 + Re Solar Photovoltaic Yield Nowcasting

A very experimental (and unfortunately unsubmitted ClimateHack2023) attempt at using the PredRNNv2 model from the paper ["PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning"](https://arxiv.org/abs/2103.09504) to generate the next four hours at five-minute intervals of High-Resolution Visible (HRV) satellite imagery over a solar site based on the current hour, followed up by an imagery to solar photovoltaic yield conversion using a pretrained and lightly finetuned ResNet50. 

The PredRNNv2 model code is directly taken from The PyTorch implementation of PredRNNv2 found [here](https://github.com/thuml/predrnn-pytorch).



# Input/Output

A tensor of satellite imagery with dimensions $12 \times 128 \times 128$, representing snapshots taken at five-minute intervals over an hour at a solar site, is initially input into the PredRNNv2 model. This model generates a $48 \times 128 \times 128$ tensor, predicting satellite imagery for the subsequent four hours. Subsequently, each image in this generated tensor is individually processed through the ResNet50 model, resulting in a $48 \times 1$ vector. This vector represents the predicted solar power yields for each of the upcoming four hours, with predictions made at five-minute intervals.

# Reasoning

The reason why I chose PredRNNv2 for this task is due to 1. the architecture of a recurrent neural network makes it well-suited for processing sequential data, and 2. the integration of spatial information into a temporal sequence model through the PredRNN paper's problem formulation, Spatiotemporal Predictive Learning, quoted down below.

**Spatiotemporal Predictive Learning**, directly from ["PredRNN: A Recurrent Neural Network for Spatiotemporal Predictive Learning](https://arxiv.org/abs/2103.09504"):
Suppose we are monitoring a dynamical system of $J$ measurements over time, where each measurement is recorded at all locations in a spatial region represented by a $M\times N$ grid. From the spatial view, the observation of these $J$ measurements at any time can be represented by a tensor of $\mathbf{X} \in \mathbb{R}^{J \times M \times N}$. From the temporal view, the observations over $T$ time-steps form a sequence of $\mathbf{X}\_{\text{in}}=\{\mathbf{X}_1,...,\mathbf{X}_T\}$. Given $\mathbf{X}_{\text{in}}$, the spatiotemporal predictive learning is to predict the most probable length-$K$ sequence in the future, $\mathbf{\hat{X}}_{\text{out}}=\{{\hat{X}}_{T+1},...,\hat{X}_{T+K}\}$. In this paper, we train neural networks parameterized by $\theta$. Concretely, we use stochastic gradient descent to find a set of parameters $\theta^*$ that maximizes the log-likelihood of producing the true target sequence $\mathbf{X}_{\text{out}}$ given the  $\mathbf{X}_{\textbf{in}}$ for all training pairs $\{(\mathbf{X} _{\text{in}}^n, \mathbf{X} _{\text{out}}^n )\}_n$. 
$$
\theta^*=\text{argmax}_\theta \sum_{(\mathbf{X} _{\text{in}}^n, \mathbf{X} _{\text{out}}^n )}\log\Pr(\mathbf{X} _{\text{out}}^n|\mathbf{X} _{\text{in}}^n;\theta)
$$

_More to add and edit later..._
