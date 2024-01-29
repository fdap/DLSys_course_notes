# Convolutional Networks
***

## Convolutional operators in deep networks

> Idea is to “slide” the weights $k \times k$ weight $w$ (called a filter, with kernel size $k$) over
the image to produce a new image, written $y = z * w$.

> Convolutions (typically with prespecified filters) are a common operation in many computer vision applications: convolution networks just move to learned filters

> Convolutions in deep networks are virtually always multi-channel convolutions: map multi-channel (e.g., RGB) inputs to multi-channel hidden units
1. Multi-channel convolutions contain a convolutional filter for each input-output channel pair, single output channel is sum of convolutions over all input channels


## Elements of practical convolutions



## Differentiating convolutions