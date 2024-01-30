# Convolutional Networks
***

## Convolutional operators in deep networks

> Idea is to “slide” the weights $k \times k$ weight $w$ (called a filter, with kernel size $k$) over
the image to produce a new image, written $y = z * w$.

> Convolutions (typically with prespecified filters) are a common operation in many computer vision applications: convolution networks just move to learned filters

> Convolutions in deep networks are virtually always multi-channel convolutions: map multi-channel (e.g., RGB) inputs to multi-channel hidden units
1. Multi-channel convolutions contain a convolutional filter for each input-output channel pair, **single output channel is sum of convolutions over all input channels**


## Elements of practical convolutions
1. Padding: variants like circular padding, padding with mean values, etc
2. Strided Convolutions / Pooling:
3. Grouped Convolutions: Group together channels, so that groups of channels in output only depend on corresponding groups of channels in input (equivalently, enforce filter weight matrices to be block-diagonal)
4. Dilate (spread out) convolution filter, so that it covers more of the image;


## Differentiating convolutions