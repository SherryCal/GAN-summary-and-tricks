Autoencoders (AE)[6] are neural networks that aims to copy their inputs to their outputs. They work by compressing the input into a latent-space representation, and then reconstructing the output from this representation. This kind of network is composed of two parts:  encoder and decoder.
Encoder is the part of the network that compresses the input into a latent-space representation.
Decoder aims to reconstruct the input from the latent space representation.
Using backpropagation, this unsupervised algorithm continuously trains itself by setting the target output values to equal the inputs. This forces the smaller hidden encoding layer to use dimensional reduction to eliminate noise and reconstruct the inputs.
Autoencoders are learned automatically from data examples. It means that it is easy to train specialized instances of the algorithm that will perform well on a specific type of input and that it does not require any new engineering, only the appropriate training data.
