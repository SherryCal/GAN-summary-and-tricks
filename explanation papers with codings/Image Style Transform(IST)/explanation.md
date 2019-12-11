# Image Style Transform(IST)
When Convolutional Neural Networks are trained on object recognition, they develop a
representation of the image that makes object information increasingly explicit along the processing
hierarchy.Therefore, along the processing hierarchy of the network, the input image
is transformed into representations that increasingly care about the actual content of the image
compared to its detailed pixel values.
Higher layers in the network capture the high-level content in terms of objects and their
arrangement in the input image but do not constrain the exact pixel values of the reconstruction. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image (Fig 1, content reconstructions
a,b,c). We therefore refer to the feature responses in higher layers of the network as the content representation.\\
To obtain a representation of the style of an input image, we use a feature space originally
designed to capture texture information.8 This feature space is built on top of the filter responses
in each layer of the network. It consists of the correlations between the different filter responses
over the spatial extent of the feature maps. By including the feature
correlations of multiple layers, we obtain a stationary, multi-scale representation of the input
image, which captures its texture information but not the global arrangement

## algorithm Overview
The principle of IST is based on the above mentioned characteristics that different layers of the network will respond to different types of features. Given a trained network, the source code used is VGG19
```
88 model = vgg19.VGG19(input_tensor=input_tensor,
89                   weights='imagenet', include_top=False)
```
