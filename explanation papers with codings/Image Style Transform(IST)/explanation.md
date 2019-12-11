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
88   model = vgg19.VGG19(input_tensor=input_tensor,
89                     weights='imagenet', include_top=False)
```
Style transfer algorithm. First content and style features are extracted and stored. The style image $\stackrel{\rightarrow}{a}$ is passed through the network
and its style representation Al on all layers included are computed and stored (left). The content image ~p is passed through the network
and the content representation Pl in one layer is stored (right). Then a random white noise image ~x is passed through the network and its
style features Gl and content features Fl are computed. On each layer included in the style representation, the element-wise mean squared
difference between Gl and Al is computed to give the style loss Lstyle (left). Also the mean squared difference between Fl and Pl is
computed to give the content loss Lcontent (right). The total loss Ltotal is then a linear combination between the content and the style loss.
Its derivative with respect to the pixel values can be computed using error back-propagation (middle). This gradient is used to iteratively
update the image ~x until it simultaneously matches the style features of the style image ~a and the content features of the content image ~p
(middle, bottom).
![Alt text](https://github.com/SherryCal/related-work-summary-and-tricks/blob/master/explanation%20papers%20with%20codings/Image%20Style%20Transform(IST)/%20flowchart.png)
