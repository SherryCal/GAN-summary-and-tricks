# Image Style Transform(IST)
When Convolutional Neural Networks are trained on object recognition, they develop a representation of the image that makes object information increasingly explicit along the processing hierarchy.Therefore, along the processing hierarchy of the network, the input image is transformed into representations that increasingly care about the actual content of the image
compared to its detailed pixel values. Higher layers in the network capture the high-level content in terms of objects and their arrangement in the input image but do not constrain the exact pixel values of the reconstruction. In contrast, reconstructions from the lower layers simply reproduce the exact pixel values of the original image (Fig 1, content reconstructions a,b,c). We therefore refer to the feature responses in higher layers of the network as the content representation.


To obtain a representation of the style of an input image, we use a feature space originally designed to capture texture information.8 This feature space is built on top of the filter responses in each layer of the network. It consists of the correlations between the different filter responses over the spatial extent of the feature maps. By including the feature
correlations of multiple layers, we obtain a stationary, multi-scale representation of the input image, which captures its texture information but not the global arrangement

## algorithm Overview
The principle of IST is based on the above mentioned characteristics that different layers of the network will respond to different types of features. Given a trained network, the source code used is VGG19
```
88   model = vgg19.VGG19(input_tensor=input_tensor,
89                     weights='imagenet', include_top=False)
```
Style transfer algorithm. First content and style features are extracted and stored. The style image $\vec{a}$ is passed through the network and its style representation Al on all layers included are computed and stored (left). The content image $\vec{p}$ is passed through the network and the content representation Pl in one layer is stored (right). Then a random white noise image $\vec{x}$ is passed through the network and its style features Gl and content features Fl are computed. On each layer included in the style representation, the element-wise mean squared difference between Gl and Al is computed to give the style loss Lstyle (left). Also the mean squared difference between Fl and Pl is computed to give the content loss Lcontent (right). The total loss Ltotal is then a linear combination between the content and the style loss. Its derivative with respect to the pixel values can be computed using error back-propagation (middle). This gradient is used to iteratively
update the image $\vec{x}$ until it simultaneously matches the style features of the style image $\vec{a}$ and the content features of the content image $\vec{p}$ (middle, bottom).
![Alt text](https://github.com/SherryCal/related-work-summary-and-tricks/blob/master/explanation%20papers%20with%20codings/Image%20Style%20Transform(IST)/%20flowchart.png)
## Content representation
explained in the right of the picture

A layer with $N_l$ distinct filters has $N_l$ feature maps each of size $M_l$, where $M_l$ is the height times the width of the feature map. So the responses in a layer l can be stored in a matrix $F^l \in R^{N_l×M_l}$ where $F^l_{i,j}$ is the activation of the $i^{th}$ filter at position $j$ in layer $l$.Let $\vec{p}$ and $\vec{x}$ be the original image and the image that is generated, and $P_l$ and $F_l$ their respective feature representation in layer $l$. We then define the squared-error loss between the two feature representations
$$L_{content(\vec{p},\vec{x},l)}=\frac{1}{2}\sum_{ij}{(F_{i,j}^l-P_{i,j}^l})^2$$
Input_tensor
```
82   input_tensor = K.concatenate([base_image,
83                              style_reference_image,
84                              combination_image], axis=0)
```
Get the model's dictionary
```
93  outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

```
In this way, we can extract the Feature Map we want based on the keyword. Feature Map [P^l]
```
layer_features = outputs_dict['block5_conv2']
base_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]
loss += content_weight * content_loss(base_image_features,
                                      combination_features)

```
