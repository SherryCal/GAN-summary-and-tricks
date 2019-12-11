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
In this way, we can extract the Feature Map we want based on the keyword. Feature Map $P^l$ （base_image_features）and $F^l$(combination_features），then use this two Feature Maps to compute loss fuction
```
154   layer_features = outputs_dict['block5_conv2']
155   base_image_features = layer_features[0, :, :, :]
156   combination_features = layer_features[2, :, :, :]
157   loss += content_weight * content_loss(base_image_features,
                                      combination_features)

```
content_weigh is the power of content loss , and the coding use 0.025. As for content loss
```
131   def content_loss(base, combination):
132       return K.sum(K.square(combination - base))
```
With the definition of the loss function, we can calculate the gradient value of the loss function with respect to $F_{i,j}$ according to the value of the loss function, thus realizing the gradient update from back to front.
## Style Presentation
The two batches of the picture, the left and the middle.

Different from the direct operation of content representation, style representation uses the form of Gram matrix expanded into 1-dimensional vectors by Feature Map. The reason for using Gram matrix is that considering that the texture feature has nothing to do with the specific position of the image, this feature can be guaranteed by scrambling the position information of the texture. The definition of Gram matrix is as follows.
$$G_{i,j}^l=\sum_{k}F_{i,k}^lF_{j,k}^l$$
```
101   def gram_matrix(x):
102       assert K.ndim(x) == 3
103       if K.image_data_format() == 'channels_first':
104           features = K.batch_flatten(x)
105       else:
106           features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
107       gram = K.dot(features, K.transpose(features))
108       return gram
```
Another way, the content is different, the style is said to use the first convolution to calculate the loss function of each block, the author thinks this way get the texture Feature is more smooth, because just using the underlying Feature of the Map to get the image more detailed but more rough, and high-level image contain more information, the content of the lost some texture information. We use layers
```
160   feature_layers = ['block1_conv1', 'block2_conv1',
162                     'block3_conv1', 'block4_conv1',
163                     'block5_conv1']
```
The loss function combining the styles of all layers is
$$L_{style}=\sum_{l}W_{l}E_{l}$$
$E_{l}$ is the Mean square error of gram of $S^l$ and the gram of $F^l$
```
163   for layer_name in feature_layers:
164       layer_features = outputs_dict[layer_name]
165       style_reference_features = layer_features[1, :, :, :]
166       combination_features = layer_features[2, :, :, :]
167       sl = style_loss(style_reference_features, combination_features)
168       loss += (style_weight / len(feature_layers)) * sl
169   loss += total_variation_weight * total_variation_loss(combination_image)
```
the style_loss is 
```
117    def style_loss(style, combination):
118        assert K.ndim(style) == 3
119        assert K.ndim(combination) == 3
120        S = gram_matrix(style)
121        C = gram_matrix(combination)
122        channels = 3
123        size = img_nrows * img_ncols
124        return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))

```
use SGD for optimation
## Style migration
Loss fuction 
$$L_{total}((\vec{p},\vec{a},\vec{x}))=\alpha L_{content}(\vec{p},\vec{x})+\beta L_{style}(\vec{a},\vec{x})$$
