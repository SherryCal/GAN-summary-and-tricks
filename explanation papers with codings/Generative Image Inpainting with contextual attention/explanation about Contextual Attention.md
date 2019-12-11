# Image Inpainting with Contextual Attention
Convolutional neural networks process image features with local convolutional kernel layer by layer thus are not effective for borrowing features from distant spatial locations. To overcome the limitation, we consider attention
mechanism and introduce a novel contextual attention layer in the deep generative network. In this section, we first discuss details of the contextual attention layer, and then address how we integrate it into our unified inpainting network.
## Image Inpainting with Contextual Attention
The contextual attention layer learns where to borrow or copy feature information from known background patches to generate missing patches. It is differentiable, thus can be trained in deep models, and fully-convolutional, which allows testing on arbitrary resolutions.
![image](https://github.com/SherryCal/related-work-summary-and-tricks/blob/master/explanation%20papers%20with%20codings/Generative%20Image%20Inpainting%20with%20contextual%20attention/contextual%20attention%20structure.png)
