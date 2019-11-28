Patch GAN的差别主要是在于Discriminator上，一般的GAN是只需要输出一个true or fasle 的矢量，这是代表对整张图像的评价；但是PatchGAN输出的是一个N x N的矩阵，这个N x N的矩阵的每一个元素，比如a(i,j) 只有True or False 这两个选择（label 是 N x N的矩阵，每一个元素是True 或者 False），这样的结果往往是通过卷积层来达到的，因为逐次叠加的卷积层最终输出的这个N x N 的矩阵，其中的每一个元素，实际上代表着原图中的一个比较大的感受野，也就是说对应着原图中的一个Patch，因此具有这样结构以及这样输出的GAN被称之为Patch GAN


(By author of P2P GAN:

Patch GAN is just a ConvNet, or you can say Patch GAN is a PatchNet, the power of ConvNet is that rather the regular GAN maps
 from the image to a single scalar output, which signifies real or fake, whereas, the Patch GAN maps from Image to an N*N array of 
 output X, where each X_ij signifies the patch ij in the image is real or fake. It should be beteer if we call it as Fully Convlutional GAN)
 
)

# Coding(Keras Model)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model([img_A, img_B], validity)
