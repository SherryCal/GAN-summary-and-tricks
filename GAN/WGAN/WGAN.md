# WGAN 归纳

在 WGAN 中，D 的任务不再是尽力区分生成样本与真实样本，而是尽量拟合出样本间的 Wasserstein 距离，从分类任务转化成回归任务。而 G 的任务则变成了尽力缩短样本间的 Wasserstein 距离。

故 WGAN 对原始 GAN 做出了如下改变:

D 的最后一层取消 sigmoid
D 的 w 取值限制在 [-c,c] 区间内。
使用 RMSProp 或 SGD 并以较低的学习率进行优化 (论文作者在实验中得出的 trick)
# WGAN 的个人一些使用经验总结,仅供参考：

WGAN 的论文指出使用 MLP，3 层 relu，最后一层使用 linear 也能达到可以接受的效果，但根据我实验的经验上，可能对于彩色图片，因为其数值分布式连续，所以使用 linear 会比较好。但针对于 MINST 上，因为其实二值图片，linear 的效果很差，可以使用 batch normalization sigmoid 效果更好。
不要在 D 中使用 batch normalization，估计原因是因为 weight clip 对 batch normalization 的影响
使用逆卷积来生成图片会比用全连接层效果好，全连接层会有较多的噪点，逆卷积层效果清晰。
关于衡量指标，Wasserstein distance 距离可以很好的衡量 WGAN 的训练进程，但这仅限于同一次，即你的代码从运行到结束这个过程内。
