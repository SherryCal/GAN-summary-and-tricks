def GroupNorm(x, gamma, beta, G, eps=1e−5):
     # x: input features with shape [N,C,H,W]
     # gamma, beta: scale and offset, with shape [1,C,1,1]
     # G: number of groups for GN
     N, C, H, W = x.shape
     x = tf.reshape(x, [N, G, C // G, H, W])
     mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True) 
     x = (x − mean) / tf.sqrt(var + eps)
     x = tf.reshape(x, [N, C, H, W]) 
     return x * gamma + beta
