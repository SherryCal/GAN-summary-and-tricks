In this paper, we present a novel CNN, namely Shift-Net, to take into account
the advantages of both exemplar-based and CNN-based methods for image inpainting. 



. In exemplar-based inpainting [4], the patch-based replication
and filling process are iteratively performed to grow the texture and structure
from the known region to the missing parts. And the patch processing order
plays a key role in yielding plausible inpainting result


. Guided by the salient structure produced by CNN, the filling process
in our Shift-Net can be finished concurrently by introducing a shift-connection
layer to connect the encoder feature of known region and the decoder feature
of missing parts. Thus, our Shift-Net inherits the advantages of exemplar-based
and CNN-based methods, and can produce inpainting result with both plausible
semantics and fine detailed textures
