Recent studies on arbitrary region competion add a new discriminator Network that considers only the filled region to emphasize 
the adversarial loss on the top of the global GAN discriminator(G-GAN), the addition Network, which called the local discriminator(L-GAN), 
facilitates exposing the local structure details, although those works have shown prominent results for the large hole filling problem,
 their main drawback is the LGAN's emphasis on conditioning to the location of the location of the mask. It is observed thatthis leads the 
  disharmony between the masked area where the LGAN is interested in and the uncorrupted texture in the masked area. The same problem is
   indicated to the synthesized image. LGAN pushes the generative network to produce independent textures that are incompatible with 
   the whole image semantics
   
   this problem can be solved by adding an extension discriminator or extension network that corrects the imperfections
