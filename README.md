# SampleGan


### Discriminator network
![alt text](https://github.com/alexchiciu/SampleGan/blob/main/images/Discrim.png)

The discriminator has two conv layers, each followed by a max pool and a LeakyRelu. Then three linear layers follow. After each linear layer there is a LeakyRelu, final output is a single scaler.


### Generator network
![alt text](https://github.com/alexchiciu/SampleGan/blob/main/images/Generator.png)

The generator has conv layer, followed by 6 residual blocks and then an upsample layer that upsamples the 3d image by a factor of two. 
i.e. (10 x 10 x 10) -> (20 x 20 x 20)
