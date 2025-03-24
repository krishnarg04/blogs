# My Journey Implementing Variational Autoencoders

Over the past few months, I've been diving deep into generative models, particularly Variational Autoencoders (VAEs). What started as a curiosity quickly became an obsession as I explored this fascinating intersection of deep learning and probability theory. In this post, I'll share what I've learned, how I implemented my own VAE, and why I believe these models are so powerful.

## What is a Variational Autoencoder?

At its core, a VAE is a generative model that learns to reconstruct its input data while simultaneously learning the underlying probability distribution of that data. Unlike traditional autoencoders that directly map inputs to a compressed representation, VAEs map inputs to a probability distribution in the latent space.

Think of it this way: a regular autoencoder learns to compress data into a fixed point in latent space, while a VAE learns to compress data into a small region of the latent space. This subtle difference is what gives VAEs their generative power.

The architecture consists of two main components:
1. **Encoder**: Transforms input data into parameters of a probability distribution
2. **Decoder**: Samples from this distribution and attempts to reconstruct the original input

## Why Do We Need VAEs?

Before diving into VAEs, I had worked with GANs (Generative Adversarial Networks), which are powerful but notoriously difficult to train. I found myself constantly battling mode collapse and instability issues. VAEs offered a more stable alternative with some unique advantages:

1. **Probabilistic Framework**: VAEs provide a principled way to generate new data by sampling from the learned distribution
2. **Meaningful Latent Space**: The latent space has a smooth structure where similar inputs are close together
3. **Regularization**: The KL divergence term prevents overfitting and encourages a well-structured latent space
4. **Stability**: Training is generally more stable than GANs

The ability to both generate new samples and perform meaningful interpolations between existing samples made VAEs particularly appealing for my projects.

## The Mathematics Behind VAEs

Here's where things get interesting (and yes, a bit technical). The mathematical foundation of VAEs lies in variational inference - a technique from Bayesian statistics.

### The ELBO Loss Function

The VAE optimization objective is the Evidence Lower Bound (ELBO), which consists of two terms:

$$\mathcal{L}(\theta, \phi; x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{KL}(q_\phi(z|x) || p(z))$$

Where:
- The first term is the reconstruction loss
- The second term is the KL divergence between the encoder's distribution and a prior (usually a standard normal distribution)

This formula reveals the fundamental trade-off in VAEs: reconstruction accuracy versus regularization of the latent space.

### The Reparameterization Trick

One of the most elegant aspects of VAEs is the reparameterization trick. Since sampling operations aren't differentiable (which would break backpropagation), we use a clever workaround:

1. The encoder outputs parameters μ (mean) and σ (standard deviation)
2. We sample from a standard normal distribution: ε ~ N(0,1)
3. We compute z = μ + σ * ε

This allows gradients to flow through the network during backpropagation while still incorporating randomness.

## My Implementation Journey

When I first started implementing a VAE, I decided to use TensorFlow and focus on the MNIST dataset as a proof of concept. Here's what my implementation journey looked like:

### Architecture Design

For my encoder, I used:
- A series of Conv2D layers to extract features
- Followed by dense layers to predict μ and log(σ²)

For my decoder:
- Dense layers to project from latent space to a shape compatible with deconvolution
- Transposed convolution layers to reconstruct the image

### The Loss Function Challenge

Implementing the loss function correctly was tricky. The reconstruction term was straightforward using binary cross-entropy since MNIST images are essentially binary. The KL divergence term, for a normal distribution with mean μ and standard deviation σ against a standard normal, simplifies to:

$$D_{KL} = \frac{1}{2} \sum (1 + \log(\sigma^2) - \mu^2 - \sigma^2)$$

I initially made the mistake of not averaging over the batch dimension, which led to unstable training. Lesson learned!

```python
def vae_loss(recon_x, x, mean, logvar, kl_weight=0.1):
    recon_loss = F.mse_loss(recon_x, x,reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon_loss + kl_weight * kl_loss
```

### Balancing the Loss Terms

I found that the KL divergence term often dominated the loss early in training, causing the model to prioritize fitting to the prior over reconstructing inputs. I experimented with annealing strategies, gradually increasing the weight of the KL term during training, which significantly improved results.

## Applications and Use Cases

After getting my VAE working, I explored several applications:

### Image Generation

The most obvious application was generating new handwritten digits by sampling from the latent space. What fascinated me was how the model captured the essence of handwriting styles, including slant, thickness, and proportions.

### Anomaly Detection

By measuring reconstruction error, I could identify images that didn't conform to the learned distribution. This proved surprisingly effective for detecting outliers in the dataset.

### Feature Learning

The latent representations captured meaningful features about the data. By visualizing the latent space, I could see clear clustering of different digit classes, even though the model was trained in an unsupervised manner.

### Data Augmentation

For datasets with limited samples, I could generate additional training examples by sampling from regions of the latent space corresponding to specific classes.

## Extending to More Complex Datasets

After success with MNIST, I moved to more complex datasets:
### Architecture

## ResNet Block
```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1,bias=False),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(inplace=True)
        )
        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1,bias=False)
        self.group_norm = nn.GroupNorm(32, out_channels)
    def forward(self, x):
        def execute(x):
            residual = x
            if self.in_channels != self.out_channels:
                residual = self.channel_up(x)
            return self.group_norm(self.block(x) + residual)
        return execute(x)
  ```  
## Encoder
```python
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.initial = nn.Conv2d(in_channels, 64, 3, padding=1,bias=False)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            nn.MaxPool2d(2),
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            nn.MaxPool2d(2),
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            nn.MaxPool2d(2),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )
        self.conv = nn.Conv2d(512, latent_dim * 2, 3, padding =1,bias=False)
        self.final = nn.Conv2d(latent_dim*2, latent_dim*2, 1, stride=1,bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x= self.conv(x)
        x = self.final(x)
        return x
```
## Decoder
```python
class Decoder(nn.Module):
    def __init__(self, out_channels=3, latent_dim=4):
        super().__init__()
        self.initial = nn.Conv2d(latent_dim, 512, 1,bias=False)
        self.residual_blocks = nn.Sequential(
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(512, 256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256, 128),
            ResidualBlock(128, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128, 64),
            ResidualBlock(64, 64)
            #nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.final1 = nn.Conv2d(64, out_channels, 3, padding=1,bias=False)
        self.final2 = nn.Conv2d(out_channels, out_channels, 1,bias=False)

    def forward(self, x):
        x = self.initial(x)
        x = self.residual_blocks(x)
        x = self.final2(self.final1(x))
        return x
```

## VAE Wrapper
```python
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_dim)
        self.decoder = Decoder(in_channels, latent_dim)
        self.apply(self.init_weight)

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=1)
        logvar = torch.clamp(logvar, -10, 10)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, logvar, eps=None):
        std = torch.exp(0.5 * logvar)
        if eps == None:
            eps = torch.randn_like(std)
        z = mean + eps * std
        return torch.tanh(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

vae = VAE(in_channels=in_channel, latent_dim=latent_dim)
```
### Faces Dataset

Working with a face dataset introduced new challenges. The reconstructions were initially blurry, which led me to explore:

1. **Beta-VAE**: Adding a β parameter to control the weight of the KL term
2. **Deeper architectures**: More convolutional layers with batch normalization
3. **Perceptual losses**: Incorporating VGG feature-based losses for sharper results

## Sample Output

## Face Construction

![VAE output of Anime Character](./data/images/vae-color.png)

## COCO 17 Dataset


![VAE output of Anime Character](./data/images/vae-coco-color.png)