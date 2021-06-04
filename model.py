import torch
import torch.nn as nn
import math

ngf = 64
nz = 300    # Base no. of out_channels
nc = 3      # No of channels in image
# n is the power of 2 for getting to the ImageSize


class VAE(nn.Module):
    def __init__(self, image_size):
        super(VAE, self).__init__()
        self.nz = nz
        n = math.log2(image_size)

        assert n == round(n), 'image_size must be a power of 2'
        assert n >= 3, 'image_size must be at least 8'
        n = int(n)

        self.conv_mu = nn.Conv2d(
            in_channels=ngf * 2 ** (n - 3), out_channels=nz, kernel_size=4)
        self.conv_log_var = nn.Conv2d(
            in_channels=ngf * 2 ** (n - 3), out_channels=nz, kernel_size=4)

        self.encoder = nn.Sequential()
        # input is (nc) x 64 x 64
        self.encoder.add_module(
            'input-conv',
            nn.Conv2d(
                in_channels=nc, out_channels=ngf, kernel_size=4, stride=2, padding=1, bias=True))
        self.encoder.add_module(
            'input-relu',
            nn.ReLU(inplace=True))
        for i in range(n - 3):
            # state size. (ngf) x 32 x 32
            self.encoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i + 1)),
                                    nn.Conv2d(ngf * 2 ** i, ngf * 2 ** (i + 1), 4, 2, 1, bias=True))
            self.encoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i + 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i + 1)))
            self.encoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i + 1)),
                                    nn.ReLU(inplace=True))

        self.decoder = nn.Sequential()
        # input is Z, going into a convolution
        self.decoder.add_module('input-conv',
                                nn.ConvTranspose2d(nz, ngf * 2 ** (n - 3), 4, 1, 0, bias=True))
        self.decoder.add_module('input-batchnorm',
                                nn.BatchNorm2d(ngf * 2 ** (n - 3)))
        self.decoder.add_module('input-relu',
                                nn.ReLU(inplace=True))

        # state size. (ngf * 2**(n-3)) x 4 x 4

        for i in range(n - 3, 0, -1):
            self.decoder.add_module('pyramid_{0}-{1}_conv'.format(ngf * 2 ** i, ngf * 2 ** (i - 1)),
                                    nn.ConvTranspose2d(ngf * 2 ** i, ngf * 2 ** (i - 1), 4, 2, 1, bias=True))
            self.decoder.add_module('pyramid_{0}_batchnorm'.format(ngf * 2 ** (i - 1)),
                                    nn.BatchNorm2d(ngf * 2 ** (i - 1)))
            self.decoder.add_module('pyramid_{0}_relu'.format(ngf * 2 ** (i - 1)),
                                    nn.ReLU(inplace=True))

        self.decoder.add_module('output-conv', nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=True))
        self.decoder.add_module('output-tanh', nn.Tanh())

        # weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def encode(self, input):
        """
        :param input: [bsz, 3, 256, 256]
        :return: mu [bsz, z_dim, 1, 1]. log_var [bsz, z_dim, 1, 1]
        """
        output = self.encoder(input)
        output = output.squeeze(-1).squeeze(-1)
        return [self.conv_mu(output), self.conv_log_var(output)]

    def reparameterize(self, mu, log_var):
        """
        :param mu: can be any shape
        :param log_var: can be any shape
        :return: same shape as mu
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        """
        :param z: [bsz, z_dim, 1, 1]
        :return reconst_x: [bsz, 3, 256, 256]
        """
        return self.decoder(z)

    def forward(self, x):
        """
        :param x: [bsz, 3, 256, 256]
        :return: reconst_x: [bsz, 3, 256, 256]
                 mu, log_var: [bsz, z_dim]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu.squeeze(-1).squeeze(-1), log_var.squeeze(-1).squeeze(-1)
