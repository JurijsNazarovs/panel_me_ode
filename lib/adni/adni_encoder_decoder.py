import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

seqlen = 3


class Encoder3d(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Encoder3d, self).__init__()
        self.adniConvInit(input_dim, output_dim)
        #self.update=self.update_maxpool_indices_size

    def forward(self, data):
        n, t = data.shape[:2]
        data = data.reshape((-1, ) + data.shape[2:])
        data = self.encoder(data)
        data = data.reshape((n, t) + data.shape[1:])
        return data

    def encoder(self, inp):
        hid = self.convA1(inp)
        hid = self.swishA1(hid)
        # print('after first conv', hid.shape)

        self.size1 = hid.size()
        hid, self.indices1 = self.maxpoolA1(hid)
        hid = self.convA2(hid)
        hid = self.swishA2(hid)
        # print('after second conv', hid.shape)

        self.size2 = hid.size()
        hid, self.indices2 = self.maxpoolA2(hid)
        hid = self.convA3(hid)
        hid = self.swishA3(hid)
        # print('after third conv', hid.shape)

        self.size3 = hid.size()
        hid, self.indices3 = self.maxpoolA3(hid)
        hid = self.convA4(hid)
        hid = self.swishA4(hid)
        # print('after fourth conv', hid.shape)

        self.size4 = hid.size()
        hid, self.indices4 = self.maxpoolA4(hid)
        hid = self.convA5(hid)
        hid = self.swishA5(hid)
        # print('after fifth conv', hid.shape)

        self.size5 = hid.size()
        hid, self.indices5 = self.maxpoolA5(hid)
        hid = self.convA6(hid)
        hid = self.swishA6(hid)
        # print('after sixth conv', hid.shape)

        self.size6 = hid.size()
        hid, self.indices6 = self.maxpoolA6(hid)
        hid = self.convA7(hid)
        hid = self.swishA7(hid)
        # print('after seventh conv', hid.shape)

        hid = self.flat1(hid)
        hid = self.fc1(hid)
        hid = self.flat2(hid)

        return hid

    def adniConvInit(self, input_dim, output_dim):
        #Convolution 1
        self.convA1 = nn.Conv3d(input_dim,
                                8,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        nn.init.xavier_uniform_(self.convA1.weight)  #Xaviers Initialisation
        self.swishA1 = nn.ReLU()

        #Max Pool 1
        self.maxpoolA1 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 2
        self.convA2 = nn.Conv3d(8,
                                16,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                bias=False)
        nn.init.xavier_uniform_(self.convA2.weight)
        self.swishA2 = nn.ReLU()

        #Max Pool 2
        self.maxpoolA2 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 3
        self.convA3 = nn.Conv3d(16,
                                32,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        nn.init.xavier_uniform_(self.convA3.weight)
        self.swishA3 = nn.ReLU()

        #Max Pool 3
        self.maxpoolA3 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 4
        self.convA4 = nn.Conv3d(32,
                                64,
                                kernel_size=3,
                                stride=1,
                                padding=2,
                                bias=False)
        nn.init.xavier_uniform_(self.convA4.weight)
        self.swishA4 = nn.ReLU()

        #Max Pool 4
        self.maxpoolA4 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 5
        self.convA5 = nn.Conv3d(64,
                                128,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        nn.init.xavier_uniform_(self.convA5.weight)
        self.swishA5 = nn.ReLU()

        #Max Pool 5
        self.maxpoolA5 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 6
        self.convA6 = nn.Conv3d(128,
                                256,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        nn.init.xavier_uniform_(self.convA6.weight)
        self.swishA6 = nn.ReLU()

        #Max Pool 6
        self.maxpoolA6 = nn.MaxPool3d(kernel_size=2, return_indices=True)

        #Convolution 7
        self.convA7 = nn.Conv3d(256,
                                output_dim,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                bias=False)
        nn.init.xavier_uniform_(self.convA7.weight)
        self.swishA7 = nn.ReLU()

        self.flat1 = nn.Flatten(2)  #output_dim, 8
        self.fc1 = nn.Linear(8, 1)  #output_dim, 1
        self.flat2 = nn.Flatten(1)  #output_dim

    def update(self, inp):
        # This is a function necessary to update pooling indices and size,
        # it reurns nothing, but makes a dry run (feedforward) with input
        _ = self.forward(inp)


class Decoder3d(nn.Module):
    def __init__(self,
                 conv_module,
                 input_dim=512,
                 output_dim=3,
                 out_shape=(121, 145, 121)):
        super(Decoder3d, self).__init__()
        self.adniDeconvInit(input_dim, output_dim)
        self.cm = conv_module  #reference to update sizes
        self.extend_to_3d = nn.Linear(input_dim, 2 * 2 * 2 * input_dim)
        self.out_shape = out_shape

    def forward(self, data_):
        # n_traj, n, t = data.shape[:3]
        # data = data.reshape((-1, data.shape[-1]))  #bs, input_dim
        # data = self.extend_to_3d(data)  #bs, inp_dim*2*2*2
        # data = data.reshape(data.shape[:-1] +
        #                     (-1, 2, 2, 2))  #bs, inp_dim, 2, 2
        # data = self.decoder(data)
        # data = data.reshape((n_traj, n, t, -1) + self.out_shape)
        # return data

        ans = []
        for i in range(data_.shape[0]):
            # Have to compute decoder of trajectories by 1, because of
            # size and ind from encoder maxpool
            data = data_[i][None]
            n_traj, n, t = data.shape[:3]

            data = data.reshape((-1, data.shape[-1]))  #bs, input_dim
            data = self.extend_to_3d(data)  #bs, inp_dim*2*2*2
            data = data.reshape(data.shape[:-1] +
                                (-1, 2, 2, 2))  #bs, inp_dim, 2, 2
            data = self.decoder(data, n)
            data = data.reshape((n_traj, n, t, -1) + self.out_shape)
            ans.append(data)
        ans = torch.cat(ans)
        return ans

    def extend_indices(self, indices, n, n_out):
        # n is batch_size
        # n_out is desired size
        indices_ = indices
        while indices_.shape[0] < n_out:
            indices_ = torch.cat([indices_, indices[-n:]])
        return indices_

    def extend_size(self, size, n_out):
        # n_out is desired size
        size_ = list(size)
        size_[0] = n_out
        return size_

    def decoder(self, hid, n):
        # n is batch size, necessary to extend last cm.indices
        # n_out size we need to achieve for indices in case of extrapolation
        n_out = hid.shape[0]
        out = self.deconvB7(hid)
        out = self.swishB7(out)
        # # print('before first unpool', out.shape)

        indices6 = self.extend_indices(self.cm.indices6, n, n_out)
        size6 = self.extend_size(self.cm.size6, n_out)
        try:
            out = self.maxunpoolB6(out, indices6, size6)
        except:
            import pdb
            pdb.set_trace()
            print("Debug")
        out = self.deconvB6(out)
        out = self.swishB6(out)

        # # print('before second unpool', out.shape)
        indices5 = self.extend_indices(self.cm.indices5, n, n_out)
        size5 = self.extend_size(self.cm.size5, n_out)
        out = self.maxunpoolB5(out, indices5, size5)
        out = self.deconvB5(out)
        out = self.swishB5(out)

        # print('before third unpool', out.shape)
        indices4 = self.extend_indices(self.cm.indices4, n, n_out)
        size4 = self.extend_size(self.cm.size4, n_out)
        out = self.maxunpoolB4(out, indices4, size4)
        out = self.deconvB4(out)
        out = self.swishB4(out)

        # print('before fourth unpool', out.shape)
        indices3 = self.extend_indices(self.cm.indices3, n, n_out)
        size3 = self.extend_size(self.cm.size3, n_out)
        out = self.maxunpoolB3(out, indices3, size3)
        out = self.deconvB3(out)
        out = self.swishB3(out)

        # print('before fifth unpool', out.shape)
        indices2 = self.extend_indices(self.cm.indices2, n, n_out)
        size2 = self.extend_size(self.cm.size2, n_out)
        out = self.maxunpoolB2(out, indices2, size2)
        out = self.deconvB2(out)
        out = self.swishB2(out)

        # print('before sixth unpool', out.shape)
        indices1 = self.extend_indices(self.cm.indices1, n, n_out)
        size1 = self.extend_size(self.cm.size1, n_out)
        out = self.maxunpoolB1(out, indices1, size1)
        out = self.deconvB1(out)
        out = self.swishB1(out)

        return out

    def adniDeconvInit(self, input_dim, output_dim):
        #De Convolution 7
        self.deconvB7 = nn.ConvTranspose3d(in_channels=input_dim,
                                           out_channels=256,
                                           kernel_size=3,
                                           padding=1)
        nn.init.xavier_uniform_(self.deconvB7.weight)
        self.swishB7 = nn.ReLU()

        #Max UnPool 6
        self.maxunpoolB6 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 6
        self.deconvB6 = nn.ConvTranspose3d(in_channels=256,
                                           out_channels=128,
                                           kernel_size=3,
                                           padding=1)
        nn.init.xavier_uniform_(self.deconvB6.weight)
        self.swishB6 = nn.ReLU()

        #Max UnPool 5
        self.maxunpoolB5 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 5
        self.deconvB5 = nn.ConvTranspose3d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=3,
                                           padding=1)
        nn.init.xavier_uniform_(self.deconvB5.weight)
        self.swishB5 = nn.ReLU()

        #Max UnPool 4
        self.maxunpoolB4 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 4
        self.deconvB4 = nn.ConvTranspose3d(in_channels=64,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=2)
        nn.init.xavier_uniform_(self.deconvB4.weight)
        self.swishB4 = nn.ReLU()

        #Max UnPool 3
        self.maxunpoolB3 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 3
        self.deconvB3 = nn.ConvTranspose3d(in_channels=32,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1)
        nn.init.xavier_uniform_(self.deconvB3.weight)
        self.swishB3 = nn.ReLU()

        #Max UnPool 2
        self.maxunpoolB2 = nn.MaxUnpool3d(kernel_size=2)

        #De Convolution 2
        self.deconvB2 = nn.ConvTranspose3d(in_channels=16,
                                           out_channels=8,
                                           kernel_size=3,
                                           padding=2)
        nn.init.xavier_uniform_(self.deconvB2.weight)
        self.swishB2 = nn.ReLU()

        #Max UnPool 1
        self.maxunpoolB1 = nn.MaxUnpool3d(kernel_size=2)

        #DeConvolution 1
        self.deconvB1 = nn.ConvTranspose3d(in_channels=8,
                                           out_channels=output_dim,
                                           kernel_size=3,
                                           padding=1)
        nn.init.xavier_uniform_(self.deconvB1.weight)
        self.swishB1 = nn.ReLU()
