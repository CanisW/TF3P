from functools import partial
import torch
from torch import nn 
from torch.nn import functional as F


def squash(input_tensor, dim=-1):
    squared_norm = (input_tensor ** 2).sum(dim, keepdim=True)
    # full equation
    # output_tensor = (
    #     squared_norm * input_tensor /
    #     ((1. + squared_norm) * torch.sqrt(squared_norm))
    # )
    output_tensor = input_tensor * torch.sqrt(squared_norm) /(1 + squared_norm)
    return output_tensor


class ConvLayer(nn.Module):
    def __init__(
        self, 
        in_channels=2, 
        out_channels=128,
        kernel_size=3,
        grid_size=50,
    ):
        super(ConvLayer, self).__init__()

        def dim_calc(i, x=2 * grid_size):
            for _ in range(i):
                x = int(x / 2) - 4
            return x
        shape_befpool = [dim_calc(i + 1) for i in range(2)]

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 64, kernel_size),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Conv3d(64, 64, kernel_size),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d(int(shape_befpool[0] / 2)),  # 2,2,2 pooling
            nn.Conv3d(64, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.AdaptiveMaxPool3d(int(shape_befpool[1] / 2)),
            nn.Conv3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Conv3d(128, out_channels, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        """forword pass
        
        Args:
            x (torch.tensor): a tensor with shape = 
                batch_size x 2 x 50 x 50 x 50
         
        Returns:
            torch.tensor: shape batch_size x 128 x 5 x 5 x 5
        """
        return self.conv(x)


class PrimaryCaps(nn.Module):
    def __init__(
        self, 
        num_capsules=32,
        in_channels=128,
        out_channels=4,
        kernel_size=3,
    ):
        '''

        Parameters
        ----------
        num_capsules: To simplify, this param does NOT mean the actual num of caps in this layer. After conv, along the
        channel dim, the output tensor can be split into chunks of tensors. The num of these chunks is num_capsules and
        the num of each chunk's channels is out_channels. Each chunk is an N dim tensor and one can see it as an N-1 dim
        grids of vectors whose dim is out_channels. One capsule outputs one vector, so the num of caps within each chunk
        is equal to the size of the grids. Hence, the actual num of caps in this layer is equal to the num of chunks *
        the size of the grids.
        in_channels: output channels from the layer before
        out_channels: dim of out vectors, not the num of out channels in conv
        kernel_size: conv kernel
        '''
        super(PrimaryCaps, self).__init__()

        self.capsules = nn.Sequential(
            nn.Conv3d(in_channels, out_channels * num_capsules, kernel_size),
            Lambda(lambda x: x.view(-1, num_capsules, out_channels, *x.shape[-3:])),
            Lambda(lambda x: x.permute(0, 1, 3, 4, 5, 2).contiguous()),
            Lambda(lambda x: x.view(-1, num_capsules * x.shape[2] ** 3, out_channels))  # -1, actual num of caps, vect_dims
        )
    
    def forward(self, x):
        u = self.capsules(x)
        return squash(u)


class DigitCaps(nn.Module):
    def __init__(
        self,
        num_capsules=166,
        num_routes=32 * 3 ** 3,
        in_channels=4,
        out_channels=8,
    ):
        '''

        Parameters
        ----------
        num_capsules: num_out_caps, actual caps here
        num_routes: the actual num of input capsules, see doc string in PrimaryCaps
        in_channels: dims_in_vectors
        out_channels: dims_out_vectors
        USE_CUDA
        '''
        super(DigitCaps, self).__init__()

        self.in_channels = in_channels
        self.num_routes = num_routes
        self.num_capsules = num_capsules

        self.W = nn.Parameter(
            torch.randn(
                1, 
                num_capsules,
                num_routes,
                out_channels, 
                in_channels
            )
        )

    def forward(self, u):
        # make all variables' dim same to self.W: batch_size * num_out_caps * num_in_caps * out_vect_dims * in_out_dims
        u = u.unsqueeze(1).unsqueeze(4)
        u_hat = torch.matmul(self.W, u)
        b_ij = torch.zeros(1, self.num_capsules, self.num_routes, 1, 1, device=u.device)

        num_iterations = 3
        for iteration in range(num_iterations):
            c_ij = F.softmax(b_ij, dim=2)
            s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)
            v_j = squash(s_j, dim=3)
            if iteration < num_iterations - 1:
                a_ij = torch.matmul(u_hat.transpose(-1, -2), v_j)
                b_ij = b_ij + a_ij

        return v_j.squeeze()


class Decoder(nn.Module):
    def __init__(
        self,
        in_capsules=166,
        in_channels=8,
        grid_size=50,
        kernel_size=3,
    ):
        super(Decoder, self).__init__()

        def dim_calc(i, x=2 * grid_size):
            for _ in range(i):
                x = int(x / 2) - 4
            return x
        shape_befpool = [dim_calc(i + 1) for i in range(2)]
        shape_aftconv = int((((grid_size - 4) / 2 - 4) / 2 - 4) - 2)  # shape after conv layers in ConvLayer and PrimaryCaps

        
        self.reconstruction_layers = nn.Sequential(
            Lambda(lambda x: x.view(-1, in_capsules * in_channels)),
            nn.Linear(in_capsules * in_channels, 128 * shape_aftconv ** 3),
            nn.ReLU(inplace=True),
            Lambda(lambda x: x.view(-1, 128, *(shape_aftconv,) * 3)),

            nn.ConvTranspose3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.Upsample((shape_befpool[1],) * 3, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(128, 128, kernel_size),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.Upsample((shape_befpool[0],) * 3, mode='trilinear', align_corners=True),
            nn.ConvTranspose3d(64, 64, kernel_size),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose3d(64, 2, kernel_size),
            # nn.Sigmoid()
        )
        
    def forward(self, v, mask):
        # fp.shape: BS * bits
        reconstructions = self.reconstruction_layers(v * mask)
        return reconstructions


class Lambda(torch.nn.Module):
    # e.g. Lambda(lambda x: x.view(-1, 20))
    def __init__(self, func, *inputs, **kwargs):
        super().__init__()
        self.func = partial(func, *inputs, **kwargs)

    def forward(self, *inputs, **kwargs):
        return self.func(*inputs, **kwargs)


class RaggedDataParallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super().__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def scatter(self, inputs, kwargs, device_ids):
        from torch.nn.parallel import scatter
        def chunk_it(seq, num, devices):
            assert isinstance(num, (int, list, tuple))
            if isinstance(num, int):
                chunk_sizes = [len(seq) / float(num), ] * num
            else:
                chunk_sizes = map(int, num)
            out = []
            last = 0.0
            for size, device in zip(chunk_sizes, devices):
                out.append(torch.tensor(seq[int(last):int(last + size)], device=device))
                last += size
            return out

        devices = [torch.device('cuda:' + str(i)) for i in device_ids]
        nums_atoms_ckd = chunk_it(inputs[3], len(device_ids), devices)
        chunk_sizes = [sum(num_atoms) for num_atoms in nums_atoms_ckd]
        gs_charge_ckd, atom_type_ckd, pos_ckd = [chunk_it(i, chunk_sizes, devices) for i in inputs[:3]]
        inputs = list(zip(gs_charge_ckd, atom_type_ckd, pos_ckd, nums_atoms_ckd))
        kwargs = scatter(kwargs, device_ids, self.dim) if kwargs else []

        if len(inputs) < len(kwargs):
            inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
        elif len(kwargs) < len(inputs):
            kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
        inputs = tuple(inputs)
        kwargs = tuple(kwargs)

        return inputs, kwargs
