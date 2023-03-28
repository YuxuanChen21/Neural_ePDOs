import torch
import torch.nn as nn
import numpy as np
from numpy import linalg as la
import time
import math

from torch.autograd import Function
from torch.autograd.function import once_differentiable
from anti import Downsample
from random import gauss
from rbf.pde.fd import weight_matrix
from rbf.basis import get_rbf


def kronecker(x, y):
    return torch.einsum('ij,mn->imjn', [x, y]).reshape(x.size(0) * y.size(0), x.size(1) * y.size(1))


def direct_sum(a, b):
    n = a.size(0) + b.size(0)
    out = torch.zeros((n, n), dtype=a.dtype, device=a.device)
    out[0:a.size(0), 0:a.size(0)] = a
    out[a.size(0):n, a.size(0):n] = b
    return out


def ortho_basis(x):  # num_basis x dim_vec
    x, _ = torch.qr(x.transpose(0, 1))
    return x.transpose(0, 1)


def solve(A):
    n = torch.matrix_rank(A)
    # A=A.to(torch.float64)
    u, s, v = torch.svd(A, some=True)
    # v=v.to(torch.float32)

    return v[::, n:A.size(1)]


class diff_rep:
    ### n represent the order of the rotation
    def __init__(self, n, flip=False):

        self.rep_e = [torch.tensor([[1.]])]
        self.rep_m = [torch.tensor([[1.]])]
        self.order = 1
        self.n = n
        t = 2 * np.pi / n
        self.flip = flip
        self.g_e = torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        if (flip == True):
            self.g_m = torch.tensor([[-1., 0.], [0., 1.]])
            self.rep_m.append(self.g_m)
        self.rep_e.append(self.g_e)

    def next(self):
        if (self.order == 4):
            print('Order is less than 5')
        a = self.rep_e[self.order]
        self.order = self.order + 1
        b = torch.zeros(self.order + 1, self.order + 1)
        for i in range(self.order):
            for j in range(1, self.order):
                b[i, j] = self.g_e[0, 0] * a[i, j] + self.g_e[0, 1] * a[i, j - 1]
            b[i, 0] = self.g_e[0, 0] * a[i, 0]
            b[i, self.order] = self.g_e[0, 1] * a[i, self.order - 1]
        for j in range(1, self.order):
            b[self.order, j] = self.g_e[1, 0] * a[i, j] + self.g_e[1, 1] * a[i, j - 1]
        b[self.order, 0] = self.g_e[1, 0] * a[i, 0]
        b[self.order, self.order] = self.g_e[1, 1] * a[i, self.order - 1]
        self.rep_e.append(b)
        if (self.flip == True):
            a = self.rep_m[self.order - 1]
            b = torch.zeros(self.order + 1, self.order + 1)
            for i in range(self.order):
                for j in range(1, self.order):
                    b[i, j] = self.g_m[0, 0] * a[i, j] + self.g_m[0, 1] * a[i, j - 1]
                b[i, 0] = self.g_m[0, 0] * a[i, 0]
                b[i, self.order] = self.g_m[0, 1] * a[i, self.order - 1]
            for j in range(1, self.order):
                b[self.order, j] = self.g_m[1, 0] * a[i, j] + self.g_m[1, 1] * a[i, j - 1]
            b[self.order, 0] = self.g_m[1, 0] * a[i, 0]
            b[self.order, self.order] = self.g_m[1, 1] * a[i, self.order - 1]
            self.rep_m.append(b)

    def __getitem__(self, i):
        if (i > 4):
            print('Order is less than 5')
        while (i > self.order):
            self.next()
        if (self.flip == True):
            return self.rep_e[i], self.rep_m[i]
        else:
            return self.rep_e[i]

    def e(self, n):
        if (self.flip):
            a, _ = self[n]
        else:
            a = self[n]
        if (n != 0):
            return direct_sum(self.e(n - 1), a)
        else:
            return a

    def m(self, n):
        if (self.flip):
            _, a = self[n]
        else:
            a = self[n]
        if (n != 0):
            return direct_sum(self.e(n - 1), a)
        else:
            return a


class c_regular:
    def __init__(self, n):
        self.type = 'cn'
        self.n = n
        self.rep_e = torch.zeros(n, n)
        self.rep_e[1:n, 0:n - 1] = torch.eye(n - 1)
        self.rep_e[0, n - 1] = 1.
        self.dim = n


class d_regular:
    def __init__(self, n):
        self.n = n
        self.type = 'dn'
        self.rep_e = torch.zeros(2 * n, 2 * n)
        self.rep_e[1:n, 0:n - 1] = torch.eye(n - 1)
        self.rep_e[0, n - 1] = 1.
        self.rep_e[n:2 * n - 1, n + 1:2 * n] = torch.eye(n - 1)
        self.rep_e[2 * n - 1, n] = 1.
        self.rep_m = torch.zeros(2 * n, 2 * n)
        self.rep_m[0:n, n:2 * n] = torch.eye(n)
        self.rep_m[n:2 * n, 0:n] = torch.eye(n)
        self.dim = 2 * n


class c_qotient:
    '''
    '''

    def __init__(self, n, m):
        self.n = n
        self.dim = n // m

        rep = c_regular(n // m)
        self.rep_e = rep.rep_e


class trivial:
    def __init__(self):
        self.dim = 1
        self.type = 'trivial'
        self.rep_e = torch.eye(1)
        self.rep_m = torch.eye(1)


def c_lin_base(in_rep, out_rep):
    '''
    return: (num_base, dim_type_out, dim_type_in)
    '''

    A = torch.eye(in_rep.size(0) * out_rep.size(1)) - kronecker(out_rep, torch.inverse(in_rep.transpose(0, 1)))
    w = solve(A).transpose(0, 1).reshape(-1, out_rep.size(0), in_rep.size(0))
    return w


def make_coord(kernel_size):
    x = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    coord = torch.meshgrid([-x, x])
    coord = torch.stack([coord[1], coord[0]], -1)

    return coord.reshape(kernel_size ** 2, 2)


def d_lin_base(type_in, type_out, type_in_m, type_out_m):
    '''
    return: (num_base, dim_type_out, dim_type_in)
    '''

    A = torch.eye(type_in.size(0) * type_out.size(1)) - kronecker(type_out, torch.inverse(type_in.transpose(0, 1)))
    A_ = torch.eye(type_in_m.size(0) * type_out_m.size(1)) - kronecker(type_out_m,
                                                                       torch.inverse(type_in_m.transpose(0, 1)))
    w = solve(torch.cat((A, A_), dim=0)).transpose(0, 1).reshape(-1, type_out.size(0), type_in.size(0))
    return w


def cat_lin_base(a, b, dim=0):
    b1, n1, m1, h, w = a.shape
    b2, n2, m2, h, w = b.shape
    if (dim == 0):
        c = torch.zeros((b1 + b2, n1 + n2, m1, h, w))
        c[0:b1, 0:n1] = a
        c[b1:, n1:] = b
    else:
        c = torch.zeros((b1 + b2, n1, m1 + m2, h, w))
        c[0:b1, ::, 0:m1] = a
        c[b1:, ::, m1:] = b
    return c


def kaiming_init(base, num_in, num_out, normal=True):
    '''
        base: the base of the conv_base or conv_fast
        num_in: number of representation in of the input in conv_base of conv_fast
        num_out: number of representation in of the output in conv_base of conv_fast
        normal: using the normal or constant initialization
    '''
    f = torch.sum(base * base) * num_in / (base.size(1))
    if (normal == True):
        weight = torch.sqrt(1 / f) * torch.randn(num_in, num_out, base.size(0))
    else:
        weight = torch.sqrt(12 / f) * (torch.rand(num_in, num_out, base.size(0)) - 0.5)
    return weight


def make_rbffd(order, kernel_size):
    diff = []
    coord = make_coord(kernel_size)

    for i in range(order + 1):
        w = weight_matrix(torch.zeros(1, 2).numpy(), coord.numpy(), kernel_size ** 2, [i, order - i],
                          phi='phs6', eps=0.5).toarray()
        w = torch.tensor(w).reshape(kernel_size, kernel_size)
        # print("rbf dis")
        diff.append(w)
        tensor = torch.stack(diff, 0)

    return tensor.to(torch.float32)


def make_gauss(order, kernel_size):
    diff = []
    coord = make_coord(kernel_size)
    gauss = get_rbf('ga')
    for i in range(order + 1):
        w = gauss(coord, torch.zeros(1, 2), eps=0.99, diff=[i, order - i]).reshape(kernel_size, kernel_size)
        w = torch.tensor(w)
        diff.append(w)
    tensor = torch.stack(diff, 0)
    return tensor.to(torch.float32)


class conv(nn.Module):
    def __init__(self, base, num_in, num_out, groups=1, stride=1):
        '''
            Group: group containing the base
            bases: number of basis kernel in a specified type
            param: parameter tensor of shape dim_in x dim_out x bases
            base: bases of a specified type of kernel, shape of  bases x dim_rep_out x dim_rep_in x kernel_size x kernel_size
        '''
        super(conv, self).__init__()
        self.base = torch.nn.Parameter(base, requires_grad=False)
        self.num_in = num_in // groups
        self.num_out = num_out
        self.groups = groups
        self.dim_rep_in = self.base.size(2)
        self.dim_rep_out = self.base.size(1)
        self.bases = self.base.size(0)
        self.param = torch.nn.Parameter(kaiming_init(base, self.num_in, num_out))
        self.stride = stride
        self.size = self.base.size(4)

        if (self.stride != 1):
            self.pool = nn.MaxPool2d(self.stride, self.stride)
        self.eval()

    def forward(self, x):
        # get the kernel of the conv from the base
        if (self.training):
            self.kernel = torch.matmul(self.param.reshape(-1, self.bases), self.base.reshape(self.bases, -1)).reshape( \
                self.num_in, self.num_out, self.dim_rep_out, self.dim_rep_in, self.size, self.size)
            self.kernel = self.kernel.permute((1, 2, 0, 3, 4, 5)).reshape(self.num_out * self.dim_rep_out,
                                                                          self.num_in * self.dim_rep_in, self.size,
                                                                          self.size)

        out = nn.functional.conv2d(x, self.kernel, bias=None, stride=1, padding=math.floor(self.size / 2),
                                   groups=self.groups)
        if (self.stride != 1):
            return self.pool(out)
        else:
            return out

    def eval(self):
        self.kernel = torch.einsum('ijk,kmnpq->jminpq', self.param, self.base) \
            .reshape(self.num_out * self.dim_rep_out, self.num_in * self.dim_rep_in, self.size, self.size)
        self.kernel = self.kernel.detach()


class gnorm(nn.Module):
    def __init__(self, in_type, groups):
        super(gnorm, self).__init__()
        self.bn = nn.GroupNorm(groups, in_type[1])
        self.num_in = in_type[1]
        self.weight = nn.Parameter(torch.ones(1, self.num_in, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, self.num_in, 1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        return (self.bn(x.reshape(b, self.num_in, c // self.num_in, h, w)) * self.weight + self.bias).reshape(x.size())


class GroupBatchNorm(nn.Module):

    def __init__(self, num_rep, dim_rep, affine=False, momentum=0.1, track_running_stats=True):
        super(GroupBatchNorm, self).__init__()
        self.momentum = momentum
        self.bn = nn.BatchNorm3d(num_rep, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.num_rep = num_rep
        self.dim_rep = dim_rep

    def forward(self, x):
        shape = x.shape
        x = self.bn(x.reshape(x.size(0), self.num_rep, self.dim_rep, x.size(2), x.size(3)))
        x = x.reshape(shape)
        return x


class GroupPooling(nn.Module):
    def __init__(self, dim_rep, num_rep, type='max'):
        super(GroupPooling, self).__init__()
        if (type == 'avg'):
            self.pool = nn.AdaptiveAvgPool3d((dim_rep, 1, 1), (dim_rep, 1, 1))
        else:
            # self.pool=nn.AdaptiveMaxPool3d(1)
            self.pool = nn.MaxPool3d((dim_rep, 1, 1), (dim_rep, 1, 1))
        self.dim = dim_rep
        self.num_rep = num_rep

    def forward(self, x):
        size = x.size()
        x = x.reshape(x.size(0), self.num_rep, self.dim, x.size(2), x.size(3))
        x = self.pool(x)
        return x.reshape(x.size(0), x.size(1), x.size(3), x.size(4))


class nlpdo_torch(nn.Module):
    def __init__(self, group, in_type, out_type, order, reduction, s, g, stride=1):
        super(nlpdo_torch, self).__init__()
        self.group = group
        self.reduction = reduction
        self.g = g
        self.stride = stride
        self.rep, self.num_in = in_type
        self.num_mid = int(self.num_in * self.group.dim_rep(self.rep) // (self.reduction * self.group.dim))
        self.order = order
        self.s = s

        self.conv1 = self.group.conv1x1(in_type, ('regular', self.num_mid))
        self.gn = gnorm(('regular', self.num_mid), self.num_mid)

        self.conv2 = self.make_conv2()
        self.conv3 = self.group.conv1x1(in_type, out_type)


    def forward(self, x):
        x_ = nn.functional.relu(self.gn(self.conv1(x)))
        W = self.conv2(x_) / 5
        b, c, h, w = W.shape
        W = W.reshape(b, 1, -1, self.group.dim_rep(self.rep) * 25, h, w)
        W = W / (torch.norm(W, 1, dim=3, keepdim=True) + 0.01)
        W = W.reshape(b, 1, -1, 25, h, w)
        x = nn.functional.unfold(x, 5, padding=2).reshape(b, -1, self.g * self.group.dim_rep(self.rep), 25, h, w)
        out = torch.sum(W * x, dim=3)

        return self.conv3(out.reshape(b, -1, h, w))

    def make_conv2(self):
        self.trans = self.group.trans(self.order).reshape(-1, 25)
        base_ = True
        if (isinstance(self.rep, tuple)):
            for i in self.rep:
                out_e = kronecker(self.group.rep[i[0]].rep_e,
                                  torch.inverse(self.group.diff_rep.e(self.order)).transpose(0, 1))
                if (self.group.flip):
                    out_m = kronecker(self.group.rep[i[0]].rep_m,
                                      torch.inverse(self.group.diff_rep.m(self.order)).transpose(0, 1))
                    base = d_lin_base(self.group.rep['regular'].rep_e, out_e, self.group.rep['regular'].rep_m, out_m)
                else:
                    base = c_lin_base(self.group.rep['regular'].rep_e, out_e)
                n, p, q = base.size()
                base = torch.cat(i[1] * [base], dim=1)
                base = torch.einsum('nklq, lp->nkpq', base.reshape(n, -1, (self.order + 1) * (self.order + 2) // 2, q),
                                    self.trans).reshape(n, -1)
                base = ortho_basis(base).reshape(n, -1, q, 1, 1)
                if (base_ is True):
                    base_ = base
                else:
                    base_ = cat_lin_base(base_, base, dim=0)
            return conv(base_, self.num_mid, self.g, groups=self.s)

        else:

            for i in range(self.order + 1):

                if (self.group.flip):
                    out_e = kronecker(self.group.rep[self.rep].rep_e,
                                      torch.inverse(self.group.diff_rep[i][0]).transpose(0, 1))
                    out_m = kronecker(self.group.rep[self.rep].rep_m,
                                      torch.inverse(self.group.diff_rep[i][1]).transpose(0, 1))
                    base = d_lin_base(self.group.rep[self.rep].rep_e, out_e, self.group.rep[self.rep].rep_m, out_m)
                else:
                    out_e = kronecker(self.group.rep[self.rep].rep_e,
                                      torch.inverse(self.group.diff_rep[i]).transpose(0, 1))
                    base = c_lin_base(self.group.rep[self.rep].rep_e, out_e)
                if (base_ is True):
                    base_ = base.reshape(base.size(0), -1, i + 1, base.size(2), 1)
                else:
                    base_ = cat_lin_base(base_, base.reshape(base.size(0), -1, i + 1, base.size(2), 1), dim=1)
            base = base_.squeeze(-1)
            n, p, _, q = base.size()
            base = torch.einsum('nklq, lp->nkpq', base.reshape(n, -1, (self.order + 1) * (self.order + 2) // 2, q),
                                self.trans).reshape(n, -1)
            base = ortho_basis(base).reshape(n, -1, q, 1, 1)
        return conv(base, self.num_mid, self.g, groups=self.s)



class FlipRestrict(nn.Module):
    def __init__(self, n, num_in_rep):
        super(FlipRestrict, self).__init__()
        a = torch.zeros(2 * n, 2 * n)
        a[0:n, 0:n] = torch.eye(n)
        for i in range(n):
            a[n + i, 2 * n - i - 1] = 1.
        self.param = torch.nn.Parameter(a, False)
        self.num_in_rep = num_in_rep
        self.dim = n * 2

    def forward(self, x):
        x = torch.einsum('ij,bkjmn->bkimn', self.param,
                         x.reshape(x.size(0), self.num_in_rep, self.dim, x.size(2), x.size(3))) \
            .reshape(x.shape)
        return x


class Group:
    def __init__(self, n, flip=False, dis='fd'):
        '''
            n: n means the group has n pure rotational element.
            flip: wether the group include flip element
        '''
        self.dif_ker = { \
            '0': torch.tensor([[[1.]]]), \
            '1': torch.tensor(
                [[[0., 0., 0.], [-0.5, 0., 0.5], [0., 0., 0.]], [[0., 0.5, 0.], [0., 0., 0.], [0., -0.5, 0.]]]), \
            '2': torch.tensor(
                [[[0., 0., 0.], [1., -2., 1.], [0., 0., 0.]], [[-0.25, 0, 0.25], [0., 0., 0.], [0.25, 0., -0.25]],
                 [[0., 1., 0.], [0., -2., 0.], [0., 1., 0.]]]), \
            '3': torch.tensor([[[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [-0.5, 1., 0., -1., 0.5],
                                [0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.]], \
                               [[0., 0., 0., 0., 0.], [0., 0.5, -1., 0.5, 0.], [0., 0., 0., 0., 0.],
                                [0., -0.5, 1., -0.5, 0.], [0., 0., 0., 0., 0.]], \
                               [[0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.], [0., 1., 0., -1., 0.],
                                [0., -0.5, 0., 0.5, 0.], [0., 0., 0., 0., 0.]], \
                               [[0., 0., 0.5, 0., 0.], [0., 0., -1., 0., 0.], [0., 0., 0., 0., 0.],
                                [0., 0., 1., 0., 0.], [0., 0., -0.5, 0., 0.]]]), \
            '4': torch.tensor([ \
                [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [1., -4., 6., -4., 1.], [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]], \
                [[0., 0., 0., 0., 0.], [-0.25, 0.5, 0., -0.5, 0.25], [0., 0., 0., 0., 0.], [0.25, -0.5, 0., 0.5, -0.25],
                 [0., 0., 0., 0., 0.]], \
                [[0., 0., 0., 0., 0.], [0., 1., -2., 1., 0.], [0., -2., 4., -2., 0.], [0., 1., -2., 1., 0.],
                 [0., 0., 0., 0., 0.]], \
                [[0., -0.25, 0., 0.25, 0.], [0., 0.5, 0., -0.5, 0.], [0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.],
                 [0., 0.25, 0., -0.25, 0.]], \
                [[0., 0., 1., 0., 0.], [0., 0., -4., 0., 0.], [0., 0., 6., 0., 0.], [0., 0., -4., 0., 0.],
                 [0., 0., 1., 0., 0.]]]), \
            '5': torch.tensor([ \
                [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]]]), \
            '6': torch.tensor([ \
                [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., -0.5, 0., 0.5, 0.], [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]], \
                [[0., 0., 0., 0., 0.], [0., 0., 0.5, 0., 0.], [0., 0., 0., 0., 0.], [0., 0., -0.5, 0., 0.],
                 [0., 0., 0., 0., 0.]]]), \
            '7': torch.tensor([ \
                [[0., 0., 0., 0., 0.], [0., 0., 0., 0., 0.], [0., 1., -2., 1., 0.], [0., 0., 0., 0., 0.],
                 [0., 0., 0., 0., 0.]], \
                [[0., 0., 0., 0., 0.], [0., -0.25, 0, 0.25, 0.], [0., 0., 0., 0., 0.], [0., 0.25, 0., -0.25, 0.],
                 [0., 0., 0., 0., 0.]], \
                [[0., 0., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., -2., 0., 0.], [0., 0., 1., 0., 0.],
                 [0., 0., 0., 0., 0.]]])}
        self.filter = {'5': torch.tensor([[1.]]), \
                       '6': torch.tensor([[1., 0.], [0., 1.]])}
        self.filter['7'] = torch.eye(3).to(torch.float)
        self.filter['3'] = torch.eye(4).to(torch.float)
        self.filter['3'][0, 0] = 0.
        self.filter['3'][3, 3] = 0.
        self.filter['4'] = torch.zeros(5, 5).to(torch.float)
        self.filter['4'][2, 2] = 1
        self.dis = dis
        self.discritization(self.dis)
        self.n = n
        self.flip = flip
        self.dim = n
        if (flip == True):
            self.dim = self.dim * 2
        self.diff_rep = diff_rep(n, flip)
        self.rep = {'regular': None, 'trivial': trivial()}
        if (flip == True):
            self.rep['regular'] = d_regular(n)
        else:
            self.rep['regular'] = c_regular(n)
            for i in range(1, n):
                j = 2 ** i
                if (j >= n):
                    break
                self.rep['quo_' + str(j)] = c_qotient(n, j)

        self.bases = []
        for i in range(8):
            self.bases.append({})
        self.base3 = []
        for i in range(8):
            self.base3.append({('regular', 'regular'): None, ('regular', 'trivial'): None, ('trivial', 'trivial'): None,
                               ('trivial', 'regular'): None})
        self.fast_base = {}
        self.lin_base = {'ok': None}
        self.base_3x3 = {('regular', 'regular'): None, ('regular', 'trivial'): None, ('trivial', 'trivial'): None,
                         ('trivial', 'regular'): None}

    def discritization(self, dis):
        if (dis == 'fd'):
            return 0
        elif (dis == 'gauss'):
            for i in range(5):
                if (i < 3):
                    n = i + 5
                else:
                    n = i
                self.dif_ker[str(n)] = make_gauss(i, 5)
        else:
            for i in range(5):
                if (i < 3):
                    n = i + 5
                else:
                    n = i
                self.dif_ker[str(n)] = make_rbffd(i, 5)

    def coef(self, in_rep, out_rep, df):
        d = kronecker(df.transpose(0, 1), in_rep.transpose(0, 1))
        n1 = d.size(0)
        n2 = out_rep.size(0)
        return kronecker(out_rep, torch.eye(n1)) - kronecker(torch.eye(n2), d)

    def base(self, order, in_rep, out_rep):
        '''
            order: an integer representing the order of differential operator.
            in_rep: a string indicate the representation type of input feature
            out_rep: a string indicate the representation type of output feature
            return the bases of the interwiners. of shape (bases x dim_out_rep x dim_in_rep x kernel_size x kernel_size)
        '''
        if ((in_rep, out_rep) in self.bases[order]):
            return self.bases[order][(in_rep, out_rep)]
        in_rep_ = self.rep[in_rep]
        out_rep_ = self.rep[out_rep]
        if (self.flip == True):
            df1, df2 = self.diff_rep[order % 5]
            w1 = self.coef(in_rep_.rep_e, out_rep_.rep_e, df1)
            w2 = self.coef(in_rep_.rep_m, out_rep_.rep_m, df2)
            w = torch.cat((w1, w2))
        else:
            df = self.diff_rep[order % 5]
            w = self.coef(in_rep_.rep_e, out_rep_.rep_e, df)
        w = solve(w).transpose(0, 1)
        dim_in_rep = in_rep_.dim
        dim_out_rep = out_rep_.dim
        n = w.size(0)
        w = w.reshape(n, dim_out_rep, order % 5 + 1, dim_in_rep, ).transpose(2, 3)
        w = w.to(torch.float32)
        print("shape of w:{}".format(w.shape))
        print("shape of difkernel:{}".format(self.dif_ker[str(order)].shape))
        self.bases[order][(in_rep, out_rep)] = torch.einsum('ijkl,lmn->ijkmn', w, self.dif_ker[str(order)])
        shape = self.bases[order][(in_rep, out_rep)].shape
        if (torch.sum(self.bases[order][(in_rep, out_rep)] ** 2) > 0):
            b = self.bases[order][(in_rep, out_rep)].reshape(shape[0], -1)
            b = b / torch.norm(b, dim=1, keepdim=True)
            self.bases[order][(in_rep, out_rep)] = b.reshape(shape)
        return self.bases[order][(in_rep, out_rep)]

    def fast_base_(self, in_rep, out_rep, order):
        orderlist = range(order + 1)
        if ((in_rep, out_rep) not in self.fast_base):
            base = []
            for i in orderlist:
                if (i < 3):
                    i = i + 5

                base.append(self.base(i, in_rep, out_rep))
            base = torch.cat(base)
            shape = base.shape
            self.fast_base[(in_rep, out_rep)] = base
        else:
            base = self.fast_base[(in_rep, out_rep)]
        return base

    def conv5x5(self, in_type, out_type, order=4, stride=1, groups=1):
        '''
            in_type: a list indicate the type of input feature, of the form [in_rep, dim_in]
            out_type: a list indicate the type of output feature, of the form [out_rep, dim_out]
            return: a 5x5 equivariant conv layer (conbination of all 0-4 order differential operator)
        '''
        orderlist = range(order + 1)
        in_rep, num_in = in_type
        out_rep, num_out = out_type
        if ((in_rep, out_rep) in self.fast_base):
            base = self.fast_base[(in_rep, out_rep)]
        else:
            if (isinstance(in_rep, tuple) or isinstance(out_rep, tuple)):
                if (isinstance(in_rep, tuple) is False):
                    in_rep = ((in_rep, 1),)
                if (isinstance(out_rep, tuple) is False):
                    out_rep = ((out_rep, 1),)
                in_re = ()
                out_re = ()
                for i in in_rep:
                    in_re = in_re + i[1] * (i[0],)
                for i in out_rep:
                    out_re = out_re + i[1] * (i[0],)
                base = True
                for j in out_re:
                    b = True
                    for i in in_re:
                        if (b is True):
                            b = self.fast_base_(i, j, order)
                        else:
                            b = cat_lin_base(b, self.fast_base_(i, j, order), dim=1)
                    if (base is True):
                        base = b
                    else:
                        base = cat_lin_base(base, b, dim=0)
                self.fast_base[(in_rep, out_rep)] = base
            else:
                base = self.fast_base_(in_rep, out_rep, order)

        return conv(base, num_in, num_out, groups, stride)

    def lin_base_(self, in_rep, out_rep):
        if ((in_rep, out_rep) not in self.lin_base):
            if (self.flip):
                base = d_lin_base(self.rep[in_rep].rep_e, self.rep[out_rep].rep_e, self.rep[in_rep].rep_m,
                                  self.rep[out_rep].rep_m)
            else:
                base = c_lin_base(self.rep[in_rep].rep_e, self.rep[out_rep].rep_e)
            base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
            self.lin_base[(in_rep, out_rep)] = base
        return self.lin_base[(in_rep, out_rep)]

    def conv1x1(self, in_type, out_type, stride=1, groups=1):
        in_rep, num_in = in_type
        out_rep, num_out = out_type
        if ((in_rep, out_rep) in self.lin_base):
            return conv(self.lin_base[(in_rep, out_rep)], num_in, num_out, groups, stride)
        if (isinstance(in_rep, tuple) or isinstance(out_rep, tuple)):
            if (isinstance(in_rep, tuple) is False):
                in_rep = ((in_rep, 1),)
            if (isinstance(out_rep, tuple) is False):
                out_rep = ((out_rep, 1),)
            in_re = ()
            out_re = ()
            for i in in_rep:
                in_re = in_re + i[1] * (i[0],)
            for i in out_rep:
                out_re = out_re + i[1] * (i[0],)
            a = True
            for j in out_re:
                b = True
                for i in in_re:
                    if (b is True):
                        b = self.lin_base_(i, j)
                    else:
                        b = cat_lin_base(b, self.lin_base_(i, j), dim=1)
                if (a is True):
                    a = b
                else:
                    a = cat_lin_base(a, b, dim=0)
            self.lin_base[(in_rep, out_rep)] = a
            return conv(a, num_in, num_out, groups, stride)

        else:
            base = self.lin_base_(in_rep, out_rep)
            return conv(self.lin_base[(in_rep, out_rep)], num_in, num_out, groups, stride)

    def conv1x1_c(self, num_in, in_rep, num_out, out_rep, stride=1, groups=1):

        base = c_lin_base(in_rep, out_rep)
        base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return conv(base, num_in, num_out, groups, stride)

    def conv1x1_d(self, num_in, in_rep_e, in_rep_m, num_out, out_rep_e, out_rep_m, stride=1, groups=1):

        base = d_lin_base(in_rep_e, out_rep_e, in_rep_m, out_rep_m)
        base = base.unsqueeze(dim=-1).unsqueeze(dim=-1)
        return conv(base, num_in, num_out, groups, stride)

    def flip_restrict(self, in_type):
        in_rep, num_in = in_type
        return FlipRestrict(self.n, num_in)

    def dim_rep(self, rep):
        if (isinstance(rep, tuple)):
            dim = 0
            for i in rep:
                dim += i[1] * self.rep[i[0]].dim
            return dim
        return self.rep[rep].dim

    def trans(self, order):
        s = []
        for i in range(order + 1):
            if (i < 3):
                i = i + 5
            s.append(self.dif_ker[str(i)])
        return torch.cat(s)

    def nlpdo(self, in_type, out_type, order, reduction, s, g, stride=1):
        return nlpdo_torch(self, in_type, out_type, order, reduction, s, g, stride)


    def norm(self, in_type, affine=True, momentum=0.1, track_running_stats=True):
        '''
            num_rep: number of representation in the input
            rep: a string indicate the type of representation
            other argument is the same with the standard BatchNorm
        '''
        rep, num_rep = in_type
        dim_rep = self.dim_rep(rep)
        return GroupBatchNorm(num_rep, dim_rep, affine, momentum, track_running_stats)

    def GroupPool(self, in_type, type='max'):
        rep, num_rep = in_type
        return GroupPooling(self.dim_rep(rep), num_rep, type)

    def MaxPool(self, in_type, kernel_size=5):
        rep, num_rep = in_type
        C = self.dim_rep(rep) * num_rep
        return nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=1),
                             Downsample(channels=C, filt_size=kernel_size, stride=2))

    def AvgPool(self, in_type, kernel_size=5):
        rep, num_rep = in_type
        C = self.dim_rep(rep) * num_rep
        return nn.Sequential(nn.AvgPool2d(kernel_size=2, stride=1),
                             Downsample(channels=C, filt_size=kernel_size, stride=2))


def rot90(x, rep):
    b, c, h, w = x.shape
    x = torch.rot90(x, 3, [2, 3])
    x = torch.einsum('bcdhw, ad->bcahw', x.reshape(b, c // rep.size(0), rep.size(0), h, w), rep).reshape(b, c, h, w)
    return x


def reflect(x, rep):
    b, c, h, w = x.shape
    x = x.transpose(2, 3)
    x = torch.rot90(x, 3, [2, 3])
    x = torch.einsum('bcdhw, ad->bcahw', x.reshape(b, c // rep.size(0), rep.size(0), h, w), rep).reshape(b, c, h, w)
    return x


def test():
    x = torch.randn(10, 16 * 6, 20, 20).cuda()
    g = Group(8, True, 'gauss')
    net = g.conv5x5(('regular', 6), ('regular', 6)).cuda()
    rep = g.rep['regular'].rep_e.cuda()
    rep = torch.matmul(rep, rep)
    y_ = net(rot90(x, rep))
    y = rot90(net(x), rep)
    print(torch.sum((y - y_) ** 2))
    print(torch.sum(y ** 2) / (10 * 32 * 20 * 20))


def compute_param(net):
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

