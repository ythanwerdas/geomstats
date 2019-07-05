"""Pytorch based computation backend."""

import numpy as np
import torch
import warnings
warnings.simplefilter('ignore', category=ImportWarning)

double = 'torch.DoubleTensor'
float16 = 'torch.Float'
float32 = 'torch.FloatTensor'
float64 = 'torch.DoubleTensor'
int32 = 'torch.LongTensor'
int8 = 'torch.ByteTensor'

CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if CUDA else 'cpu')


def tril_indices(m, n=None):
    if n is None:
        n = m
    return torch.tril_indices((m, n))


def to_bool(x):
    return cast(x, int8)


def real(x):
    # Complex tensors not supported in pytorch
    return x


def cond(pred, true_fn, false_fn):
    if pred:
        return true_fn()
    return false_fn()


def amax(x):
    return torch.max(x)


def boolean_mask(x, mask):
    return x[mask.byte()]


def arctan2(*args, **kwargs):
    return torch.arctan2(*args, **kwargs)


def cast(x, dtype):
    x = array(x)
    return x.type(dtype)


def divide(*args, **kwargs):
    return torch.div(*args, **kwargs)


def repeat(*args, **kwargs):
    return torch.repeat(*args, **kwargs)


def asarray(x):
    return np.asarray(x)


def concatenate(seq, axis=0, out=None):
    seq = [t.float() for t in seq]
    return torch.cat(seq, dim=axis, out=out)


def identity(val):
    return torch.eye(val)


def hstack(seq):
    return concatenate(seq, axis=1)


def stack(*args, **kwargs):
    return torch.stack(*args, **kwargs)


def vstack(seq):
    return concatenate(seq)


def array(val):
    if type(val) == list:
        if type(val[0]) != torch.Tensor:
            val = np.copy(np.array(val))
        else:
            val = concatenate(val)

    if type(val) == bool:
        val = np.array(val)
    if type(val) == np.ndarray:
        if val.dtype == bool:
            val = torch.from_numpy(np.array(val, dtype=np.uint8))
        elif val.dtype == np.float32 or val.dtype == np.float64:
            val = torch.from_numpy(np.array(val, dtype=np.float32))
        else:
            val = torch.from_numpy(val)

    if type(val) != torch.Tensor:
        val = torch.Tensor([val])
    if val.dtype == torch.float64:
        val = val.float()
    return val


def abs(val):
    return torch.abs(val)


def zeros(*args):
    return torch.zeros(*args).to(DEVICE)


def ones(*args):
    return torch.ones(*args).to(DEVICE)


def ones_like(x):
    return torch.ones_like(x).to(x.device)


def empty_like(x):
    return torch.empty_like(x).to(x.device)


def all(x, axis=None):
    if axis is None:
        return x.byte().all()
    result = np.concatenate(
        [np.array([one_x.byte().all()]) for one_x in x], axis=0)
    assert result.shape == (len(x),), result.shape
    return torch.from_numpy(result.astype(int))


def allclose(a, b, **kwargs):
    a = torch.tensor(a)
    b = torch.tensor(b)
    a = a.float()
    b = b.float()
    a = to_ndarray(a, to_ndim=1)
    b = to_ndarray(b, to_ndim=1)
    n_a = a.shape[0]
    n_b = b.shape[0]
    ndim = len(a.shape)
    if n_a > n_b:
        reps = (int(n_a / n_b),) + (ndim-1) * (1,)
        b = tile(b, reps)
    elif n_a < n_b:
        reps = (int(n_b / n_a),) + (ndim-1) * (1,)
        a = tile(a, reps)
    return torch.allclose(a, b, **kwargs)


def sin(val):
    return torch.sin(val)


def cos(val):
    return torch.cos(val)


def cosh(*args, **kwargs):
    return torch.cosh(*args, **kwargs)


def arccosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1) / x)
    return c0 + c1


def sinh(*args, **kwargs):
    return torch.sinh(*args, **kwargs)


def tanh(*args, **kwargs):
    return torch.tanh(*args, **kwargs)


def arcsinh(x):
    return torch.log(x + torch.sqrt(x*x+1))


def arcosh(x):
    return torch.log(x + torch.sqrt(x*x-1))


def tan(val):
    return torch.tan(val)


def arcsin(val):
    return torch.asin(val)


def arccos(val):
    return torch.acos(val)


def shape(val):
    return val.shape


def dot(a, b):
    return torch.mm(a, b)


def maximum(a, b):
    return torch.max(array(a), array(b))


def greater(a, b):
    return torch.gt(a, b)


def greater_equal(a, b):
    return torch.ge(a, b)


def to_ndarray(x, to_ndim, axis=0):
    x = array(x)
    if x.dim() == to_ndim - 1:
        x = torch.unsqueeze(x, dim=axis)
    assert x.dim() >= to_ndim
    return x


def sqrt(val):
    return torch.sqrt(torch.tensor(val).float())


def norm(val, axis):
    return torch.linalg.norm(val, axis=axis)


def rand(*args, **largs):
    return torch.random.rand(*args, **largs)


def isclose(*args, **kwargs):
    return torch.isclose(*args, **kwargs)


def less(a, b):
    return torch.lt(a, b)


def less_equal(a, b):
    return torch.le(a, b)


def eye(*args, **kwargs):
    return torch.eye(*args, **kwargs)


def average(*args, **kwargs):
    return torch.average(*args, **kwargs)


def matmul(*args, **kwargs):
    return torch.matmul(*args, **kwargs)


def sum(x, axis=None, **kwargs):
    if axis is None:
        return torch.sum(x, **kwargs)
    return torch.sum(x, dim=axis, **kwargs)


def einsum(*args, **kwargs):
    return torch.from_numpy(np.einsum(*args, **kwargs)).float()


def T(x):
    return torch.t(x)


def transpose(x, axes=None):
    if axes:
        return x.permute(axes)
    if len(shape(x)) == 1:
        return x
    return x.t()


def squeeze(x, axis=None):
    return torch.squeeze(x, dim=axis)


def zeros_like(x):
    return torch.zeros_like(x).to(x.device)


def trace(x, axis1=0, axis2=1):
    if axis1 == 1 and axis2 == 2:
        trace = torch.zeros(x.shape[0]).to(x.device)
        for i, one_x in enumerate(x):
            trace[i] = torch.trace(one_x)
        return trace
    else:
        return torch.trace(x)


def mod(*args, **kwargs):
    return torch.fmod(*args, **kwargs)


def linspace(start, stop, num):
    return torch.linspace(start=start, end=stop, steps=num)


def equal(a, b, **kwargs):
    if a.dtype == torch.ByteTensor:
        a = cast(a, torch.uint8).float()
    if b.dtype == torch.ByteTensor:
        b = cast(b, torch.uint8).float()
    return torch.equal(a, b, **kwargs)


def floor(*args, **kwargs):
    return torch.floor(*args, **kwargs)


def cross(x, y):
    return torch.from_numpy(np.cross(x, y))


def triu_indices(*args, **kwargs):
    return torch.triu_indices(*args, **kwargs)


def where(*args, **kwargs):
    return torch.where(*args, **kwargs)


def tile(x, y):
    y = [int(one_y) for one_y in y]
    return x.repeat(y)


def clip(x, amin, amax):
    return np.clip(x, amin, amax)


def diag(*args, **kwargs):
    return torch.diag(*args, **kwargs)


def any(x):
    return x.byte().any()


def expand_dims(x, axis=0):
    return torch.unsqueeze(x, dim=axis)


def outer(*args, **kwargs):
    return torch.ger(*args, **kwargs)


def hsplit(*args, **kwargs):
    return torch.hsplit(*args, **kwargs)


def argmax(*args, **kwargs):
    return torch.argmax(*args, **kwargs)


def diagonal(*args, **kwargs):
    return torch.diagonal(*args, **kwargs)


def exp(*args, **kwargs):
    return torch.exp(*args, **kwargs)
    return torch.cov(*args, **kwargs)


def eval(x):
    return x


def ndim(x):
    return x.dim()


def gt(*args, **kwargs):
    return torch.gt(*args, **kwargs)


def eq(*args, **kwargs):
    return torch.eq(*args, **kwargs)


def nonzero(*args, **kwargs):
    return torch.nonzero(*args, **kwargs)


def copy(x):
    return x.clone()


def seed(x):
    torch.manual_seed(x)


def sign(*args, **kwargs):
    return torch.sign(*args, **kwargs)


def mean(x, axis=None):
    if axis is None:
        return torch.mean(x)
    else:
        return np.mean(x, axis)
