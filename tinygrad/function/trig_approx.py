from collections.abc import Sequence
from enum import Enum
import math
from typing import Tuple

from tinygrad.dtype import dtypes
from tinygrad.multi import MultiLazyBuffer
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer


class Precision(Enum):
  SINGLE = 0
  DOUBLE = 1


init_jk = {Precision.SINGLE: 3, Precision.DOUBLE: 4}

ipio2_bits = [
  0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
  0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
  0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
  0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
  0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
  0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
  0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
  0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
  0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
  0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
  0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
]
two24 = float.fromhex('0x1.0000000000000p+24')
twon24 = float.fromhex('0x1.0000000000000p-24')


def __kernel_rem_pio2(xs: list[LazyBuffer], e0: LazyBuffer, prec: Precision) -> Tuple[LazyBuffer, list[LazyBuffer]]:
  return e0.const(0), [xs[0].const(float('nan'))]


def __ieee754_rem_pio2f(x: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  invpio2 = x.const(float.fromhex('0x1.45f306dc9c883p-1'))
  pio2_1 = x.const(float.fromhex('0x1.921fb50000000p+0'))
  pio2_1t = x.const(float.fromhex('0x1.110b4611a6263p-26'))

  hx = x.cast(dtypes.int32, bitcast=True)
  ix = hx.e(BinaryOps.MOD, hx.const(1 << 31))

  e0 = ix.e(BinaryOps.DIV, ix.const(1 << 23)).e(BinaryOps.SUB, ix.const(150))
  z = ix.e(BinaryOps.SUB, e0.e(BinaryOps.MUL, e0.const(1 << 23))).cast(dtypes.float32, bitcast=True)
  n, ty = __kernel_rem_pio2([z], e0, Precision.SINGLE)
  hx_lt_0 = hx.e(BinaryOps.CMPLT, hx.const(0))
  n = hx_lt_0.e(TernaryOps.WHERE, n.e(UnaryOps.NEG), n)
  y = hx_lt_0.e(TernaryOps.WHERE, ty[0].e(UnaryOps.NEG), ty[0])

  x_is_inf_or_nan = ix.const(0x7f800000 - 1).e(BinaryOps.CMPLT, ix)
  n = x_is_inf_or_nan.e(TernaryOps.WHERE, n.const(0), n)
  y = x_is_inf_or_nan.e(TernaryOps.WHERE, x.e(BinaryOps.SUB, x), y)

  # |x| ~< 2^28*(pi/2), medium size
  x_is_medium = ix.e(BinaryOps.CMPLT, ix.const(0x4dc90fdb))
  fn = x.e(BinaryOps.MUL, invpio2)
  r = x.e(BinaryOps.SUB, fn.e(BinaryOps.MUL, pio2_1))
  w = fn.e(BinaryOps.MUL, pio2_1t)
  y = x_is_medium.e(TernaryOps.WHERE, r.e(BinaryOps.SUB, w), y)
  n = x_is_medium.e(TernaryOps.WHERE, fn.cast(dtypes.int32), n)

  return n, y


def __kernel_sindf(x: LazyBuffer) -> LazyBuffer:
  s1 = x.const(float.fromhex('-0x15555554cbac77.0p-55'))
  s2 = x.const(float.fromhex('0x111110896efbb2.0p-59'))
  s3 = x.const(float.fromhex('-0x1a00f9e2cae774.0p-65'))
  s4 = x.const(float.fromhex('0x16cd878c3b46a7.0p-71'))

  z = x.e(BinaryOps.MUL, x)
  w = z.e(BinaryOps.MUL, z)
  r = z.e(BinaryOps.MUL, s4).e(BinaryOps.ADD, s3)
  s = z.e(BinaryOps.MUL, x)
  out = z.e(BinaryOps.MUL, s2).e(BinaryOps.ADD, s1).e(BinaryOps.MUL, s).e(BinaryOps.ADD, x)
  out = out.e(BinaryOps.ADD, s.e(BinaryOps.MUL, w.e(BinaryOps.MUL, r)))

  return out


def __kernel_cosdf(x: LazyBuffer) -> LazyBuffer:
  one = x.const(1)
  c0 = x.const(float.fromhex('-0x1ffffffd0c5e81.0p-54'))
  c1 = x.const(float.fromhex('0x155553e1053a42.0p-57'))
  c2 = x.const(float.fromhex('-0x16c087e80f1e27.0p-62'))
  c3 = x.const(float.fromhex('0x199342e0ee5069.0p-68'))

  z = x.e(BinaryOps.MUL, x)
  w = z.e(BinaryOps.MUL, z)
  r = z.e(BinaryOps.MUL, c3).e(BinaryOps.ADD, c2)
  out = z.e(BinaryOps.MUL, c0).e(BinaryOps.ADD, one)
  out = w.e(BinaryOps.MUL, c1).e(BinaryOps.ADD, out)
  out = out.e(BinaryOps.ADD, w.e(BinaryOps.MUL, z).e(BinaryOps.MUL, r))

  return out


def sin(x: LazyBuffer) -> LazyBuffer:
  if x.dtype == dtypes.float32:
    s1pio2 = x.const(math.pi / 2)
    s2pio2 = x.const(math.pi)
    s3pio2 = x.const(3 * math.pi / 2)
    s4pio2 = x.const(2 * math.pi)

    hx = x.cast(dtypes.int32, bitcast=True)
    ix = hx.e(BinaryOps.MOD, hx.const(1 << 31))

    # Base case: needs general argument reduction
    n, y = __ieee754_rem_pio2f(x)
    n_mod_4 = n.e(BinaryOps.MOD, n.const(4))
    out = n_mod_4.e(BinaryOps.CMPEQ, n.const(2)).e(TernaryOps.WHERE, __kernel_sindf(y.e(UnaryOps.NEG)),
                                                   __kernel_cosdf(y).e(UnaryOps.NEG))
    out = n_mod_4.e(BinaryOps.CMPEQ, n.const(1)).e(TernaryOps.WHERE, __kernel_cosdf(y), out)
    out = n_mod_4.e(BinaryOps.CMPEQ, n.const(0)).e(TernaryOps.WHERE, __kernel_sindf(y), out)

    # Is Inf or NaN?
    out = ix.const(0x7f800000 + 1).e(BinaryOps.CMPLT, ix).e(TernaryOps.WHERE, x.e(BinaryOps.SUB, x), out)

    # |x| ~<= 9pi/4
    hx_gt_0 = hx.const(0).e(BinaryOps.CMPLT, hx)
    out_9pio4_offset = hx_gt_0.e(TernaryOps.WHERE, s4pio2.e(UnaryOps.NEG), s4pio2)
    out_9pio4 = __kernel_sindf(x.e(BinaryOps.ADD, out_9pio4_offset))
    x_lt_9pio4 = ix.const(0x40e231d5 + 1).e(BinaryOps.CMPLT, ix)
    out = x_lt_9pio4.e(TernaryOps.WHERE, out_9pio4, out)

    # |x| ~<= 7pi/4
    cos_x_minus_s1pio2 = __kernel_cosdf(x.e(BinaryOps.SUB, s3pio2)).e(UnaryOps.NEG)
    neg_cos_x_plus_s1pio2 = __kernel_cosdf(x.e(BinaryOps.ADD, s3pio2))
    out_7pio4 = hx_gt_0.e(TernaryOps.WHERE, cos_x_minus_s1pio2, neg_cos_x_plus_s1pio2)
    x_lt_7pio4 = ix.e(BinaryOps.CMPLT, ix.const(0x40afeddf + 1))
    out = x_lt_7pio4.e(TernaryOps.WHERE, out_7pio4, out)

    # |x| ~<= 5pi/4
    out_5pio4_offset = hx_gt_0.e(TernaryOps.WHERE, s2pio2, s2pio2.e(UnaryOps.NEG))
    out_5pio4 = __kernel_sindf(out_5pio4_offset.e(BinaryOps.SUB, x))
    x_lt_5pio4 = ix.const(0x407b53d1 + 1).e(BinaryOps.CMPLT, ix)
    out = x_lt_5pio4.e(TernaryOps.WHERE, out_5pio4, out)

    # |x| ~<= 3pi/4
    cos_x_minus_s1pio2 = __kernel_cosdf(x.e(BinaryOps.SUB, s1pio2))
    neg_cos_x_plus_s1pio2 = __kernel_cosdf(x.e(BinaryOps.ADD, s1pio2)).e(UnaryOps.NEG)
    out_3pio4 = hx_gt_0.e(TernaryOps.WHERE, cos_x_minus_s1pio2, neg_cos_x_plus_s1pio2)
    x_lt_3pio4 = ix.e(BinaryOps.CMPLT, ix.const(0x40afeddf + 1))
    out = x_lt_3pio4.e(TernaryOps.WHERE, out_3pio4, out)

    # |x| ~<= pi/4
    out_pio4 = __kernel_sindf(x)
    x_lt_pio4 = ix.e(BinaryOps.CMPLT, ix.const(0x3f490fda + 1))
    out = x_lt_pio4.e(TernaryOps.WHERE, out_pio4, out)

    # |x| < 2^-12
    out_tiny = x
    x_is_tiny_1 = ix.e(BinaryOps.CMPLT, ix.const(0x39800000))
    x_is_tiny_2 = (int_x := x.cast(dtypes.int32)).e(BinaryOps.CMPEQ, int_x.const(0))
    out = x_is_tiny_1.e(TernaryOps.WHERE, x_is_tiny_2.e(TernaryOps.WHERE, out_tiny, out), out)

    return out

  if x.dtype in [dtypes.bfloat16, dtypes.float16]:
    return sin(x.cast(dtypes.float32)).cast(x.dtype)

  raise NotImplementedError(f"No `sin` for type {x.dtype}")


def cos(x: LazyBuffer) -> LazyBuffer:
  return sin(x.const(math.pi / 2).e(BinaryOps.SUB, x))


class Sin(Function):
  x: LazyBuffer

  def forward(self, x: LazyBuffer) -> LazyBuffer:
    self.x = x
    return sin(x)

  def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return cos(self.x).e(BinaryOps.MUL, grad_output)
