from enum import Enum
import math
from sys import float_info
from typing import Tuple

from tinygrad.dtype import dtypes
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.tensor import Function
from tinygrad.lazy import LazyBuffer


class Precision(Enum):
  SINGLE = 0
  DOUBLE = 1


def rem_pio2_large(tx: LazyBuffer, e0: LazyBuffer, param: Precision) -> Tuple[LazyBuffer, LazyBuffer]:
  # rem_pio2_large :: float64 -> int32 -> forall (p :: Precision) -> (int32, float64)
  return e0.const(0), tx.const(float('nan'))


def rem_pio2f(x: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer]:
  # rem_pio2f :: float32 -> (int32, float64)
  x64 = x.cast(dtypes.float64)
  TOINT = x64.const(1.5 / float_info.epsilon)
  INV_PIO2 = x64.const(float.fromhex('0x1.45f306dc9c883p-1'))  # 53 bits of 2/pi
  PIO2_1 = x64.const(float.fromhex('0x1.921fb5p0'))  # first 25 bits of pi/2
  PIO2_1T = x64.const(float.fromhex('0x1.110b4611a6263p-26'))  # pi/2 - pio2_1

  ix = x.cast(dtypes.uint32, bitcast=True)
  sign = ix.e(BinaryOps.DIV, ix.const(2 ** 31)).e(BinaryOps.CMPEQ, ix.const(0))
  sign = sign.e(BinaryOps.CMPEQ, sign.const(False))
  ix = ix.e(BinaryOps.MOD, ix.const(2 ** 31))

  # base case:
  e0 = ix.e(BinaryOps.DIV, ix.const(2 ** 23)).e(BinaryOps.SUB, ix.const(0x7f + 23))  # e0 = ilogb(|x|)-23, positive
  e0 = e0.cast(dtypes.int32)
  tx = ix.e(BinaryOps.SUB, e0.e(BinaryOps.MUL, e0.const(2 ** 23)).cast(dtypes.uint32))
  tx = tx.cast(dtypes.float32, bitcast=True).cast(dtypes.float64)
  tx = tx.reshape((1, *tx.shape))
  n, ty = rem_pio2_large(tx, e0, Precision.SINGLE)
  n_2 = sign.e(TernaryOps.WHERE, n.e(UnaryOps.NEG), n)
  ty = ty.reshape(ty.shape[1:])  # this is safe because 'ty' will always have a shape like (1, ...)
  y_2 = sign.e(TernaryOps.WHERE, ty.e(UnaryOps.NEG), ty)

  # x is inf or NaN
  n_inf_or_nan = ix.const(0).cast(dtypes.int32)
  y_inf_or_nan = x64.e(BinaryOps.SUB, x64)
  x_is_inf_or_nan = ix.const(0x7f800000 - 1).e(BinaryOps.CMPLT, ix)
  n_1 = x_is_inf_or_nan.e(TernaryOps.WHERE, n_inf_or_nan, n_2)
  y_1 = x_is_inf_or_nan.e(TernaryOps.WHERE, y_inf_or_nan, y_2)

  # x is "medium-sized"
  tmp = x64.e(BinaryOps.MUL, INV_PIO2).e(BinaryOps.ADD, TOINT)
  f_n = tmp.e(BinaryOps.SUB, TOINT)
  n_medium = f_n.cast(dtypes.int32)
  y_medium = x64.e(BinaryOps.SUB, f_n.e(BinaryOps.MUL, PIO2_1)).e(BinaryOps.SUB, f_n.e(BinaryOps.MUL, PIO2_1T))
  x_is_medium = ix.e(BinaryOps.CMPLT, ix.const(0x4dc90fdb))  # |x| ~< 2^28*(pi/2), medium size
  n = x_is_medium.e(TernaryOps.WHERE, n_medium, n_1)
  y = x_is_medium.e(TernaryOps.WHERE, y_medium, y_1)

  return n, y


def k_sinf(x: LazyBuffer) -> LazyBuffer:
  # k_sinf :: float64 -> float32
  S1 = x.const(float.fromhex('-0x15555554cbac77.0p-55'))
  S2 = x.const(float.fromhex('0x111110896efbb2.0p-59'))
  S3 = x.const(float.fromhex('-0x1a00f9e2cae774.0p-65'))
  S4 = x.const(float.fromhex('0x16cd878c3b46a7.0p-71'))

  z = x.e(BinaryOps.MUL, x)
  w = z.e(BinaryOps.MUL, z)
  r = S3.e(BinaryOps.ADD, z.e(BinaryOps.MUL, S4))
  s = z.e(BinaryOps.MUL, x)

  out = S2.e(BinaryOps.MUL, z).e(BinaryOps.ADD, S1)
  out = out.e(BinaryOps.MUL, s).e(BinaryOps.ADD, x)
  out = out.e(BinaryOps.ADD, s.e(BinaryOps.MUL, w).e(BinaryOps.MUL, r))

  return out.cast(dtypes.float32)


def k_cosf(x: LazyBuffer) -> LazyBuffer:
  # k_cosf :: float64 -> float32
  C0 = x.const(float.fromhex('-0x1ffffffd0c5e81.0p-54'))
  C1 = x.const(float.fromhex('0x155553e1053a42.0p-57'))
  C2 = x.const(float.fromhex('-0x16c087e80f1e27.0p-62'))
  C3 = x.const(float.fromhex('0x199342e0ee5069.0p-68'))
  one = x.const(1.0)

  z = x.e(BinaryOps.MUL, x)
  w = z.e(BinaryOps.MUL, z)
  r = C2.e(BinaryOps.ADD, z.e(BinaryOps.MUL, C3))

  out = one.e(BinaryOps.ADD, z.e(BinaryOps.MUL, C0))
  out = out.e(BinaryOps.ADD, w.e(BinaryOps.MUL, C1))
  out = out.e(BinaryOps.ADD, w.e(BinaryOps.MUL, z).e(BinaryOps.MUL, r))

  return out.cast(dtypes.float32)


def sin(x: LazyBuffer) -> LazyBuffer:
  # sin :: a -> a
  if x.dtype == dtypes.float32:
    x64 = x.cast(dtypes.float64)

    # Small multiples of pi/2 rounded to double precision
    S1_PIO2 = x64.const(float.fromhex('0x1.921fb54442d18p0'))  # pi/2
    S2_PIO2 = x64.const(float.fromhex('0x1.921fb54442d18p1'))  # pi
    S3_PIO2 = x64.const(float.fromhex('0x1.2d97c7f3321d2p2'))  # 3pi/2
    S4_PIO2 = x64.const(float.fromhex('0x1.921fb54442d18p2'))  # 2pi

    ix = x.cast(dtypes.uint32, bitcast=True)
    sign = ix.e(BinaryOps.DIV, ix.const(2 ** 31)).e(BinaryOps.CMPEQ, ix.const(0))
    sign = sign.e(BinaryOps.CMPEQ, sign.const(False))
    ix = ix.e(BinaryOps.MOD, ix.const(2 ** 31))

    (n, y) = rem_pio2f(x)
    n = n.e(BinaryOps.MOD, n.const(4))
    out_n_eq_0 = k_sinf(y)
    out_n_eq_1 = k_cosf(y)
    out_n_lt_2 = n.e(BinaryOps.CMPEQ, n.const(0)).e(TernaryOps.WHERE, out_n_eq_0, out_n_eq_1)
    out_n_eq_2 = k_sinf(y.e(UnaryOps.NEG))
    out_n_eq_3 = k_cosf(y).e(UnaryOps.NEG)
    out_n_gte_2 = n.e(BinaryOps.CMPEQ, n.const(2)).e(TernaryOps.WHERE, out_n_eq_2, out_n_eq_3)
    out_4 = n.e(BinaryOps.CMPLT, n.const(2)).e(TernaryOps.WHERE, out_n_lt_2, out_n_gte_2)

    out_inf_or_nan = x.e(BinaryOps.SUB, x)
    x_eq_inf_or_nan = ix.const(0x7f800000 - 1).e(BinaryOps.CMPLT, ix)
    out_3 = x_eq_inf_or_nan.e(TernaryOps.WHERE, out_inf_or_nan, out_4)

    out_lteq_7pi_over_4 = sign.e(TernaryOps.WHERE,
                                 k_cosf(x64.e(BinaryOps.ADD, S3_PIO2)),
                                 k_cosf(x64.e(BinaryOps.SUB, S3_PIO2).e(UnaryOps.NEG)))
    out_lteq_9pi_over_4 = k_sinf(
      sign.e(TernaryOps.WHERE, x64.e(BinaryOps.ADD, S4_PIO2), x64.e(BinaryOps.SUB, S4_PIO2))
    )
    x_lteq_7pi_over_4 = ix.e(BinaryOps.CMPLT, ix.const(0x40afeddf + 1))
    out_lteq_9pi_over_4 = x_lteq_7pi_over_4.e(TernaryOps.WHERE,
                                              out_lteq_7pi_over_4,
                                              out_lteq_9pi_over_4)
    x_lteq_9pi_over_4 = ix.e(BinaryOps.CMPLT, ix.const(0x40e231d5 + 1))
    out_2 = x_lteq_9pi_over_4.e(TernaryOps.WHERE, out_lteq_9pi_over_4, out_3)

    out_lteq_3pi_over_4 = sign.e(TernaryOps.WHERE,
                                 k_cosf(x64.e(BinaryOps.ADD, S1_PIO2)).e(UnaryOps.NEG),
                                 k_cosf(x64.e(BinaryOps.SUB, S1_PIO2)))
    out_lteq_5pi_over_4 = k_sinf(
      sign.e(TernaryOps.WHERE,
             (x64.e(BinaryOps.ADD, S2_PIO2)).e(UnaryOps.NEG),
             x64.e(BinaryOps.SUB, S2_PIO2).e(UnaryOps.NEG))
    )
    x_lt_3pi_over_4 = ix.e(BinaryOps.CMPLT, ix.const(0x4016cbe3 + 1))
    out_lteq_5pi_over_4 = x_lt_3pi_over_4.e(TernaryOps.WHERE, out_lteq_3pi_over_4, out_lteq_5pi_over_4)
    x_lt_5pi_over_4 = ix.e(BinaryOps.CMPLT, ix.const(0x407b53d1 + 1))
    out_1 = x_lt_5pi_over_4.e(TernaryOps.WHERE, out_lteq_5pi_over_4, out_2)

    out_lteq_2_to_neg_12 = x
    out_lteq_pi_over_4 = k_sinf(x64)
    x_lt_2_neg_12 = ix.e(BinaryOps.CMPLT, ix.const(0x39800000))
    out_lteq_pi_over_4 = x_lt_2_neg_12.e(TernaryOps.WHERE, out_lteq_2_to_neg_12, out_lteq_pi_over_4)
    x_lt_pi_over_4 = ix.e(BinaryOps.CMPLT, ix.const(0x3f490fda + 1))
    out = x_lt_pi_over_4.e(TernaryOps.WHERE, out_lteq_pi_over_4, out_1)

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
