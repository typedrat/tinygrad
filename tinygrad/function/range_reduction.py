from collections.abc import Sequence
from typing import Tuple
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import UnaryOps, BinaryOps, TernaryOps
from tinygrad.lazy import LazyBuffer

class FloatFormat:
  def __init__(self, buffer: LazyBuffer):
    self.FLOAT_TYPE = buffer.dtype

    if buffer.dtype == dtypes.float64:
      self.UINT_TYPE = dtypes.uint64
      buffer = buffer.cast(self.UINT_TYPE)
      self.SIGN_SCALE = buffer.const(1 << 63)
      self.EXPONENT_SCALE = buffer.const(1 << 52)
      self.EXPONENT_MASK = buffer.const(0x800)
      self.MANTISSA_IMPLICIT_ONE_AT = 53
      self.MANTISSA_CHUNK_SIZE = 24
      self.MANTISSA_MASK = buffer.const(1 << 52)
      self.MANTISSA_HIGH_SCALE = buffer.const(1 << 29)
      self.MANTISSA_MID_SCALE = buffer.const(1 << 5)
      self.MANTISSA_MID_MASK = buffer.const(1 << 24)
      self.MANTISSA_LOW_MASK = buffer.const(1 << 5)
    else:
      raise ValueError("Attempted to get the FloatFormat for unsupported type {}".format(buffer.dtype))

    self.SIGN_MASK = buffer.const(2)


def mantissa_scale_search(n: LazyBuffer, bias: int, high: int, low: int = 0) -> Tuple[LazyBuffer, LazyBuffer]:
  while (high - low) >= 1:
    pivot = (high + low) // 2
    comparison = n.e(BinaryOps.CMPLT, n.const(1 << (pivot + 1)))

    if (high - low) < 1:
      break

    if comparison.e(TernaryOps.WHERE, n.const(True), n.const(False)):  # If the condition is true
      high = pivot - 1  # Search the lower half
    else:
      low = pivot + 1  # Search the upper half

  comparison = n.e(BinaryOps.CMPLT, n.const(1 << (pivot + 1)))
  exponent = comparison.e(TernaryOps.WHERE, n.const(bias - low), n.const(bias - high))
  factor = comparison.e(TernaryOps.WHERE, n.const(1 << (bias - low)), n.const(1 << (bias - high)))
  scaled_n = n.e(BinaryOps.MUL, factor)

  return exponent, scaled_n

def disassemble_float(fmt: FloatFormat, val: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer, LazyBuffer, LazyBuffer, LazyBuffer]:
  int_val = val.cast(fmt.UINT_TYPE, bitcast=True)
  sign_bit = int_val.e(BinaryOps.DIV, fmt.SIGN_SCALE)
  sign_bit = sign_bit.e(BinaryOps.MOD, fmt.SIGN_MASK)
  exponent = int_val.e(BinaryOps.DIV, fmt.EXPONENT_SCALE)
  exponent = exponent.e(BinaryOps.MOD, fmt.EXPONENT_MASK)
  mantissa_full = int_val.e(BinaryOps.MOD, fmt.MANTISSA_MASK)
  mantissa_high = mantissa_full.e(BinaryOps.DIV, fmt.MANTISSA_HIGH_SCALE)
  mantissa_mid = mantissa_full.e(BinaryOps.DIV, fmt.MANTISSA_MID_SCALE)
  mantissa_mid = mantissa_mid.e(BinaryOps.MOD, fmt.MANTISSA_MID_MASK)
  mantissa_low = mantissa_full.e(BinaryOps.MOD, fmt.MANTISSA_LOW_MASK)
  return sign_bit, exponent, mantissa_high, mantissa_mid, mantissa_low

def assemble_float(fmt: FloatFormat, sign_bit: LazyBuffer, exponent: LazyBuffer, mantissa: LazyBuffer) -> LazyBuffer:
  sign_bit = sign_bit.e(BinaryOps.MOD, fmt.SIGN_MASK)
  sign_bit = sign_bit.e(BinaryOps.MUL, fmt.SIGN_SCALE)
  exponent = exponent.e(BinaryOps.MOD, fmt.EXPONENT_MASK)
  exponent = exponent.e(BinaryOps.MUL, fmt.EXPONENT_SCALE)
  mantissa = mantissa.e(BinaryOps.MOD, fmt.MANTISSA_MASK)
  result = sign_bit.e(BinaryOps.ADD, exponent)
  result = result.e(BinaryOps.ADD, mantissa)
  return result.cast(fmt.FLOAT_TYPE, bitcast=True)

def reformat_for_reduction(fmt: FloatFormat, val: LazyBuffer) -> Tuple[LazyBuffer, LazyBuffer, LazyBuffer]:
  sign_bit, exponent, mantissa_high, mantissa_mid, mantissa_low = disassemble_float(fmt, val)

  float_high = assemble_float(fmt, sign_bit, exponent, mantissa_high)
  exponent_mid, scaled_mantissa_mid = mantissa_scale_search(mantissa_mid, fmt.MANTISSA_IMPLICIT_ONE_AT, fmt.MANTISSA_CHUNK_SIZE)
  exponent_mid = exponent_mid.e(BinaryOps.ADD, exponent_mid.const(fmt.MANTISSA_CHUNK_SIZE))
  float_mid = assemble_float(fmt, sign_bit, exponent.e(BinaryOps.SUB, exponent_mid), scaled_mantissa_mid)
  exponent_low, scaled_mantissa_low = mantissa_scale_search(mantissa_low, fmt.MANTISSA_IMPLICIT_ONE_AT, fmt.MANTISSA_CHUNK_SIZE)
  exponent_low = exponent_mid.e(BinaryOps.ADD, exponent_low.const(2 * fmt.MANTISSA_CHUNK_SIZE))
  float_low = assemble_float(fmt, sign_bit, exponent.e(BinaryOps.SUB, exponent_low), scaled_mantissa_low)

  return float_high, float_mid, float_low

def round_buffer(x: LazyBuffer) -> LazyBuffer:
  one = x.const(1)
  half = x.const(0.5)
  zero = x.const(0)

  is_negative = x.e(BinaryOps.CMPLT, zero)
  abs_x = is_negative.e(TernaryOps.WHERE, x.e(UnaryOps.NEG), x)
  floor_abs_x = abs_x.e(BinaryOps.SUB, abs_x.e(BinaryOps.MOD, one))
  frac_part = abs_x.e(BinaryOps.SUB, floor_abs_x)
  should_round_up = half.e(BinaryOps.CMPLT, frac_part)
  rounded_result = should_round_up.e(TernaryOps.WHERE, floor_abs_x.e(BinaryOps.ADD, one), floor_abs_x)
  return is_negative.e(TernaryOps.WHERE, rounded_result.e(UnaryOps.NEG), rounded_result)

def range_reduction(x: LazyBuffer, consts: dict[DType, Tuple[float, Sequence[float]]]) -> LazyBuffer:
  fmt = FloatFormat(x)
  xs = reformat_for_reduction(fmt, x)
  n = len(xs)
  consts = consts[x.dtype]
  bs = list(map(x.const, consts[1]))
  m = len(bs)
  ys = []

  for i in range(m + n - 1):
    y = bs[0].const(0)
    for j in range(m):
      if 0 <= i - j < n:
        y = y.e(BinaryOps.ADD, bs[j].e(BinaryOps.MUL, xs[i - j]))
    ys.append(y)

  y = ys[-1].e(BinaryOps.MOD, x.const(4))
  for y_n in reversed(ys[:-1]):
    y = y_n.e(BinaryOps.MOD, x.const(4)).e(BinaryOps.ADD, y)

  t = ys[0].e(BinaryOps.SUB, y)
  for y_n in ys[1:]:
    t.e(BinaryOps.ADD, y_n)

  n = round_buffer(y)
  f = y.e(BinaryOps.SUB, n).e(BinaryOps.ADD, t)
  r = f.e(BinaryOps.MUL, f.const(consts[0]))

  return r

PI_OVER_TWO_RANGE_REDUCTION_CONSTANTS = {
  dtypes.float64: (
    float.fromhex("0x1.921fb54442d18p0"), [
      float.fromhex("0x1.45f306p-1"),
      float.fromhex("0x1.b93910p-26"),
      float.fromhex("0x1.529fc2p-52"),
      float.fromhex("0x1.d5f47dp-78"),
      float.fromhex("0x1.34ddc0p-104"),
      float.fromhex("0x1.b6c52bp-129"),
      float.fromhex("0x1.93c439p-156"),
      float.fromhex("0x1.07f945p-186")
    ])}
