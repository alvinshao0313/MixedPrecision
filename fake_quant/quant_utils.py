import math
import transformers
import torch
import utils
import hadamard_utils
import fast_hadamard_transform


def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits - 1) - 1)
        minq = -maxq - 1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq


def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero


def asym_dequant(q, scale, zero):
    return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq, fp4=None):
    if fp4:
        return asym_dequant(quant_to_fp4((x / scale), fp4) + zero, scale, zero)
    else:
        return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return q, scale


def sym_dequant(q, scale):
    return scale * q


def sym_quant_dequant(x, scale, maxq, fp4=None):
    if fp4:
        return sym_dequant(quant_to_fp4(x / scale, fp4), scale)
    else:
        return sym_dequant(*sym_quant(x, scale, maxq))


def two_compl(x, bits: int):
    return torch.where(x < 0, 2 ** bits + x, x)

# Pack the int tensor. Each uint8 stores two int4 value.


def pack_i4(q):
    assert torch.is_signed(q), 'The tensor to be packed should be signed int'
    minq, maxq = get_minq_maxq(4, True)
    assert torch.all(torch.logical_and(q >= minq, q <= maxq))

    q_i8 = two_compl(q.to(dtype=torch.int8), 4).to(torch.uint8)
    q_i4 = q_i8[:, 0::2] | (q_i8[:, 1::2] << 4)
    return q_i4


# Unpack the quantized int4 tensor (stored in uint8) into int32 tensor.
def unpack_i4(x: torch.Tensor):
    assert x.dtype == torch.uint8, 'The tensor to be unpacked should be stored in uint8'

    out_shape = list(x.shape)
    out_shape[-1] *= 2  # Each uint8 packs two numbers

    # Low 4 bits
    x0 = (x & 0x0f).to(torch.int8)
    x0[x0 >= 8] -= 16
    x0 = x0.view(-1, x0.shape[-1])

    # High 4 bits
    x1 = ((x & 0xf0) >> 4).to(torch.int8)
    x1[x1 >= 8] -= 16
    x1 = x1.view(-1, x1.shape[-1])

    out = torch.empty(out_shape, device=x.device, dtype=torch.int32)
    out = out.view(-1, out.shape[-1])
    # Interleaving
    out[:, 0::2] = x0
    out[:, 1::2] = x1

    return out.view(out_shape)


def quant_to_fp4(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    FP4 e2m1 量化（1 sign + 2 exp + 1 mantissa，bias=1），超出范围截断
    输入：任意形状的 float16 或 float32 Tensor
    输出：同形状的量化后 float32 Tensor（仅做示例浮点重构，不输出 bit 级编码）
    mode: 
    """
    if mode == 'e2m1_0':    # {+_6, +_4, +_3, +_2, +_1.5, +_1, +_0.5, +_0}
        # 1. 符号位：正数为 +1，负数为 -1，零保留为 0
        sign = torch.sign(x)
        # 取绝对值，并防止 log2(0)
        x_abs = x.abs().clamp(min=1e-30)
        # 2. 计算真实指数 exp = floor(log2(x_abs))，再截断到 e ∈ { -2, -1, 0, 1, 2 }
        exp = torch.round(torch.log2(x_abs)).clamp(min=-2, max=2)
        # 3. 计算尾数分数部分 frac = x_abs / 2^exp - 1 ∈ [0, 1)
        frac = x_abs / (2.0 ** exp) - 1.
        # 量化尾数 m ∈ {0, 1} → 2 个等级：0 或 0.5
        frac_q = torch.round(frac * 2.0).clamp(min=0.0, max=1.0)
        # 4. 重构量化后的值
        x_fp4 = torch.where(exp >= 0, sign * (2.0 ** exp) * (1.0 + frac_q * 0.5), sign * (1.0 + exp * 0.5))
    elif mode == 'e2m1':    # {+_6, +_4, +_3, +_2, +_1.5, +_1, +_0.75, +_0.5}
        sign = torch.sign(x)
        x_abs = x.abs().clamp(min=1e-30)
        exp = torch.round(torch.log2(x_abs)).clamp(min=-1, max=2)
        frac = x_abs / (2.0 ** exp) - 1.
        frac_q = torch.round(frac * 2.0).clamp(min=0.0, max=1.0)
        x_fp4 = sign * (2.0 ** exp) * (1.0 + frac_q * 0.5)
    elif mode == 'e3m0':    # {+_16, +_8, +_4, +_2, +_1, +_0.5, +_0.25, +_0.125}
        sign = torch.sign(x)
        x_abs = x.abs().clamp(min=1e-30)
        exp = torch.round(torch.log2(x_abs)).clamp(min=-3, max=4)
        x_fp4 = sign * (2.0 ** exp)
    elif mode == 'e3m0_0':    # {+_16, +_8, +_4, +_2, +_1, +_0.5, +_0.25, +_0}
        sign = torch.sign(x)
        x_abs = x.abs().clamp(min=1e-30)
        exp = torch.round(torch.log2(x_abs)).clamp(min=-3, max=4)
        x_fp4 = torch.where(exp != -3, sign * (2.0 ** exp), 0.)
    return x_fp4


class ActQuantizer(torch.nn.Module):

    '''
        A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
        for the activations.
    '''

    def __init__(self):
        super(ActQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(1))
        self.register_buffer('zero', torch.zeros(1))
        self.bits = 16

    def free(self):
        self.zero = None
        self.scale = None

    def outlier_quant(self, x):
        self.outlier_scale = x.max() / self.maxq
        self.outlier_zero = None

    def forward(self, x):
        if self.first_token_protect:
            first_token = x[:, 0, :].squeeze()

        if self.mix_precision:
            protected_x = torch.where(torch.abs(x) > self.protect_threshold, x, 0.)
            self.protected_num += torch.sum(protected_x != 0)
            x = torch.where(torch.abs(x) <= self.protect_threshold, x, 0.)
            if self.protected_quant:
                self.outlier_quant(x)
                protected_x = sym_quant_dequant(protected_x, self.outlier_scale,
                                                self.outlier_zero, self.fp4).to(x.dtype)

        x_dtype = x.dtype

        if self.bits == 16:
            return x

        self.find_params(x)  # 首次量化参数计算

        if self.residual:
            if self.sym:
                tmp_q = sym_quant_dequant(x, self.scale, self.maxq, self.fp4).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = sym_quant_dequant(x - tmp_q, self.scale, self.maxq, self.fp4).to(x_dtype)
                return tmp_q + residual_q
            else:
                tmp_q = asym_quant_dequant(x, self.scale, self.zero, self.maxq, self.fp4).to(x_dtype)

                self.find_params(x - tmp_q)  # 为残差重新计算参数
                residual_q = asym_quant_dequant(x - tmp_q, self.scale, self.zero, self.maxq, self.fp4).to(x_dtype)
                out = tmp_q + residual_q
        elif self.sym:
            out = sym_quant_dequant(x, self.scale, self.maxq, self.fp4).to(x_dtype)
        else:
            out = asym_quant_dequant(x, self.scale, self.zero, self.maxq, self.fp4).to(x_dtype)

        if self.mix_precision:
            out += protected_x

        if self.first_token_protect:
            out[:, 0, :] = first_token

        return out

    # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
    def quantize(self, x):
        if self.sym:
            return sym_quant(x, self.scale, self.maxq)
        else:
            return asym_quant(x, self.scale, self.zero, self.maxq)

    def configure(self, bits,
                  groupsize=-1,
                  sym=False,
                  clip_ratio=1.0,
                  residual=False,
                  first_token_protect=False,
                  mix_precision=False,
                  protect_threshold=None,
                  protected_quant=False,
                  fp4=None):
        _, self.maxq = get_minq_maxq(bits, sym)
        self.maxq = torch.tensor(6.) if fp4 else self.maxq
        self.bits = bits
        self.groupsize = groupsize
        self.sym = sym
        self.clip_ratio = clip_ratio
        assert self.clip_ratio <= 1 and self.clip_ratio > 0, 'Clip ratio should be in (0, 1]'
        self.residual = residual
        self.first_token_protect = first_token_protect
        self.mix_precision = mix_precision
        self.protect_threshold = protect_threshold
        if self.mix_precision:
            assert self.protect_threshold > 0, 'Protect threshold should be positive'
            self.protected_num = 0
            self.protected_quant = protected_quant
        self.fp4 = fp4

    def find_params_per_token_groupwise(self, x):
        init_shape = x.shape
        reshaped_x = x.reshape(-1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize)

        xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = xmax / self.maxq
            self.scale[tmp] = 1
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin /
                                    self.scale) if not self.fp4 else quant_to_fp4((xmax + xmin) / 2 / self.scale, self.fp4)

        self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
        self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

    def find_params(self, x):
        if self.bits == 16:
            return

        dev = x.device
        self.maxq = self.maxq.to(dev)

        init_shape = x.shape

        if self.groupsize > 0:
            # group-wise per-token quantization
            self.find_params_per_token_groupwise(x)
            # utils.cleanup_memory(verbos=False)    # QuaRot 源码,不要加会导致推理时间X10
            return

        reshaped_x = x.reshape((-1, x.shape[-1]))

        tmp = torch.zeros(reshaped_x.shape[0], device=dev)
        xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
        xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            tmp = xmax == 0
            self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
            self.scale[tmp] = 1
            self.scale = self.scale.reshape(init_shape)
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            self.scale = (xmax - xmin) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

            self.scale = self.scale.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)
            self.zero = self.zero.unsqueeze(1).repeat(1, reshaped_x.shape[-1]).reshape(init_shape)


class ActQuantWrapper(torch.nn.Module):
    '''
        This class is a wrapper for the activation quantization.
        We extract the FP features in the forward pass and quantize the rest using
        the self.quantizer object.
        If a rotation Q is provided, the weight matrix will be rotated,
        a pre-forward hook will be registerd to rotate the activation before quantization.
    '''

    def __init__(self, module: torch.nn.Linear):
        super(ActQuantWrapper, self).__init__()
        assert isinstance(module, torch.nn.Linear)
        self.module = module
        self.weight = module.weight
        self.bias = module.bias
        self.quantizer = ActQuantizer()
        self.out_quantizer = ActQuantizer()
        self.register_buffer('had_K', torch.tensor(0))
        self._buffers['had_K'] = None
        self.K = 1
        self.online_full_had = False
        self.online_partial_had = False
        self.had_dim = 0
        self.fp32_had = False

    def extra_repr(self) -> str:
        str_ = f'Input Quantizer Bits: {self.quantizer.bits}'
        if self.quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.quantizer.sym else f' (Symmetric Per-Token)'

        str_ += f'\nOutput Quantizer Bits: {self.out_quantizer.bits}'
        if self.out_quantizer.bits < 16:
            str_ += f' (Asymmetric Per-Token)' if not self.out_quantizer.sym else f' (Symmetric Per-Token)'

        return str_

    def forward(self, x):
        x_dtype = x.dtype

        # Rotate, if needed
        if self.online_full_had:

            if self.fp32_had:  # Full Hadamard in FP32
                x = hadamard_utils.matmul_hadU_cuda(x.float(), self.had_K, self.K).to(x_dtype)
            else:  # Full Hadamard in FP16
                x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)

        elif self.online_partial_had:
            # todo: implement this in QAttention to avoid reshaping!

            if self.fp32_had:
                x = x.float()

            init_shape = x.shape
            if self.K == 1:
                x = fast_hadamard_transform.hadamard_transform(x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim).transpose(1, 2),
                                                               scale=1 / math.sqrt(init_shape[-1] // self.had_dim)).transpose(1, 2)
            else:
                x = (self.had_K.to(x.dtype) @ x.reshape(-1,
                     init_shape[-1] // self.had_dim, self.had_dim)) / math.sqrt(init_shape[-1] // self.had_dim)

            if self.fp32_had:
                x = x.to(x_dtype)
            x = x.reshape(init_shape)

        if self.quantizer.bits < 16:  # Quantize, if needed
            # self.quantizer.find_params(x)  # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.quantizer(x).to(x_dtype)
            self.quantizer.free()

        x = self.module(x).to(x_dtype)

        if self.out_quantizer.bits < 16:  # Quantize the output, if needed
            # self.out_quantizer.find_params(x) # QuaRot 源码，现修改在 quantizer.forward 函数中
            x = self.out_quantizer(x).to(x_dtype)
            self.out_quantizer.free()

        return x


class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,
        groupsize=-1, static_groups=False, fp4=None
    ):
        self.bits = bits
        self.perchannel = perchannel
        if fp4:
            assert sym, 'FP4 quantization only supports symmetric quantization'
        self.fp4 = fp4
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.maxq = torch.tensor(2**(bits - 1) - 1) if sym else torch.tensor(2**bits - 1)
        self.groupsize = groupsize
        self.static_groups = static_groups
        self.maxq = torch.tensor(6.) if fp4 else self.maxq

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
            self.scale = xmax / self.maxq
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            xmin.masked_fill(tmp, -1)
            xmax.masked_fill(tmp, +1)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = xmax1 / self.maxq
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq, self.fp4)
                else:
                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1),
                                           zero1.unsqueeze(1), self.maxq, self.fp4)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]

        if not self.perchannel:
            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq, self.fp4).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero,
                                      self.maxq, self.fp4).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


def add_actquant(module, name='', layers=[torch.nn.Linear,
                                          ActQuantWrapper,
                                          transformers.models.falcon.modeling_falcon.FalconLinear]):
    if isinstance(module, ActQuantWrapper):
        return
    for attr in dir(module):
        tmp = getattr(module, attr)
        if type(tmp) in layers:
            setattr(module, attr, ActQuantWrapper(tmp))
        if type(tmp) == torch.nn.Sequential:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.Sequential(*replaced))
        if type(tmp) == torch.nn.ModuleList:
            replaced = []
            for i, child in enumerate(tmp.children()):
                if type(child) in layers:
                    replaced.append(ActQuantWrapper(child))
                else:
                    replaced.append(child)
            setattr(module, attr, torch.nn.ModuleList(replaced))
    for name1, child in module.named_children():
        add_actquant(child, name + '.' + name1 if name != '' else name1, layers)


def find_qlayers(module, layers=[torch.nn.Linear,
                                 ActQuantWrapper], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


from transformers.activations import ACT2FN


def quant_act_fn(x: torch.Tensor,
                 act_fn: str,
                 q_config: dict,
                 use_r4: bool = False
                 ):
    act_fn = ACT2FN[config.hidden_act]
    quantizer = ActQuantizer()
    quantizer.configure(**q_config)
    if use_r4:
        x = hadamard_utils.matmul_hadU_cuda(x, self.had_K, self.K)
    else:
        quantizer.configure(bits=8, groupsize=-1, sym=True, residual=False)
    x = quantizer(act_fn(x))
    quantizer.free()
    return x


def protection_ratio(model, total_tokens):
    """
    计算保护比例
    :param model: 模型
    :param total_tokens: 总的 token 数量
    :return: 保护比例
    """
    protected_num = 0
    for name, module in model.named_modules():
        if isinstance(module, ActQuantizer) and hasattr(module, 'protected_num'):
            protected_num += module.protected_num
    total_param = model.config.num_hidden_layers * model.config.hidden_size * total_tokens * 5 \
        + model.config.num_hidden_layers * model.config.intermediate_size * total_tokens
    return protected_num / total_param * 100
