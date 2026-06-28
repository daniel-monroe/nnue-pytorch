from typing import Generator

import torch
from torch import nn

from .stacked_linear import FactorizedStackedLinear, StackedLinear
from .config import LayerStacksConfig
from ..quantize import QuantizationManager


class _GradScale(torch.autograd.Function):
    """Identity in the forward pass, scales the gradient in the backward pass.

    Used to throttle how strongly the auxiliary value head perturbs the shared
    trunk without affecting the gradient the head's own parameters receive.
    """

    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def _grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    if scale == 1.0:
        return x
    return _GradScale.apply(x, scale)


class LayerStacks(nn.Module):
    def __init__(self, count: int, config: LayerStacksConfig, quantization: QuantizationManager):
        super().__init__()

        self.count = count
        self.L1 = config.L1
        self.L2 = config.L2
        self.L3 = config.L3
        self.quantization = quantization

        # Factorizer only for the first layer because later
        # there's a non-linearity and factorization breaks.
        # This is by design. The weights in the further layers should be
        # able to diverge a lot.
        self.l1 = FactorizedStackedLinear(2 * self.L1 // 2, self.L2 + 1, count, quantization, "ls_l1")
        self.l2 = StackedLinear(self.L2 * 2, self.L3, count, quantization, "ls_l2")
        self.output = StackedLinear(self.L3, 1, count, quantization, "ls_output")

        with torch.no_grad():
            self.output.linear.bias.zero_()

        # Auxiliary categorical value head. It takes the same L3 (32-neuron)
        # activation that feeds the scalar `output`, and predicts a distribution
        # over the (binned) teacher value. Like `output`, it is a StackedLinear,
        # so each of the `count` buckets gets its own independent parameters.
        # It is full-precision (not quantized / not weight-clipped) and is never
        # exported to the .nnue: it only shapes the trunk during training.
        self.num_value_bins = getattr(config, "num_value_bins", 0)
        if self.num_value_bins > 0:
            # Save/restore the RNG around the head so that adding it does not
            # perturb the initialization of any later modules (e.g. the feature
            # transformer). This keeps a with-head run's trunk init bit-identical
            # to an otherwise-identical no-head run.
            rng_state = torch.get_rng_state()
            self.value_head = StackedLinear(self.L3, self.num_value_bins, count)
            torch.set_rng_state(rng_state)

    def forward(
        self, x: torch.Tensor,
        ls_indices: torch.Tensor,
        fake_quantize_acts: bool=True,
        fake_quantize_weights: bool=True,
        return_value_logits: bool=False,
        value_grad_scale: float=1.0,
    ):
        l1c_ = self.l1(x, ls_indices, fake_quantize_weights)
        l1x_, l1x_out = l1c_.split(self.L2, dim=1)

        l1_sqr = torch.pow(l1x_, 2.0)
        if fake_quantize_acts:
            l1_sqr = self.quantization.fake_quantize_ls_act(l1_sqr)
        l1_sqr = l1_sqr * (self.quantization.sqr_crelu_correction_factor)

        if fake_quantize_acts:
            l1x_ = self.quantization.fake_quantize_ls_act(l1x_)

        l1x_ = torch.cat([l1_sqr, l1x_], dim=1)
        l1x_ = self.quantization.clip_ls_act(l1x_)

        l2c_ = self.l2(l1x_, ls_indices, fake_quantize_weights)
        if fake_quantize_acts:
            l2c_ = self.quantization.fake_quantize_ls_act(l2c_)
        l2x_ = self.quantization.clip_ls_act(l2c_)

        l3c_ = self.output(l2x_, ls_indices, fake_quantize_weights)
        if fake_quantize_acts:
            l1x_out = self.quantization.fake_quantize_skip_act(l1x_out)

        l3x_ = l3c_ + l1x_out
        if fake_quantize_acts:
            l3x_ = self.quantization.fake_quantize_output(l3x_)

        if return_value_logits:
            value_logits = None
            # `getattr` guard keeps old checkpoints (pickled before the head
            # existed) loadable: they simply won't produce auxiliary logits.
            if getattr(self, "value_head", None) is not None:
                # The head reads the *same* L3 activation as the scalar output.
                value_logits = self.value_head(
                    _grad_scale(l2x_, value_grad_scale),
                    ls_indices,
                    fake_quantize_weights=False,
                )
            return l3x_, value_logits

        return l3x_

    @torch.no_grad()
    def zero_virtual_weights(self) -> None:
        self.l1.zero_virtual_weights()

    @torch.no_grad()
    def get_coalesced_layer_stacks(
        self,
    ) -> Generator[tuple[nn.Linear, nn.Linear, nn.Linear], None, None]:
        # During training the buckets are represented by a single, wider, layer.
        # This representation needs to be transformed into individual layers
        # for the serializer, because the buckets are interpreted as separate layers.
        for i in range(self.count):
            yield self.l1.at_index(i), self.l2.at_index(i), self.output.at_index(i)

    @torch.no_grad()
    def coalesce_layer_stacks_inplace(self) -> None:
        self.l1.coalesce_weights()
