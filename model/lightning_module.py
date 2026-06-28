import lightning as L
import torch

from torch import Tensor, nn
from torchmetrics import MeanMetric, MetricCollection

from .config import NNUELightningConfig
from .model import NNUEModel
from .lambda_utils import LambdaController


def _get_parameters(layers: list[nn.Module], get_biases: bool = False):
    return [
        p
        for layer in layers
        for name, p in layer.named_parameters()
        if ("bias" in name) == get_biases and p.requires_grad
    ]


def calculate_sf_loss(scorenet, score, outcome, loss_params, actual_lambda):
    # convert the network and search scores to an estimate match result
    # based on the win_rate_model, with scalings and offsets optimized
    q = (scorenet - loss_params.in_offset) / loss_params.in_scaling
    qm = (-scorenet - loss_params.in_offset) / loss_params.in_scaling
    qf = 0.5 * (1.0 + q.sigmoid() - qm.sigmoid())

    s = (score - loss_params.out_offset) / loss_params.out_scaling
    sm = (-score - loss_params.out_offset) / loss_params.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())

    # blend that eval based score with the actual game outcome
    t = outcome

    pt = pf * actual_lambda + t * (1.0 - actual_lambda)

    # use a MSE-like loss function
    loss = torch.pow(torch.abs(pt - qf), loss_params.pow_exp)
    if loss_params.qp_asymmetry != 0.0:
        loss = loss * ((qf > pt) * loss_params.qp_asymmetry + 1)

    weights = 1 + (2.0**loss_params.w1 - 1) * torch.pow((pf - 0.5) ** 2 * pf * (1 - pf), loss_params.w2)
    loss = (loss * weights).sum() / weights.sum()

    return loss


def calculate_value_head_loss(value_logits, score, loss_params):
    """Cross-entropy of the auxiliary categorical value head against the teacher.

    The teacher `score` is mapped to a win probability in [0, 1] with the same
    conversion the main loss uses, then discretized over `num_bins` uniform bins.
    With `aux_value_label_smoothing > 0` the target is an HL-Gauss soft label
    (a Gaussian centered on the true value, integrated per bin); otherwise it is
    a hard one-hot label.
    """
    num_bins = value_logits.shape[1]

    # Teacher value -> win probability in [0, 1] (matches calculate_sf_loss).
    s = (score - loss_params.out_offset) / loss_params.out_scaling
    sm = (-score - loss_params.out_offset) / loss_params.out_scaling
    pf = 0.5 * (1.0 + s.sigmoid() - sm.sigmoid())
    pf = pf.reshape(-1).clamp(0.0, 1.0)

    edges = torch.linspace(0.0, 1.0, num_bins + 1, device=value_logits.device, dtype=value_logits.dtype)
    bin_width = 1.0 / num_bins
    sigma = loss_params.aux_value_label_smoothing * bin_width

    if sigma > 0.0:
        # Mass per bin from the Gaussian CDF (HL-Gauss).
        z = (edges.unsqueeze(0) - pf.unsqueeze(1)) / (sigma * (2.0 ** 0.5))
        cdf = 0.5 * (1.0 + torch.erf(z))
        target = cdf[:, 1:] - cdf[:, :-1]
        target = target / target.sum(dim=1, keepdim=True).clamp_min(1e-12)
    else:
        idx = torch.bucketize(pf, edges[1:-1].contiguous())
        target = torch.zeros_like(value_logits)
        target.scatter_(1, idx.unsqueeze(1), 1.0)

    log_probs = torch.log_softmax(value_logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


class NNUE(L.LightningModule):

    def __init__(
        self,
        config: NNUELightningConfig,
        max_epoch=None,
        num_batches_per_epoch=None,
        param_index=0,
        num_psqt_buckets=8,
        num_ls_buckets=8,
    ):
        super().__init__()

        self.model: NNUEModel = NNUEModel(
            config.features,
            config.model_config,
            num_psqt_buckets,
            num_ls_buckets,
        )
        self.config = config
        self.max_epoch = max_epoch
        self.num_batches_per_epoch = num_batches_per_epoch
        self.param_index = param_index

        # lazy init so `resume_from_model` with config changes works correctly
        self.optimizer_wrapper = None

        # Initialize the lambda controller
        self.lambda_scheduler = LambdaController()

        self.loss_metrics = MetricCollection(
            {
                "train_loss_epoch": MeanMetric(),
                "val_loss_epoch": MeanMetric(),
                "test_loss_epoch": MeanMetric(),
            }
        )

    # --- setup optimizers and training hooks ---
    def configure_optimizers(self):
        optimizer_config = self.config.optimizer_config
        self.optimizer_wrapper = optimizer_config.get_optimizer_wrapper()

        LRs = [optimizer_config.lr] * 10

        ft_wd = optimizer_config.ft_weight_decay
        dense_wd = optimizer_config.dense_weight_decay
        factorized_wd = optimizer_config.factorized_weight_decay

        train_params = [
            # Feature Transformer
            {
                "params": _get_parameters([self.model.input], get_biases=False),
                "lr": LRs[0],
                "weight_decay": ft_wd,
            },
            {
                "params": _get_parameters([self.model.input], get_biases=True),
                "lr": LRs[1],
                "weight_decay": 0.0,
            },
            # Dense Layer Stacks
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.weight],
                "lr": LRs[2],
                "weight_decay": factorized_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.factorized_linear.bias],
                "lr": LRs[3],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.weight],
                "lr": LRs[4],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l1.linear.bias],
                "lr": LRs[5],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.weight],
                "lr": LRs[6],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.l2.linear.bias],
                "lr": LRs[7],
                "weight_decay": 0.0,
            },
            {
                "params": [self.model.layer_stacks.output.linear.weight],
                "lr": LRs[8],
                "weight_decay": dense_wd,
            },
            {
                "params": [self.model.layer_stacks.output.linear.bias],
                "lr": LRs[9],
                "weight_decay": 0.0,
            },
        ]

        # Auxiliary value head (training-only, not quantized/exported). Guarded so
        # disabling the head or resuming a pre-head checkpoint stays valid.
        value_head = getattr(self.model.layer_stacks, "value_head", None)
        if value_head is not None:
            train_params += [
                {
                    "params": [value_head.linear.weight],
                    "lr": optimizer_config.lr,
                    "weight_decay": dense_wd,
                },
                {
                    "params": [value_head.linear.bias],
                    "lr": optimizer_config.lr,
                    "weight_decay": 0.0,
                },
            ]

        return self.optimizer_wrapper.configure_optimizers(train_params)

    # --- train / eval switch ---
    def train(self, mode: bool = True):
        retval = super().train(mode)

        if self.optimizer_wrapper is not None:
            if mode:
                self.optimizer_wrapper.switch_to_train(True)
            else:
                self.optimizer_wrapper.switch_to_eval()

        return retval

    def eval(self):
        return self.train(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    # --- lightning hooks ---
    def on_train_epoch_start(self):
        self.optimizer_wrapper.on_train_epoch_start(self)

    def on_train_epoch_end(self):
        self.optimizer_wrapper.on_train_epoch_end(self)
        self._log_epoch_end("train_loss_epoch")

    def on_validation_epoch_start(self):
        self.optimizer_wrapper.on_validation_epoch_start(self)

    def on_validation_epoch_end(self):
        self._log_epoch_end("val_loss_epoch")

    def on_test_epoch_start(self):
        self.optimizer_wrapper.on_test_epoch_start(self)

    def on_test_epoch_end(self):
        self._log_epoch_end("test_loss_epoch")

    def on_save_checkpoint(self, checkpoint):
        self.optimizer_wrapper.on_save_checkpoint(self, checkpoint)
        self.lambda_scheduler.on_save_checkpoint(checkpoint)

    def on_load_checkpoint(self, checkpoint):
        self.lambda_scheduler.on_load_checkpoint(self, checkpoint)

    def on_train_batch_start(self, batch, batch_idx):
        self.optimizer_wrapper.on_train_batch_start(self, batch, batch_idx)

    def _log_epoch_end(self, loss_type):
        self.log(
            f"{loss_type}",
            self.loss_metrics[f"{loss_type}"],
            prog_bar=False,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

    # --- Training step implementation ---

    def training_step(self, batch, batch_idx):
        return self.step_(batch, batch_idx, "train_loss")

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "val_loss")

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        self.step_(batch, batch_idx, "test_loss")

    def step_(self, batch: tuple[Tensor, ...], batch_idx, loss_type):
        _ = batch_idx  # unused, but required by pytorch-lightning

        (
            us,
            them,
            white_indices,
            black_indices,
            outcome,
            score,
            piece_count,
        ) = batch
        loss_params = self.config.loss_params

        scorenet, value_logits = self.model(
            us,
            them,
            white_indices,
            black_indices,
            piece_count,
            self.config.use_fake_act_quantization,
            self.config.use_fake_weight_quantization,
            return_value_logits=True,
            value_grad_scale=loss_params.aux_value_grad_scale,
        )

        scorenet = scorenet * self.model.quantization.nnue2score

        actual_lambda = self.lambda_scheduler(
            loss_params=loss_params,
            current_epoch=self.current_epoch,
            max_epoch=self.max_epoch,
            is_training=self.training,
            scorenet=scorenet
        )

        # Main head: blends teacher eval (score) with game result (outcome) via
        # actual_lambda.
        sf_loss = calculate_sf_loss(
            scorenet, score, outcome, loss_params, actual_lambda
        )

        # The auxiliary value head contributes a small, separately-weighted term.
        # value_logits is None when the head is disabled or absent (e.g. resuming
        # an old checkpoint), in which case the objective is unchanged.
        #
        # NOTE: the aux head targets the TEACHER value (score) ONLY. It is
        # deliberately independent of lambda and never sees `outcome`/the game
        # result, regardless of what lambda the main head uses.
        total_loss = sf_loss
        if value_logits is not None and loss_params.aux_value_weight > 0.0:
            aux_loss = calculate_value_head_loss(value_logits, score, loss_params)
            total_loss = total_loss + loss_params.aux_value_weight * aux_loss
            self.log(
                f"{loss_type}_aux",
                aux_loss,
                prog_bar=False,
                sync_dist=False,
                on_epoch=False,
                on_step=True,
            )

        # Track the main (sf) loss for the epoch metric so it stays comparable
        # to runs without the auxiliary head.
        self.loss_metrics[f"{loss_type}_epoch"].update(sf_loss)
        self.log(
            loss_type,
            sf_loss,
            prog_bar=False,
            sync_dist=False,
            on_epoch=False,
            on_step=True,
        )
        return total_loss
