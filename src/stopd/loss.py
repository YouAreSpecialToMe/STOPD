"""The STOPD distillation loss, registered in verl's loss registry as `stopd_k1`.

src/sitecustomize.py imports this module right after verl's loss module loads,
in every process. The loss is verl's sampled-token reverse KL,
k1 = log p_student(y_t) - log p_teacher(y_t) on the student's own tokens, with
each token multiplied by its IPW weight and by N_obs / C_k
(stopd.verl_patch.weight_losses). verl then uses -loss as the per-token
advantage of its policy-gradient update.
"""

from verl.trainer.distillation.losses import DistillationLossSettings, register_distillation_loss
from verl.trainer.ppo.core_algos import kl_penalty
from verl.utils.metric import AggregationType, Metric
from verl.workers.utils.padding import no_padding_2_padding

from stopd.verl_patch import LOSS_MODE, weight_losses


@register_distillation_loss(DistillationLossSettings(names=[LOSS_MODE], use_estimator=True))  # type: ignore[arg-type]
def stopd_k1(config, distillation_config, model_output, data):
    student = no_padding_2_padding(model_output["log_probs"], data)
    teacher = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    mask = data["response_mask"]
    mask = mask.bool().to_padded_tensor(False) if mask.is_nested else mask.bool()
    assert teacher.shape == student.shape == mask.shape
    raw = kl_penalty(logprob=student, ref_logprob=teacher, kl_penalty="k1")
    losses = weight_losses(raw, mask, data)
    # k1 can be negative, so log the mean absolute (unweighted) loss, as verl does.
    metrics = {"distillation/abs_loss": Metric(AggregationType.MEAN, raw[mask].abs().mean())}
    return losses, metrics
