"""opdlens — on-policy distillation with intermediate-layer (lens) supervision.

A four-arm distillation trainer (A logits / B logit-lens / C jacobian-lens /
D hidden-mse) sharing one rollout+teacher+optimizer+eval spine; the arms differ
only in ``arm.aux_loss``. See ``docs`` / the design plan for the architecture.
"""

__version__ = "0.1.0"
