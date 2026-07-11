"""Offline artifact fitting (Jacobian lens, OPRD bridge).

These are standalone CLI entrypoints run *before* training to produce ``lens.pt``
(arm C) and ``bridge.pt`` (arm D). They are never imported by the training loop —
the arms only ``torch.load`` the artifacts.
"""
