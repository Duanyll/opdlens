# Lens gradient, endpoint, and frozen-readout audit

Date: 2026-07-13 to 2026-07-14

Branch: `exp/lens-audit`

Implementation commits: `3a5d0a8`, `c7991f4`, `cf6ccc4`, `329baad`

All experiments use the round-2 standard OPD settings:

- `base_beta = 1.0`;
- `base_temperature = 1.0`;
- `rollout_temperature = 1.0`;
- full fine-tuning of Qwen3.5-2B from the same initialization;
- Qwen3.5-9B teacher;
- teacher layers `[16, 24]`, mapped to student layers `[12, 18]`;
- lens temperature 1.0.

## 1. Questions

1. How large are the raw base and lens gradients, and what is their angle?
2. Does a lens-gradient perturbation move the final student policy as strongly as a base-gradient perturbation?
3. Does the live student final norm / LM head provide an unstable optimization path?
4. How much does an offline student J-lens become stale after 300 training steps?

## 2. Audit protocol

The gradient audit uses two fixed on-policy GSM8K rollouts generated from the initial student. It independently backpropagates the round-2 base loss, the complete auxiliary loss, and each supervised layer's auxiliary term.

For each objective it records:

- global and parameter-group gradient norms;
- base/aux gradient cosine;
- the expected combined direction for weights 0.003, 0.01, 0.03, 0.1, and 0.3;
- coordinate-wise sign flips and auxiliary-dominated coordinates;
- per-layer hidden-gradient cosine.

A coordinate is auxiliary-dominated when $|\lambda g_{\mathrm{aux},i}|>|g_{\mathrm{base},i}|$, and it flips when $\operatorname{sign}(g_{\mathrm{base},i}+\lambda g_{\mathrm{aux},i})\ne\operatorname{sign}(g_{\mathrm{base},i})$.

The endpoint audit perturbs the captured student hidden state in the normalized negative-gradient direction. Both base and auxiliary perturbations have exactly 1% of the hidden state's RMS. It then reruns the remaining student layers and measures final-policy KL on the same completion positions.

This is a hidden-interface audit, not an optimizer-step simulation. It isolates how strongly the downstream model responds to equal-sized changes in the two gradient directions.

Raw JSON reports are under:

`/gdata/users/duanyll/opdlens/lens-audit/gradient/`

## 3. Initial-state parameter gradient geometry

| Arm | Base grad norm | Raw aux grad norm | Base/aux cosine | Weighted aux/base norm at 0.01 | Base/combined cosine at 0.1 |
|---|---:|---:|---:|---:|---:|
| B logit lens | 27.81 | 44.26 | 0.200 | 0.016 | 0.989 |
| C teacher J-lens | 27.80 | 63.01 | 0.242 | 0.023 | 0.979 |
| E symmetric J-lens | 27.80 | 64.23 | 0.258 | 0.023 | 0.979 |

Global L2 geometry alone does not predict the historical degradation. At weight 0.01 the combined gradient is almost indistinguishable from the base gradient; even at weight 0.1 its cosine with the base gradient remains about 0.98 to 0.99.

The optimizer clips the combined gradient to norm 1. Since all audited raw norms exceed 1, its scale is $s(\lambda)=1/\|g_{\mathrm{base}}+\lambda g_{\mathrm{aux}}\|$. Relative to base-only clipping, the retained coefficient on the base gradient is:

| Arm | Weight 0.01 | Weight 0.1 | Weight 0.3 |
|---|---:|---:|---:|
| B | 99.7% | 95.8% | 83.9% |
| C | 99.4% | 92.8% | 74.7% |
| E | 99.4% | 92.4% | 73.8% |

Large auxiliary weight therefore steals part of the fixed pre-Adam clipped-gradient budget even without an exploding norm. This matters especially because the endpoint audit below shows that the purchased auxiliary direction has very low immediate endpoint gain. At weight 0.1 the 4% to 8% reduction is not large enough to explain the whole accuracy gap by itself, but it accumulates with coordinate conflict. This coefficient is an input-to-Adam diagnostic, not a claim that the adaptive optimizer scales its final parameter update linearly.

The individual shallow layer is much less aligned than the layer-averaged auxiliary:

| Arm | Pair | Parameter-gradient cosine | Hidden-gradient cosine |
|---|---|---:|---:|
| B | s12 / t16 | -0.024 | -0.004 |
| B | s18 / t24 | 0.216 | 0.025 |
| C | s12 / t16 | 0.033 | -0.004 |
| C | s18 / t24 | 0.288 | 0.027 |
| E | s12 / t16 | 0.120 | 0.002 |
| E | s18 / t24 | 0.241 | 0.029 |

This supports a depth-specific conflict, especially at the shallower workspace-band layer, but averaging and shared parameter paths hide it in the global cosine.

## 4. Endpoint response

Final-policy KL caused by equal 1%-RMS hidden perturbations:

| Arm | Pair | Base direction KL | Aux direction KL | Aux/base endpoint KL |
|---|---|---:|---:|---:|
| B | s12 / t16 | 0.07364 | 0.000292 | 0.0040 |
| B | s18 / t24 | 0.06103 | 0.000475 | 0.0078 |
| C | s12 / t16 | 0.07439 | 0.000185 | 0.0025 |
| C | s18 / t24 | 0.06013 | 0.000408 | 0.0068 |
| E | s12 / t16 | 0.07367 | 0.000191 | 0.0026 |
| E | s18 / t24 | 0.06013 | 0.000550 | 0.0092 |

The auxiliary hidden-gradient directions are 100 to 400 times less coupled to the current final policy than the base direction. The deeper layer is consistently more coupled than the shallower layer, but the absolute effect remains small.

This result argues against the simplest version of the semantic-time overshoot hypothesis. The immediate failure mode is not that a same-sized lens step moves the final policy too far. Instead, lens optimization primarily moves weakly downstream-coupled representation directions.

That observation has two interpretations that later experiments must distinguish:

1. the auxiliary is writing useful non-motor workspace information that the output loss cannot see;
2. the auxiliary is optimizing probe-visible directions that the real downstream computation largely ignores.

## 5. Coordinate takeover and tied embeddings

Qwen3.5-2B ties its LM head to its input embedding. With the live lens readout, the auxiliary therefore has a direct dense gradient path into the same matrix that defines all input token embeddings.

At the nominal weight 0.01:

| Arm | Readout | Global base-sign flips | Global aux-dominated coordinates | Embedding/head sign flips | Embedding/head aux-dominated |
|---|---|---:|---:|---:|---:|
| B | live | 15.4% | 30.8% | 47.4% | 94.6% |
| B | frozen | 0.3% | 0.7% | 0.0% | 0.0% |
| C | live | 14.0% | 30.9% | 42.6% | 94.1% |
| C | frozen | 0.5% | 1.0% | 0.0% | 0.0% |
| E | live | 16.2% | 28.9% | 49.5% | 87.5% |
| E | frozen | 0.5% | 1.0% | 0.0% | 0.0% |

At weight 0.1, freezing reduces global base-sign flips from 19% to 22% down to 3.5% to 4.8%.

This effect is almost invisible in global L2 cosine because the affected embedding/head coordinates individually have small base gradients, but there are very many of them. It is the clearest concrete invariant violation found by the audit: the probe's vocabulary codebook is also a live, tied model parameter.

Freezing changes the raw auxiliary norm only from 44.26 to 44.07 for B, 63.01 to 62.70 for C, and 64.23 to 63.82 for E; base/aux cosine is unchanged to three decimals. At step 0 the probe forward map and hidden/backbone gradient are identical, so freezing removes only the direct readout-parameter gradient. The dramatic sign-flip count is therefore caused by very many low-magnitude head coordinates, not by most of the auxiliary L2 energy.

The training ablation is necessary before calling this causal. A live readout may be harmful because it pollutes the tied embedding, or it may act as a release valve that lets the model satisfy the auxiliary without forcing larger backbone changes.

## 6. Historical B checkpoints

Historical task accuracy under the same round-2 recipe:

| Aux weight | Step 50 | Step 100 | Step 300 |
|---:|---:|---:|---:|
| 0.003 | 0.748 | 0.812 | 0.834 |
| 0.1 | 0.758 | 0.779 | 0.798 |
| 0.3 | 0.698 | 0.735 | 0.779 |

The 0.3 run shows an early collapse and partial recovery. Checkpoint audits still show high global base/combined cosine:

| Run | Step | Weighted aux/base norm | Base/combined cosine | Block sign-flip range | Embedding/head sign flips |
|---|---:|---:|---:|---:|---:|
| weight 0.003 | 100 | 0.007 | 1.000 | 0.2% to 0.3% | 47.8% |
| weight 0.1 | 100 | 0.101 | 0.997 | 2.0% to 3.7% | 49.0% |
| weight 0.3 | 100 | 0.236 | 0.981 | 5.7% to 9.6% | 46.7% |
| weight 0.003 | 300 | 0.003 | 1.000 | 0.1% | 47.4% |
| weight 0.1 | 300 | 0.089 | 0.997 | 2.1% to 3.6% | 48.6% |
| weight 0.3 | 300 | 0.177 | 0.988 | 4.8% to 7.4% | 47.4% |

The embedding/head sign-flip fraction is high even at small weight because the live auxiliary supplies dense gradients where the base gradient is very small. The magnitude of backbone takeover, especially in late blocks, tracks auxiliary weight more cleanly than global cosine.

Combined-gradient clipping retains 99.8% to 99.9% of the base-only coefficient in the weight-0.003 checkpoints, 93.5% to 95.1% at weight 0.1, and 89.6% to 93.3% at weight 0.3. This persistent pre-Adam pressure is not a transient initialization artifact.

The weak endpoint coupling also persists after training. The table reports auxiliary/base final-policy KL for equal 1%-RMS hidden perturbations:

| Aux weight | Step | s12 / t16 | s18 / t24 |
|---:|---:|---:|---:|
| 0.003 | 100 | 0.00062 | 0.0110 |
| 0.003 | 300 | 0.00087 | 0.0167 |
| 0.1 | 100 | 0.00099 | 0.0150 |
| 0.1 | 300 | 0.00111 | 0.0183 |
| 0.3 | 100 | 0.00163 | 0.0108 |
| 0.3 | 300 | 0.00130 | 0.0153 |

Even in the degraded runs, a lens-gradient perturbation of the same hidden RMS causes less than 0.2% of the base-direction KL at layer 12 and less than 2% at layer 18. There is no endpoint-overshoot transition as auxiliary weight increases. The transition occurs in parameter-coordinate takeover, not in the immediate functional gain from hidden movement to output policy.

## 7. J-lens drift after training

The student J-lens was independently refit on four round-2 step-300 full checkpoints: standard A, E at weight 0.01, and the weight-0.1 E live/frozen pair. All fits use the exact original 264-prompt GSM8K CoT recipe and layers `[6, 12, 18]`.

Artifacts:

- `/gdata/users/duanyll/opdlens/artifacts/lens-audit/logits-r2-step300-refit.pt`
- `/gdata/users/duanyll/opdlens/artifacts/lens-audit/symjlens-r2-step300-refit.pt`
- `/gdata/users/duanyll/opdlens/artifacts/lens-audit/symjlens-aw01-live-step300-refit.pt`
- `/gdata/users/duanyll/opdlens/artifacts/lens-audit/symjlens-aw01-frozen-step300-refit.pt`

Matrix drift from the common step-0 lens:

| Run | Layer | Frobenius cosine | Relative difference over step-0 | Norm ratio after/before |
|---|---:|---:|---:|---:|
| A | 6 | 0.959 | 0.292 | 0.892 |
| A | 12 | 0.984 | 0.191 | 0.921 |
| A | 18 | 0.996 | 0.106 | 0.946 |
| E, weight 0.01 | 6 | 0.953 | 0.304 | 0.972 |
| E, weight 0.01 | 12 | 0.979 | 0.206 | 0.982 |
| E, weight 0.01 | 18 | 0.994 | 0.120 | 0.948 |
| E, weight 0.1, live | 6 | 0.939 | 0.357 | 1.026 |
| E, weight 0.1, live | 12 | 0.964 | 0.271 | 1.017 |
| E, weight 0.1, live | 18 | 0.985 | 0.187 | 0.921 |
| E, weight 0.1, frozen | 6 | 0.935 | 0.377 | 1.061 |
| E, weight 0.1, frozen | 12 | 0.961 | 0.288 | 1.040 |
| E, weight 0.1, frozen | 18 | 0.984 | 0.191 | 0.926 |

Finite-difference cosine on each trained model:

| Run | Layer | Step-0 lens | Refit lens | Relative error, old | Relative error, refit |
|---|---:|---:|---:|---:|---:|
| A | 6 | 0.849 | 0.882 | 0.559 | 0.483 |
| A | 12 | 0.913 | 0.931 | 0.423 | 0.372 |
| A | 18 | 0.976 | 0.981 | 0.226 | 0.195 |
| E, weight 0.01 | 6 | 0.848 | 0.886 | 0.548 | 0.478 |
| E, weight 0.01 | 12 | 0.911 | 0.932 | 0.423 | 0.372 |
| E, weight 0.01 | 18 | 0.974 | 0.980 | 0.238 | 0.201 |
| E, weight 0.1, live | 6 | 0.843 | 0.892 | 0.548 | 0.463 |
| E, weight 0.1, live | 12 | 0.901 | 0.935 | 0.442 | 0.366 |
| E, weight 0.1, live | 18 | 0.965 | 0.979 | 0.284 | 0.205 |
| E, weight 0.1, frozen | 6 | 0.843 | 0.892 | 0.550 | 0.468 |
| E, weight 0.1, frozen | 12 | 0.898 | 0.934 | 0.451 | 0.369 |
| E, weight 0.1, frozen | 18 | 0.963 | 0.978 | 0.288 | 0.208 |

On the untouched base model the ordering reverses, as expected: the common step-0 lens is consistently better than either refit lens.

The offline student lens does become stale, especially at shallow layers. The drift and fidelity recovery are almost identical between A and low-weight E, so at weight 0.01 this is ordinary OPD coordinate drift rather than a lens-specific failure.

At weight 0.1, relative matrix drift rises from about 19% to 21% into 27% to 29% at layer 12, and from about 11% to 12% into 19% at layer 18. The stale lens's finite-difference cosine correspondingly falls to about 0.90 and 0.96. This is a real high-pressure staleness effect, but not catastrophic loss of fidelity. Live and frozen runs are nearly identical, so readout motion is not its cause. Staleness is best treated as a weight-dependent feedback amplifier, not a complete explanation of collapse.

Raw reports:

`/gdata/users/duanyll/opdlens/lens-audit/fidelity/`

## 8. Frozen-readout training ablation

The experimental `student_readout` knob accepts `live` or `frozen`. Frozen mode snapshots the student's step-0 final norm and LM head, disables gradients to the snapshot parameters, and still preserves the lens gradient into intermediate hidden states. The normal endpoint forward and base loss continue to use the live model.

Jobs:

| Job | Arm | Readout | Aux weight | Status |
|---:|---|---|---:|---|
| historical | B | live | 0.1 | complete |
| 4466 | B | frozen | 0.1 | complete |
| 4467 | C | live | 0.1 | complete |
| 4468 | C | frozen | 0.1 | complete |
| 4469 | E | live | 0.1 | complete |
| 4470 | E | frozen | 0.1 | complete |
| historical | B | live | 0.3 | complete |
| 4493 | B | frozen | 0.3 | complete |

The jobs retain the original evaluation path and differ only in arm type, auxiliary weight, readout mode, experiment identity, and checkpoint path.

The initial evaluation itself varies by up to 0.8 points across independently launched jobs, despite an identical student and temperature-zero evaluation. Both absolute accuracy and change from each run's step-0 value are therefore shown below.

| Arm | Readout | Step 0 | Step 50 | Step 100 | Step 200 | Step 300 | Change, 0 to 300 |
|---|---|---:|---:|---:|---:|---:|---:|
| A baseline, historical | none | 0.749 | 0.816 | 0.817 | 0.835 | 0.836 | +0.087 |
| B, weight 0.1, historical | live | 0.748 | 0.758 | 0.779 | 0.792 | 0.798 | +0.051 |
| B, weight 0.1 | frozen | 0.757 | 0.768 | 0.791 | 0.798 | 0.798 | +0.042 |
| C, weight 0.1 | live | 0.749 | 0.754 | 0.774 | 0.813 | 0.816 | +0.067 |
| C, weight 0.1 | frozen | 0.757 | 0.745 | 0.761 | 0.797 | 0.816 | +0.059 |
| E, weight 0.1 | live | 0.757 | 0.766 | 0.788 | 0.818 | 0.828 | +0.071 |
| E, weight 0.1 | frozen | 0.757 | 0.792 | 0.780 | 0.827 | 0.826 | +0.069 |
| B, weight 0.3, historical | live | 0.748 | 0.698 | 0.735 | 0.766 | 0.779 | +0.031 |
| B, weight 0.3 | frozen | 0.749 | 0.740 | 0.732 | 0.780 | 0.769 | +0.020 |

Freezing does not rescue any final endpoint. B at weight 0.1 and C finish at exactly the same accuracy as their live controls. E differs by only three GSM8K examples. At weight 0.3, freezing B avoids most of the severe step-50 drop and is 1.4 points ahead at step 200, but finishes 1.0 point below live. This is a transient trajectory change, not a stable cure. Every condition remains below the standard A endpoint.

Across all 300 common train steps, frozen minus live mean base-loss difference is +0.00156 for C and -0.00004 for E. Mean auxiliary loss is 0.141 higher for both frozen runs. The live head is therefore an easy route for reducing probe loss, but closing it leaves base optimization essentially unchanged and does not improve final task behavior.

The result narrows the causal claim: direct tied-codebook gradients are formally undesirable and dominate coordinate counts, but they carry little of the auxiliary L2 energy and are not the main source of collapse. The backbone objective mismatch survives an exactly frozen observer.

## 9. Working interpretation

The combined gradient is $g=g_{\mathrm{base}}+\lambda g_{\mathrm{aux}}$. None of the audited runs shows an arithmetic explosion: losses and gradient norms remain finite, and the global base/combined cosine remains high. The failure is geometric and functional rather than a floating-point overflow.

For a hidden direction $v$, define its local endpoint gain as $G_l(v)=D_{\mathrm{KL}}(p(h_l)\|p(h_l+\epsilon v))/\epsilon^2$. The measured $G_l$ for lens-gradient directions is two to three orders of magnitude below the base-gradient direction at the shallow layer. The auxiliary therefore spends update budget in directions that are easy for the probe to see but to which the actual downstream computation is nearly insensitive. Once weighted strongly enough, those directions take over a nontrivial fraction of backbone coordinates and reduce progress on the base objective without a proportionate endpoint effect.

Two invariance issues are separable:

1. **Fixed-observer invariance.** A lens intended as a measuring device should not change the model's vocabulary codebook. The live readout violates this because Qwen3.5 ties the LM head and input embedding. Freezing removes this violation exactly, but the training ablation shows that it is not the whole failure.
2. **Functional or gauge invariance.** A hidden-coordinate change $h_l\mapsto h_lR$ can be canceled by an inverse change in the downstream map without changing the model's endpoint function. A fixed lens does not co-transform, so its loss assigns a cost to functionally equivalent internal coordinates. The similar J-lens drift under A and low-weight E is direct evidence that ordinary OPD moves this coordinate system even without lens pressure.

The best-supported causal picture is therefore: live-readout codebook pollution is a real secondary defect; J-lens drift is background error at low weight and a feedback amplifier at high weight; and the remaining primary mismatch is that probe-space loss is not calibrated by downstream endpoint gain. Freezing only closes one shortcut; it cannot make the auxiliary objective functionally invariant.

## 10. Limits

- The gradient and endpoint audit is local: two fixed on-policy sequences and one checkpoint at a time. It does not simulate hundreds of optimizer steps.
- The endpoint test compares equal hidden RMS, not equal parameter-space optimizer steps.
- Each live/frozen training condition has one run. Differences near one accuracy point are within the observed launch-to-launch and evaluation variation.
- Fidelity is measured on held-out GSM8K prompts and random hidden directions; it does not establish fidelity on a pretraining-like concept distribution.
