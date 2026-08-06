# OptionPuffer: IDBD

Sutton's IDBD (1992) as an additive tail for `muon_step`, plus the Autostep
normalizer that makes it survive a sweep.

## Why this piece of OaK

OaK's step 1 — learn the policy and value function — is the one Sutton marks as
"would be done if we could do continual deep learning and metalearning," and he
names IDBD as likely part of the answer. It is also the only piece of OaK that
pays for itself immediately under Puffer's own priorities: *you always sweep
learning rate*, lr is the most expensive dimension in every Protein sweep, and
IDBD moves it inside the agent. It is metalearning in Sutton's sense — trying
one way of learning, then another, and keeping the better — which he argues can
only happen at runtime.

The rest of OaK is not here. Options need decoder changes, feature discovery
needs the consumers working first, and planning trades sample efficiency for
compute, which a 3M SPS simulator makes a bad trade.

## What it changes

Nothing before the weight update. Momentum, Nesterov and Newton-Schulz
orthogonalization all run as they do now. Only the final line

```
wb = wb * (1 - lr*wd) - lr * update
```

gets a per-weight step size that is itself learned. `muon_weight_update` is
untouched; `idbd_weight_update` is a second tail.

## The update

```
v_i    <- max(|g_i h_i|, v_i + (1/tau) * alpha_i * g_i^2 * (|g_i h_i| - v_i))
beta_i <- clamp(beta_i - theta * g_i * h_i / v_i)
alpha_i = exp(beta_i)
w_i    <- w_i - alpha_i * g_i
h_i    <- h_i * max(0, 1 - alpha_i * g_i^2) - alpha_i * g_i
```

`g` is Muon's orthogonalized update, not the raw gradient. `h_i` tracks
`dw_i/dbeta_i`. Three param-sized float buffers (`beta`, `h`, `v`) alongside
Muon's existing momentum buffer.

## Measured

`tests/test_idbd.c` reproduces the three problems from Degris, Javed,
Sharifnassab, Liu & Sutton, *Step-size Optimization for Continual Learning*
(arXiv:2401.17401), to their spec: inputs `x ~ N(0,1)^20`, 15 target weights
constant at 0, 5 at +/-1, and one of the five flipping sign every 20 samples.
100k steps, mean of 3 seeds, every method swept over its own parameters.

### 1. Weight-flipping

| | MSE |
|---|---|
| classic SGD (best alpha) | 3.37 |
| RMSProp (best eta, gamma) | 3.75 |
| **normalized IDBD (best theta)** | **1.42** |
| oracle SGD — 15 weights pinned to 0 | 1.46 |

This reproduces the paper's Figure 2, including the part that looks like a
mistake: RMSProp is *worse* than plain SGD. All components of `x` share a
variance and the error is global, so normalization hands every weight the same
step size and cannot tell a constant weight from a flipping one. IDBD reaches
oracle performance by learning step size 0.126 on the flipping inputs against
0.0003 on the constant ones, a 416x ratio. It edges past the oracle because the
oracle is the best *constant* step-size vector and IDBD's varies over time.

### 2. Meta-step-size scale sensitivity

The paper's stated open problem (their Figure 4): IDBD's meta-step-size is
sensitive to gradient magnitude, and the best value moved ~5 orders of magnitude
across target-weight scales, which "makes it difficult to use IDBD in many
common settings". Best theta over a decade grid, target weights at +/-0.1, +/-1,
+/-10:

| | spread in best theta over 100x gradient scale |
|---|---|
| plain IDBD | 1000x |
| normalized IDBD | 10x — one grid step |

That is the whole reason the normalizer is in. theta multiplies a quantity whose
scale is the gradient's; dividing it out is what lets theta replace lr in a
sweep instead of renaming it.

### 3. 1D noisy rate-tracking — the test that could have sunk this

The risk in fixing IDBD with a normalizer is that normalization is exactly what
makes RMSProp fail the paper's rate-tracking problem. There `x = 1` always, so a
larger error means faster target drift and calls for a *larger* step; RMSProp
reads the larger gradient and shrinks the step instead, doing precisely the
opposite of what is needed. A normalized IDBD could plausibly inherit that.

It does not. Both swept, 8 segments of 50k steps with sigma redrawn from U(0,3):

| | MSE | mean \|log(alpha/alpha\*)\| |
|---|---|---|
| RMSProp | 6.34 | — |
| normalized IDBD | 4.19 | 0.031 |

alpha stays within about 3% of the closed-form optimum (paper eq. 2). The
normalizer divides by a decaying max of `\|g h\|`, and `h` carries the sign
correlation that tells fast drift apart from noise, so the scale is removed
without removing the signal RMSProp discards.

## Wiring it up

`src/idbd.cuh` is standalone and is not included by anything yet, so this branch
changes no existing file. Three lines turn it on, in `src/algo.cu` beside the
Muon include and in `muon_step`:

```c
#include "idbd.cuh"                                  // beside the Muon code

idbd_init(&m->idbd, param_alloc, alloc,              // in muon_init
    puf_ini_get(ini, "train", "idbd_theta"), 10000.0f, -12.0f, 0.0f);

if (m->idbd.theta > 0.0f) {                          // replacing the tail call
    idbd_step(&m->idbd, weights, update, n_grad, stream);
} else {
    muon_weight_update<<<...>>>(...);                // unchanged
}
```

`beta` is initialized to `log(lr)`, so `idbd_theta = 0` and `idbd_theta > 0`
start from the same weights and a sweep over theta stays comparable to the fixed
-lr baseline.

## Verified

- `tests/test_idbd.c` — numerics above, all assertions passing.
- `tests/test_idbd.cu` — kernel compiles under nvcc 12.8 for sm_80: 25
  registers, 0 spill stores, 0 spill loads.

## Not verified

**No training run.** Whether IDBD beats a CARBS/Protein-swept lr on a real Ocean
env is unmeasured — I had no GPU. These are the problems IDBD was designed for
and they say the arithmetic is right; they say nothing about a MinGRU policy at
3M SPS. Three things to watch there:

- All three problems are linear regression with one input per weight. The paper
  is explicit that generalizing IDBD to deep networks is open.
- The h decay uses `g^2` where the paper uses `x^2`. `idbd.cuh` sits after Muon's
  orthogonalization and never sees the per-weight input activation, so `x^2` is
  not available to it. `g` is also already scale-controlled by Muon, so the
  curvature proxy may behave differently than on raw LMS gradients. If IDBD
  underperforms on a real env, this substitution is the first suspect.
- Three extra param-sized buffers and an elementwise pass cost bandwidth on a
  kernel that is already memory-bound.

The first experiment is the cheap one: `idbd_theta = 0` against a swept lr on
the same wallclock budget, on one Ocean env, checking whether the flatness in
theta survives.
