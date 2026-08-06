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

`tests/test_idbd.c`, Sutton's nonstationary tracking task — 20 binary inputs, 5
relevant with weights flipping every 20 examples, 15 irrelevant. Asymptotic MSE,
mean of 5 seeds:

| | MSE |
|---|---|
| best fixed step size (swept over 6 values) | 5.01 |
| IDBD, theta = 0.05 | **2.58** |

It gets there by separating the groups: step size 0.189 on the relevant inputs
against 0.001 on the irrelevant ones, a 180x ratio no single alpha can express.

The number that matters more is the spread: **7.6% across theta from 0.01 to
0.5**. A knob that still needs a 50x sweep has not replaced lr, it has renamed
it. This one does not.

## Why the normalizer

Plain IDBD — no `v_i` — reaches MSE 2.72 but goes NaN at theta >= 0.05, and a
sweep over theta would find that cliff immediately. Two fixes were measured:

- Cap `alpha_i` at `1/g_i^2`: stable everywhere, but MSE 4.16. It throttles the
  step exactly when the error is large and fast tracking is wanted, giving back
  most of the benefit.
- Normalize the meta-update by `v_i`, a decaying max of `|g h|` (Autostep,
  Mahmood et al. 2012): stable to theta = 0.5, MSE 2.58.

The second is in. theta multiplies a quantity whose scale is the gradient's;
dividing that out is what makes the knob insensitive.

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
env is unmeasured — I had no GPU. The tracking task is the setting IDBD was
designed for and says the arithmetic is right; it says nothing about how this
behaves on a MinGRU policy at 3M SPS. Two things to watch there: `g` is an
orthogonalized direction whose scale Muon already controls, so the curvature
proxy `g^2` may behave differently than it does on raw LMS gradients; and three
extra param-sized buffers plus an elementwise pass cost bandwidth on a kernel
that is already memory-bound.

The first experiment is the cheap one: `idbd_theta = 0` against a swept lr on
the same wallclock budget, on one Ocean env, checking whether the flatness in
theta survives.
