#pragma once

/*
 * IDBD (Sutton 1992, "Adapting Bias by Gradient Descent") as an alternative
 * tail for muon_step. Everything before the weight update — momentum, Nesterov,
 * Newton-Schulz orthogonalization — is unchanged; this replaces only the final
 *
 *     wb = wb * (1 - lr*wd) - lr * update
 *
 * with a per-weight step size that is itself learned by gradient descent on the
 * same objective. muon_weight_update stays; this is a second tail, selected by
 * train.idbd_theta > 0.
 *
 * WHY THIS ONE: "you always sweep learning rate", and lr is the most expensive
 * dimension in every Protein sweep. IDBD moves it inside the agent, which is
 * also the only way to get it at runtime rather than design time — Sutton's
 * point that metalearning requires trying one way of learning, then another,
 * and keeping the better. He names IDBD as likely key to the continual-learning
 * problem that blocks OaK step 1.
 *
 * The update, with g the incoming per-weight update (Muon's orthogonalized
 * direction, not the raw gradient):
 *
 *     v_i    <- max(|g_i h_i|, v_i + (1/tau) * alpha_i * g_i^2 * (|g_i h_i| - v_i))
 *     beta_i <- clamp(beta_i - theta * g_i * h_i / v_i)
 *     alpha_i = exp(beta_i)
 *     w_i    <- w_i - alpha_i * g_i
 *     h_i    <- h_i * max(0, 1 - alpha_i * g_i^2) - alpha_i * g_i
 *
 * h_i tracks dw_i/dbeta_i. Sutton's LMS form uses the input x_i^2 as the
 * curvature proxy in the h decay; g_i^2 is the standard generalization when
 * there is no single input per weight.
 *
 * DECISION: the meta-update is normalized by v_i, a decaying max of |g h|
 * (Autostep, Mahmood et al. 2012). Plain IDBD diverged to NaN at theta >= 0.05
 * on the tracking task in tests/test_idbd.c, because theta multiplies a
 * quantity whose scale is the gradient's, and a sweep over theta would find
 * that cliff. Capping alpha by 1/g^2 instead also stabilizes it, but costs
 * roughly half the benefit (MSE 4.16 vs 2.58) because it throttles alpha
 * exactly when the error is large and fast tracking is wanted. Normalizing
 * makes the meta-update scale-free: measured MSE is flat within 8% for theta
 * across 0.01 to 0.5, which is the property that lets theta replace lr in a
 * sweep rather than just rename it.
 *
 * DECISION: beta is clamped, not the step size. Clamping in log space bounds
 * the ratio between the fastest and slowest weight, which is the quantity that
 * blows up, and costs one fminf/fmaxf. beta_max = 0 means no weight ever
 * exceeds step size 1.
 *
 * COUPLING: beta is initialized to log(lr) at muon_init, so theta = 0 and
 * theta > 0 start from the same place and a sweep over theta stays comparable
 * to the fixed-lr baseline.
 */

struct Idbd {
    float theta;      // meta step size; 0 disables and muon_weight_update runs
    float tau;        // v_i averaging window
    float beta_min;
    float beta_max;
    Float beta;       // log per-weight step size, param-sized
    Float h;          // decaying trace of dw/dbeta, param-sized
    Float v;          // decaying max of |g h|, param-sized
};

__global__ void idbd_fill(float* __restrict__ dst, float value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = value;
    }
}

__global__ void idbd_weight_update(float* __restrict__ wb,
        const precision_t* __restrict__ update,
        float* __restrict__ beta, float* __restrict__ h, float* __restrict__ v,
        float theta, float tau, float beta_min, float beta_max, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float g = to_float(update[idx]);
    float gh = g * h[idx];
    float mag = fabsf(gh);
    float alpha = __expf(beta[idx]);
    float vi = fmaxf(mag, v[idx] + (alpha * g * g / tau) * (mag - v[idx]));
    float b = beta[idx] - theta * gh / (vi + 1e-8f);
    b = fminf(fmaxf(b, beta_min), beta_max);
    alpha = __expf(b);
    float step = alpha * g;
    v[idx] = vi;
    beta[idx] = b;
    wb[idx] -= step;
    h[idx] = h[idx] * fmaxf(0.0f, 1.0f - alpha * g * g) - step;
}

// beta starts at log(lr) so theta=0 and theta>0 begin identically.
void idbd_init(Idbd* d, Allocator* param_alloc, Allocator* alloc,
        float theta, float tau, float beta_min, float beta_max) {
    d->theta = theta;
    d->tau = tau;
    d->beta_min = beta_min;
    d->beta_max = beta_max;
    d->beta = {.shape = {param_alloc->total_elems}};
    d->h = {.shape = {param_alloc->total_elems}};
    d->v = {.shape = {param_alloc->total_elems}};
    alloc_register(alloc, &d->beta);
    alloc_register(alloc, &d->h);
    alloc_register(alloc, &d->v);
}

void idbd_reset(Idbd* d, float lr, cudaStream_t stream) {
    int n = (int)numel(d->beta.shape);
    idbd_fill<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(d->beta.data, logf(lr), n);
    idbd_fill<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(d->h.data, 0.0f, n);
    idbd_fill<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(d->v.data, 0.0f, n);
}

void idbd_step(Idbd* d, Float weights, Prec update, int n, cudaStream_t stream) {
    idbd_weight_update<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
        weights.data, update.data, d->beta.data, d->h.data, d->v.data,
        d->theta, d->tau, d->beta_min, d->beta_max, n);
}
