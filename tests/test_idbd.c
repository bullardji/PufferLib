// Validates the update rule in src/idbd.cuh against the problems in
// Degris, Javed, Sharifnassab, Liu & Sutton, "Step-size Optimization for
// Continual Learning" (arXiv:2401.17401). Host-only arithmetic identical to the
// kernel, so the algorithm can be checked without a GPU. tests/test_idbd.cu
// checks the kernel agrees with this reference.
//
// Three experiments, each able to falsify a different claim:
//   1. weight-flipping — beat SGD and RMSProp, approach Oracle SGD.
//   2. scale shift     — the paper's open problem (their Fig 4: the best
//                        meta-step-size moves ~5 orders of magnitude with
//                        gradient scale). Does normalizing actually fix it?
//   3. rate tracking   — RMSProp fails there by moving the step size the wrong
//                        way. Our fix IS a normalizer, so this is the test that
//                        could sink it.
//
// DIVERGENCE FROM THE PAPER: the h decay uses g^2 where the paper uses x^2.
// idbd.cuh sits after Muon's orthogonalization and receives only the update
// tensor, never the per-weight input activation, so x^2 is not available to it.
// Build: cc -O2 tests/test_idbd.c -lm -o build_test_idbd
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_IN 20
#define N_ZERO 15          // first 15 target weights are constant 0
#define FLIP_EVERY 20      // one of the last 5 flips sign every 20 samples
#define STEPS 100000
#define TAU 10000.0f
#define BETA_MIN (-12.0f)
#define BETA_MAX 1.0f

static float gaussian(unsigned int* rng) {
    float u1 = (rand_r(rng) + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand_r(rng) + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

static void task_init(float* w_true, float mag) {
    for (int i = 0; i < N_ZERO; i++) {
        w_true[i] = 0.0f;
    }
    for (int i = N_ZERO; i < N_IN; i++) {
        w_true[i] = mag;
    }
}

static float task(unsigned int* rng, float* w_true, float* x, int tick) {
    if (tick % FLIP_EVERY == 0) {
        int j = N_ZERO + (int)(rand_r(rng) % (N_IN - N_ZERO));
        w_true[j] = -w_true[j];
    }
    float y = 0.0f;
    for (int i = 0; i < N_IN; i++) {
        x[i] = gaussian(rng);
        y += w_true[i] * x[i];
    }
    return y;
}

// oracle=1 pins the 15 constant weights at their optimal 0, so only the
// flipping ones learn: the best any constant step-size vector can do.
static float run_sgd(float alpha, unsigned int seed, float mag, int oracle) {
    unsigned int rng = seed;
    float w_true[N_IN], w[N_IN] = {0}, x[N_IN];
    task_init(w_true, mag);
    double err = 0.0;
    int lo = oracle ? N_ZERO : 0;
    for (int t = 0; t < STEPS; t++) {
        float y = task(&rng, w_true, x, t);
        float pred = 0.0f;
        for (int i = 0; i < N_IN; i++) {
            pred += w[i] * x[i];
        }
        float delta = y - pred;
        err += (double)delta * delta;
        for (int i = lo; i < N_IN; i++) {
            w[i] += alpha * delta * x[i];
        }
    }
    return (float)(err / STEPS);
}

// gamma_m = 1 disables momentum, which is RMSProp.
static float run_rmsprop(float eta, unsigned int seed, float mag, int gamma_idx) {
    const float gammas[] = {0.001f, 0.01f, 0.1f};
    float gamma_g = gammas[gamma_idx];
    unsigned int rng = seed;
    float w_true[N_IN], w[N_IN] = {0}, x[N_IN], g[N_IN];
    task_init(w_true, mag);
    for (int i = 0; i < N_IN; i++) {
        g[i] = 1.0f;
    }
    double err = 0.0;
    for (int t = 0; t < STEPS; t++) {
        float y = task(&rng, w_true, x, t);
        float pred = 0.0f;
        for (int i = 0; i < N_IN; i++) {
            pred += w[i] * x[i];
        }
        float delta = y - pred;
        err += (double)delta * delta;
        for (int i = 0; i < N_IN; i++) {
            float grad = -delta * x[i];
            g[i] = (1.0f - gamma_g) * g[i] + gamma_g * grad * grad;
            w[i] -= eta * grad / (sqrtf(g[i]) + 1e-8f);
        }
    }
    return (float)(err / STEPS);
}

// The exact arithmetic of idbd_weight_update. normalized=0 is plain IDBD
// (Sutton 1992); normalized=1 adds the Autostep v divisor that idbd.cuh ships.
static float run_idbd(float theta, unsigned int seed, float mag, int normalized,
        float* alpha_zero, float* alpha_flip) {
    unsigned int rng = seed;
    float w_true[N_IN], w[N_IN] = {0}, beta[N_IN], h[N_IN] = {0}, v[N_IN] = {0};
    float x[N_IN];
    task_init(w_true, mag);
    for (int i = 0; i < N_IN; i++) {
        beta[i] = logf(0.01f);
    }
    double err = 0.0;
    for (int t = 0; t < STEPS; t++) {
        float y = task(&rng, w_true, x, t);
        float pred = 0.0f;
        for (int i = 0; i < N_IN; i++) {
            pred += w[i] * x[i];
        }
        float delta = y - pred;
        err += (double)delta * delta;
        for (int i = 0; i < N_IN; i++) {
            float g = -delta * x[i];
            float gh = g * h[i];
            float alpha = expf(beta[i]);
            float b;
            if (normalized) {
                float mgh = fabsf(gh);
                v[i] = fmaxf(mgh, v[i] + (alpha * g * g / TAU) * (mgh - v[i]));
                b = beta[i] - theta * gh / (v[i] + 1e-8f);
            } else {
                b = beta[i] - theta * gh;
            }
            b = fminf(fmaxf(b, BETA_MIN), BETA_MAX);
            alpha = expf(b);
            float stp = alpha * g;
            beta[i] = b;
            w[i] -= stp;
            h[i] = h[i] * fmaxf(0.0f, 1.0f - alpha * g * g) - stp;
        }
    }
    float z = 0.0f, f = 0.0f;
    for (int i = 0; i < N_ZERO; i++) {
        z += expf(beta[i]);
    }
    for (int i = N_ZERO; i < N_IN; i++) {
        f += expf(beta[i]);
    }
    *alpha_zero = z / N_ZERO;
    *alpha_flip = f / (N_IN - N_ZERO);
    return (float)(err / STEPS);
}

static float sweep(float (*fn)(float, unsigned int, float, int),
        const float* grid, int n, float mag, int flag, float* argbest) {
    float best = 1e30f;
    for (int k = 0; k < n; k++) {
        double m = 0.0;
        for (unsigned int s = 1; s <= 3; s++) {
            m += fn(grid[k], s, mag, flag);
        }
        m /= 3.0;
        if (isfinite(m) && m < best) {
            best = (float)m;
            if (argbest) {
                *argbest = grid[k];
            }
        }
    }
    return best;
}

static const float THETA_GRID[] = {1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f, 10.0f};

static float best_theta(float mag, int normalized, float* out_mse) {
    float best = 1e30f, arg = 0.0f;
    for (int k = 0; k < 7; k++) {
        double m = 0.0;
        for (unsigned int s = 1; s <= 3; s++) {
            float az, af;
            m += run_idbd(THETA_GRID[k], s, mag, normalized, &az, &af);
        }
        m /= 3.0;
        if (isfinite(m) && m < best) {
            best = (float)m;
            arg = THETA_GRID[k];
        }
    }
    *out_mse = best;
    return arg;
}

// Paper eq. 2: optimal step size when the target drifts with variance sigma^2
// under unit observation noise.
static float optimal_alpha(float sigma) {
    float s2 = sigma * sigma;
    return (-s2 + sqrtf(s2 * s2 + 4.0f * s2)) / 2.0f;
}

// x is always 1 here, so a normalizer keyed on gradient magnitude risks moving
// the step size the wrong way, exactly as RMSProp does.
static float run_rate_tracking(float param, unsigned int seed, int method,
        float* mean_abs_log_ratio) {
    const int segment = 50000, segments = 8;
    unsigned int rng = seed;
    float w = 0.0f, beta = logf(0.1f), h = 0.0f, v = 0.0f, g2 = 1.0f;
    float z = 0.0f;
    double err = 0.0, logratio = 0.0;
    int counted = 0;
    for (int seg = 0; seg < segments; seg++) {
        float sigma = 3.0f * (rand_r(&rng) / (float)RAND_MAX);
        float target = optimal_alpha(sigma);
        for (int t = 0; t < segment; t++) {
            z += sigma * gaussian(&rng);
            float y = z + gaussian(&rng);
            float delta = y - w;
            err += (double)delta * delta;
            float g = -delta;
            if (method == 0) {
                g2 = 0.999f * g2 + 0.001f * g * g;
                w -= param * g / (sqrtf(g2) + 1e-8f);
            } else {
                float gh = g * h;
                float alpha = expf(beta);
                float mgh = fabsf(gh);
                v = fmaxf(mgh, v + (alpha * g * g / TAU) * (mgh - v));
                float b = beta - param * gh / (v + 1e-8f);
                b = fminf(fmaxf(b, BETA_MIN), BETA_MAX);
                alpha = expf(b);
                float stp = alpha * g;
                beta = b;
                w -= stp;
                h = h * fmaxf(0.0f, 1.0f - alpha * g * g) - stp;
                if (t > segment / 2) {
                    logratio += fabs(log((double)alpha / (target + 1e-6f)));
                    counted++;
                }
            }
        }
    }
    *mean_abs_log_ratio = counted ? (float)(logratio / counted) : 0.0f;
    return (float)(err / ((double)segment * segments));
}

int main(void) {
    const float mag = 1.0f;
    const float alphas[] = {0.001f, 0.003f, 0.01f, 0.03f, 0.1f, 0.3f};
    const float etas[] = {0.001f, 0.003f, 0.01f, 0.03f, 0.1f};
    float az, af;

    printf("== weight-flipping (paper spec: x~N(0,1), 15 constant + 5 flipping,\n");
    printf("   one flip per %d steps, %d steps, mean of 3 seeds) ==\n",
        FLIP_EVERY, STEPS);

    float a_sgd = 0.0f;
    float best_sgd = sweep(run_sgd, alphas, 6, mag, 0, &a_sgd);
    float best_oracle = sweep(run_sgd, alphas, 6, mag, 1, NULL);
    float best_rms = 1e30f;
    for (int gi = 0; gi < 3; gi++) {
        float e = sweep(run_rmsprop, etas, 5, mag, gi, NULL);
        if (e < best_rms) {
            best_rms = e;
        }
    }
    float mse_idbd;
    float th = best_theta(mag, 1, &mse_idbd);
    double zsum = 0.0, fsum = 0.0;
    for (unsigned int s = 1; s <= 3; s++) {
        run_idbd(th, s, mag, 1, &az, &af);
        zsum += az / 3.0;
        fsum += af / 3.0;
    }

    printf("  classic SGD     (best alpha=%.3f) mse=%8.4f\n", a_sgd, best_sgd);
    printf("  RMSProp         (best eta,gamma)  mse=%8.4f\n", best_rms);
    printf("  normalized IDBD (best theta=%-6.4g) mse=%8.4f\n", th, mse_idbd);
    printf("  oracle SGD      (15 pinned to 0)  mse=%8.4f\n", best_oracle);
    printf("  learned step: constant inputs %.5f, flipping inputs %.5f (%.0fx)\n",
        zsum, fsum, fsum / (zsum + 1e-9));

    printf("\n== meta-step-size scale sensitivity (paper Fig 4) ==\n");
    const float mags[] = {0.1f, 1.0f, 10.0f};
    float plain[3], norm[3], dump;
    for (int k = 0; k < 3; k++) {
        plain[k] = best_theta(mags[k], 0, &dump);
        norm[k] = best_theta(mags[k], 1, &dump);
        printf("  target weight %+6.1f : best theta  plain=%-9.4g normalized=%-9.4g\n",
            mags[k], plain[k], norm[k]);
    }
    float pmin = plain[0], pmax = plain[0], nmin = norm[0], nmax = norm[0];
    for (int k = 1; k < 3; k++) {
        pmin = fminf(pmin, plain[k]); pmax = fmaxf(pmax, plain[k]);
        nmin = fminf(nmin, norm[k]);  nmax = fmaxf(nmax, norm[k]);
    }
    float plain_shift = pmax / pmin;
    float norm_shift = nmax / nmin;
    printf("  spread of best theta over 100x gradient scale:"
        " plain=%.0fx normalized=%.0fx\n", plain_shift, norm_shift);

    printf("\n== 1D noisy rate-tracking (paper Fig 3) ==\n");
    float ratio = 0.0f, dummy = 0.0f;
    const float rt_grid[] = {0.001f, 0.01f, 0.1f, 0.3f, 1.0f};
    float e_rms = 1e30f, e_idbd = 1e30f;
    for (int k = 0; k < 5; k++) {
        float e = run_rate_tracking(rt_grid[k], 1, 0, &dummy);
        if (isfinite(e) && e < e_rms) {
            e_rms = e;
        }
        float r;
        float d = run_rate_tracking(rt_grid[k], 1, 1, &r);
        if (isfinite(d) && d < e_idbd) {
            e_idbd = d;
            ratio = r;
        }
    }
    printf("  RMSProp         (best param) mse=%10.4f\n", e_rms);
    printf("  normalized IDBD (best param) mse=%10.4f   mean |log(alpha/alpha*)| = %.3f\n",
        e_idbd, ratio);

    assert(mse_idbd < best_sgd && "idbd must beat the best swept fixed alpha");
    assert(mse_idbd < best_rms && "idbd must beat RMSProp");
    assert(fsum > zsum * 5.0 && "idbd must separate flipping from constant inputs");
    assert(norm_shift <= plain_shift
        && "normalizing must not worsen meta-step-size scale sensitivity");
    assert(e_idbd < e_rms && "normalized idbd must not inherit RMSProp's pathology");
    printf("\nok all idbd claims hold\n");
    return 0;
}
