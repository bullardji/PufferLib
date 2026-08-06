// Validates the update rule in src/idbd.cuh on the nonstationary tracking task
// from Sutton 1992. This is a numerics test, not a kernel test: it runs the
// identical arithmetic the kernel runs, on the host, so the algorithm can be
// checked without a GPU. tests/test_idbd.cu checks that the kernel agrees with
// this reference. The CONTRACT comment in idbd.cuh is the shared spec.
//
// Task: 20 binary inputs. 5 are relevant with weights that flip sign every 20
// examples; 15 are irrelevant with true weight 0. No single step size is right
// for both groups, which is the point.
// Build: cc -O2 tests/test_idbd.c -lm -o build_test_idbd
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_IN 20
#define N_RELEVANT 5
#define FLIP_EVERY 20
#define WARMUP 20000
#define MEASURE 20000

static float task_next(unsigned int* rng, float* w_true, float* x, int tick) {
    if (tick % FLIP_EVERY == 0) {
        for (int i = 0; i < N_RELEVANT; i++) {
            w_true[i] = (rand_r(rng) % 2) ? 1.0f : -1.0f;
        }
    }
    float y = 0.0f;
    for (int i = 0; i < N_IN; i++) {
        x[i] = (rand_r(rng) % 2) ? 1.0f : -1.0f;
        y += w_true[i] * x[i];
    }
    return y;
}

// Fixed step size: the baseline a sweep would find.
static float run_fixed(float alpha, unsigned int seed) {
    unsigned int rng = seed;
    float w_true[N_IN] = {0}, w[N_IN] = {0}, x[N_IN];
    double err = 0.0;
    for (int step = 0; step < WARMUP + MEASURE; step++) {
        float y = task_next(&rng, w_true, x, step);
        float pred = 0.0f;
        for (int i = 0; i < N_IN; i++) {
            pred += w[i] * x[i];
        }
        float delta = y - pred;
        if (step >= WARMUP) {
            err += (double)delta * delta;
        }
        for (int i = 0; i < N_IN; i++) {
            w[i] += alpha * delta * x[i];
        }
    }
    return (float)(err / MEASURE);
}

// The exact arithmetic of idbd_weight_update, with g = -delta*x_i (the gradient
// of 0.5*delta^2) standing in for Muon's orthogonalized update.
static float run_idbd(float theta, unsigned int seed,
        float* alpha_relevant, float* alpha_irrelevant) {
    const float tau = 10000.0f, beta_min = -12.0f, beta_max = 0.0f;
    unsigned int rng = seed;
    float w_true[N_IN] = {0}, w[N_IN] = {0}, beta[N_IN], h[N_IN] = {0};
    float v[N_IN] = {0}, x[N_IN];
    for (int i = 0; i < N_IN; i++) {
        beta[i] = logf(0.01f);
    }
    double err = 0.0;
    for (int step = 0; step < WARMUP + MEASURE; step++) {
        float y = task_next(&rng, w_true, x, step);
        float pred = 0.0f;
        for (int i = 0; i < N_IN; i++) {
            pred += w[i] * x[i];
        }
        float delta = y - pred;
        if (step >= WARMUP) {
            err += (double)delta * delta;
        }
        for (int i = 0; i < N_IN; i++) {
            float g = -delta * x[i];
            float gh = g * h[i];
            float mag = fabsf(gh);
            float alpha = expf(beta[i]);
            float vi = fmaxf(mag, v[i] + (alpha * g * g / tau) * (mag - v[i]));
            float b = beta[i] - theta * gh / (vi + 1e-8f);
            b = fminf(fmaxf(b, beta_min), beta_max);
            alpha = expf(b);
            float stp = alpha * g;
            v[i] = vi;
            beta[i] = b;
            w[i] -= stp;
            h[i] = h[i] * fmaxf(0.0f, 1.0f - alpha * g * g) - stp;
        }
    }
    float rel = 0.0f, irr = 0.0f;
    for (int i = 0; i < N_RELEVANT; i++) {
        rel += expf(beta[i]);
    }
    for (int i = N_RELEVANT; i < N_IN; i++) {
        irr += expf(beta[i]);
    }
    *alpha_relevant = rel / N_RELEVANT;
    *alpha_irrelevant = irr / (N_IN - N_RELEVANT);
    return (float)(err / MEASURE);
}

int main(void) {
    const float alphas[] = {0.001f, 0.003f, 0.01f, 0.02f, 0.03f, 0.05f};
    float best_fixed = 1e30f, best_alpha = 0.0f;

    printf("fixed step size (asymptotic MSE, mean over 5 seeds)\n");
    for (int a = 0; a < 6; a++) {
        double mse = 0.0;
        for (unsigned int s = 1; s <= 5; s++) {
            mse += run_fixed(alphas[a], s);
        }
        mse /= 5.0;
        printf("  alpha=%-6.3f  mse=%8.4f\n", alphas[a], mse);
        if (mse < best_fixed) {
            best_fixed = (float)mse;
            best_alpha = alphas[a];
        }
    }

    // theta spans two orders of magnitude. Flatness across it is the claim:
    // theta is only worth having if it does not need the sweep lr needed.
    const float thetas[] = {0.01f, 0.05f, 0.1f, 0.5f};
    float best_idbd = 1e30f, worst_idbd = 0.0f, best_theta = 0.0f;
    float rel_at_best = 0.0f, irr_at_best = 0.0f;

    printf("\nidbd (asymptotic MSE, mean over 5 seeds)\n");
    for (int k = 0; k < 4; k++) {
        double mse = 0.0, rel = 0.0, irr = 0.0;
        for (unsigned int s = 1; s <= 5; s++) {
            float r, i2;
            mse += run_idbd(thetas[k], s, &r, &i2);
            rel += r;
            irr += i2;
        }
        mse /= 5.0; rel /= 5.0; irr /= 5.0;
        printf("  theta=%-6.3f  mse=%8.4f   alpha_relevant=%.5f  alpha_irrelevant=%.5f\n",
            thetas[k], mse, rel, irr);
        assert(isfinite(mse) && "idbd diverged");
        if (mse < best_idbd) {
            best_idbd = (float)mse;
            best_theta = thetas[k];
            rel_at_best = (float)rel;
            irr_at_best = (float)irr;
        }
        if (mse > worst_idbd) {
            worst_idbd = (float)mse;
        }
    }

    printf("\nbest fixed alpha=%.3f mse=%.4f | best idbd theta=%.3f mse=%.4f\n",
        best_alpha, best_fixed, best_theta, best_idbd);
    printf("idbd spread across theta 0.01-0.5: %.1f%%\n",
        100.0f * (worst_idbd - best_idbd) / best_idbd);

    assert(best_idbd < best_fixed && "idbd must beat the best swept fixed alpha");
    assert(rel_at_best > irr_at_best * 10.0f
        && "idbd must learn a much larger step for relevant inputs");
    assert(worst_idbd < 1.25f * best_idbd
        && "idbd must be insensitive to theta, or it has not replaced the lr sweep");
    printf("ok idbd beats best fixed alpha, separates inputs, insensitive to theta\n");
    return 0;
}
