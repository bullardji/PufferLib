// Compiles src/idbd.cuh against the core's type shapes and checks the kernel
// against the host reference in tests/test_idbd.c. Two jobs, because without a
// GPU only the first one runs:
//
//   nvcc -arch=sm_80 -c tests/test_idbd.cu -o /dev/null   # compile check
//   nvcc -arch=sm_80 tests/test_idbd.cu -o build_test_idbd_cu && ./...  # parity
//
// The type declarations below mirror src/pufferl.cu (Float, Prec, Allocator,
// grid_size, BLOCK_SIZE, to_float). They are a stand-in so this file builds
// without cublas/nccl/nvml, not a second definition to maintain: idbd.cuh is
// included by algo.cu in a real build and gets the originals there.
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PUF_MAX_DIMS 8
#define BLOCK_SIZE 256

typedef float precision_t;
__host__ __device__ static inline float to_float(precision_t x) { return x; }

static inline int grid_size(int N) {
    return (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

typedef struct {
    float* data;
    int64_t shape[PUF_MAX_DIMS];
} Float;

typedef struct {
    precision_t* data;
    int64_t shape[PUF_MAX_DIMS];
} Prec;

typedef struct {
    long total_elems;
} Allocator;

static inline long numel(int64_t* shape) {
    long n = 1;
    for (int i = 0; i < PUF_MAX_DIMS && shape[i] != 0; i++) {
        n *= shape[i];
    }
    return n;
}

// Real allocator registers into an arena; here each tensor gets its own device
// buffer, which is all the kernel cares about.
static void alloc_register(Allocator* a, Float* t) {
    cudaMalloc((void**)&t->data, numel(t->shape) * sizeof(float));
    (void)a;
}

#include "idbd.cuh"

// Same arithmetic as idbd_weight_update, in plain host C.
static void host_update(float* wb, const float* update, float* beta, float* h,
        float* v, float theta, float tau, float beta_min, float beta_max, int n) {
    for (int i = 0; i < n; i++) {
        float g = update[i];
        float gh = g * h[i];
        float mag = fabsf(gh);
        float alpha = expf(beta[i]);
        float vi = fmaxf(mag, v[i] + (alpha * g * g / tau) * (mag - v[i]));
        float b = beta[i] - theta * gh / (vi + 1e-8f);
        b = fminf(fmaxf(b, beta_min), beta_max);
        alpha = expf(b);
        float step = alpha * g;
        v[i] = vi;
        beta[i] = b;
        wb[i] -= step;
        h[i] = h[i] * fmaxf(0.0f, 1.0f - alpha * g * g) - step;
    }
}

int main(void) {
    const int n = 4096;
    const int steps = 200;
    const float theta = 0.05f, tau = 10000.0f, bmin = -12.0f, bmax = 0.0f;

    Allocator alloc = {n};
    Idbd d;
    idbd_init(&d, &alloc, &alloc, theta, tau, bmin, bmax);

    float* h_w = (float*)malloc(n * sizeof(float));
    float* h_g = (float*)malloc(n * sizeof(float));
    float* h_beta = (float*)malloc(n * sizeof(float));
    float* h_h = (float*)calloc(n, sizeof(float));
    float* h_v = (float*)calloc(n, sizeof(float));
    float* out = (float*)malloc(n * sizeof(float));

    unsigned int rng = 7;
    for (int i = 0; i < n; i++) {
        h_w[i] = (rand_r(&rng) / (float)RAND_MAX) - 0.5f;
        h_beta[i] = logf(0.01f);
    }

    Float weights = {.shape = {n}};
    Prec update = {.shape = {n}};
    cudaMalloc((void**)&weights.data, n * sizeof(float));
    cudaMalloc((void**)&update.data, n * sizeof(precision_t));
    cudaMemcpy(weights.data, h_w, n * sizeof(float), cudaMemcpyHostToDevice);
    idbd_reset(&d, 0.01f, 0);
    cudaMemset(d.h.data, 0, n * sizeof(float));
    cudaMemset(d.v.data, 0, n * sizeof(float));

    for (int s = 0; s < steps; s++) {
        for (int i = 0; i < n; i++) {
            h_g[i] = 2.0f * (rand_r(&rng) / (float)RAND_MAX) - 1.0f;
        }
        cudaMemcpy(update.data, h_g, n * sizeof(precision_t), cudaMemcpyHostToDevice);
        idbd_step(&d, weights, update, n, 0);
        host_update(h_w, h_g, h_beta, h_h, h_v, theta, tau, bmin, bmax, n);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(out, weights.data, n * sizeof(float), cudaMemcpyDeviceToHost);

    // __expf is a fast-math intrinsic, so device and host agree to within a
    // relative tolerance, not bitwise.
    float worst = 0.0f;
    for (int i = 0; i < n; i++) {
        float denom = fmaxf(1e-6f, fabsf(h_w[i]));
        worst = fmaxf(worst, fabsf(out[i] - h_w[i]) / denom);
    }
    printf("max relative deviation after %d steps: %.3e\n", steps, worst);
    if (!(worst < 1e-3f)) {
        printf("FAIL device/host parity\n");
        return 1;
    }
    printf("ok idbd device matches host reference\n");
    return 0;
}
