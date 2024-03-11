#include <assert.h>
#include <pthread.h>
#include <stdio.h>

#include <cmath>
#include <cstdlib>

#include "../matmul.h"
#include "common.h"

#ifdef QM_ARM
#include <arm_neon.h>
#endif
#ifdef QM_x86
#include <immintrin.h>
#endif
struct w4a8_thread_args {
    int start_j, end_j;
    const struct matmul_params *params;
};
static void *all_techniques_worker_func(void *args) {
    struct w4a8_thread_args *mat_args = (struct w4a8_thread_args *)args;
    const struct matmul_params *params = mat_args->params;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    int n = params->C.column, m = params->C.row, k = params->A.column, block_size = params->block_size;
    const int num_block = k / block_size;  // block_size = 32

    for (int row = 0; row < m; row++) {
        for (int col = mat_args->start_j; col < mat_args->end_j; col++) {

#ifdef QM_x86
            // order of weights with QM_x86:
            // origin order: (w0,w1), (w2,w3), (w4,w5), (w6,w7), (w8, w9), ... (w62,w63)
            // QM_x86 order: (w0,w32),(w1,w33),(w2,w34),(w3,w35),(w4, w36),... (w31,w63)
            //               |--|
            //               4 bits
            //               |------|
            //               8 bits (byte)
            //            low|----------------------------------------------------------|high
            //               0                         256 bit
            __m256 accumulator = _mm256_setzero_ps();
            float *s_ptr = &params->scales[col * k / 32];
            float *sa_ptr = &params->A_scales[row * k / 32];
            const __m256i *w_start = (__m256i *)&B->int4_data_ptr[col * k / 2];
            const __m256i *a_start0 = (__m256i *)&A->int8_data_ptr[row * k];
            const int num_block = k / block_size;
            // Compute four blocks = 128 4-bit weights in each iteration
            for (int q = 0; q < num_block; q += 4) {
                // lowbit mask
                const __m256i lowMask = _mm256_set1_epi8(0xF);

                // TODO: Unpack 128 4-bit (two __mm256i) weights into 128 8-bit (four __mm256i)
                // (1) load 256 bit from w_strat with _mm256_loadu_si256
                // (2) use _mm256_and_si256 and lowMask to extract the lower half of wegihts
                // (3) use _mm256_srli_epi16 and _mm256_and_si256 with lowMask to extract the upper half of weights
                __m256i raw_w = _mm256_loadu_si256(w_start);
                __m256i raw_w_next = _mm256_loadu_si256(w_start + 1);

                // for raw_w
                __m256i low_half_weight = _mm256_and_si256(raw_w, lowMask);
                __m256i shifted_w = _mm256_srli_epi16(raw_w, 4);
                __m256i high_half_weight = _mm256_and_si256(shifted_w, lowMask);

                // for raw_w_next
                __m256i low_half_weight_next = _mm256_and_si256(raw_w_next, lowMask);
                __m256i shifted_w_next = _mm256_srli_epi16(raw_w_next, 4);
                __m256i high_half_weight_next = _mm256_and_si256(shifted_w_next, lowMask);

                // TODO: apply zero_point to weights and convert the range from (0, 15) to (-8, 7)
                // Hint: using `_mm256_sub_epi8` to the lower-half and upper-half vectors of weights
                // Note: For the first two blocks, store the lower half and upper half of weights into `w_0` and
                // `w_128`, respectively For the last two blocks store the lower half and upper half of weights into
                // `w_0_next` and `w_128_next`, respectively
                const __m256i zero_point = _mm256_set1_epi8(8);
                __m256i w_0, w_128, w_0_next, w_128_next;

                w_0 = _mm256_sub_epi8(low_half_weight, zero_point);
                w_128 = _mm256_sub_epi8(high_half_weight, zero_point);

                w_0_next = _mm256_sub_epi8(low_half_weight_next, zero_point);
                w_128_next = _mm256_sub_epi8(high_half_weight_next, zero_point);

                // Perform int8 dot product with _mm256_maddubs_epi16
                /* Syntax of _mm256_maddubs_epi16:
                   __m256i _mm256_maddubs_epi16(__m256i s1, __m256i s2): Multiplies vertically each unsigned byte of
                   source vector s1 with the corresponding signed byte of source vector s2, producing intermediate,
                   signed 16-bit integers. Each adjacent pair of signed words is added, and the saturated result is
                   packed to the destination vector.
                */
                // To utilize _mm256_maddubs_epi16 which only takes unsigned s1, we need to:
                // (1) Get the absolute values of weights (for both lower and upper halves)
                // (2) Change the sign of activation (a0-a31 and a32-a63) depending on the sign of corresponding weights
                // (stored as another variable) (3) Perform dot product with _mm256_maddubs_epi16 and store the lower
                // and upper halves sum in `dot` and `dot2`
                __m256i dot, dot2, dot3, dot4;
                // Get absolute values of x vectors
                const __m256i ax = _mm256_sign_epi8(w_0, w_0);
                const __m256i ax_next = _mm256_sign_epi8(w_0_next, w_0_next);
                const __m256i ax2 = _mm256_sign_epi8(w_128, w_128);
                const __m256i ax2_next = _mm256_sign_epi8(w_128_next, w_128_next);
                // Load activation
                __m256i activation = a_start[0];
                __m256i activation2 = a_start[1];
                __m256i activation_next = a_start[2];
                __m256i activation2_next = a_start[3];
                // Sign the values of the y vectors
                const __m256i sy = _mm256_sign_epi8(activation, w_0);
                const __m256i sy_next = _mm256_sign_epi8(activation_next, w_0_next);
                const __m256i sy2 = _mm256_sign_epi8(activation2, w_128);
                const __m256i sy2_next = _mm256_sign_epi8(activation2_next, w_128_next);

                // TODO: Perform int8 dot product with `_mm256_maddubs_epi16`
                // Hint: use `_mm256_maddubs_epi16` to complete the following computation
                // dot = ax * sy
                // dot2 = ax2 * sy2
                // dot3 = ax_next * sy_next
                // dot4 = ax2_next * sy2_next
                dot = _mm256_maddubs_epi16(ax, sy);
                dot2 = _mm256_maddubs_epi16(ax2, sy2);
                dot3 = _mm256_maddubs_epi16(ax_next, sy_next);
                dot4 = _mm256_maddubs_epi16(ax2_next, sy2_next);

                // Convert int32 vectors to floating point vectors
                const __m256i ones = _mm256_set1_epi16(1);
                const __m256i summed_pairs = _mm256_madd_epi16(ones, dot);
                const __m256i summed_pairs2 = _mm256_madd_epi16(ones, dot2);
                const __m256i summed_pairs3 = _mm256_madd_epi16(ones, dot3);
                const __m256i summed_pairs4 = _mm256_madd_epi16(ones, dot4);
                __m256 intermediate = _mm256_cvtepi32_ps(summed_pairs);
                __m256 intermediate2 = _mm256_cvtepi32_ps(summed_pairs2);
                __m256 intermediate3 = _mm256_cvtepi32_ps(summed_pairs3);
                __m256 intermediate4 = _mm256_cvtepi32_ps(summed_pairs4);

                // Create vectors for scales and apply them to intermediate results
                __m256 v_s = _mm256_set1_ps(s_ptr[0] * sa_ptr[0]);
                __m256 v_s2 = _mm256_set1_ps(s_ptr[1] * sa_ptr[1]);
                __m256 v_s3 = _mm256_set1_ps(s_ptr[2] * sa_ptr[2]);
                __m256 v_s4 = _mm256_set1_ps(s_ptr[3] * sa_ptr[3]);
                accumulator = _mm256_fmadd_ps(intermediate, v_s, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate2, v_s2, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate3, v_s3, accumulator);
                accumulator = _mm256_fmadd_ps(intermediate4, v_s4, accumulator);
                s_ptr += 4;
                sa_ptr += 4;
                w_start += 2;
                a_start += 4;
            }
            float *ptr = (float *)&accumulator;
            C->data_ptr[row * n + col] = ptr[0] + ptr[1] + ptr[2] + ptr[3] + ptr[4] + ptr[5] + ptr[6] + ptr[7];
#endif
        }
    }

    return NULL;
}

namespace matmul {
void MatmulOperator::mat_mul_all_techniques(struct matmul_params *params) {
    int i, j, k;
    const struct matrix *A = &params->A, *B = &params->B, *C = &params->C;
    const int block_size = params->block_size;
    float *scale = params->scales, *offset = params->offset;

    assert(params->block_size % 32 == 0);  // support block size to be multiples of 32
    assert(A->row == C->row);              // support block size to be multiples of 32

    quantize_fp32_to_int8(A->data_ptr, A->int8_data_ptr, params->A_scales, A->row * A->column, block_size);

    const int num_thread = 8;
    pthread_t thread_pool[num_thread];
    struct w4a8_thread_args threads_args[num_thread];
    assert(params->block_size == 32);  // support block size 32 for now

    // TODO: Thread creation
    for (int j = 0; j < num_thread; j++) {
        threads_args[j].params = params;
        threads_args[j].start_j = j * (C->column / num_thread); 
        threads_args[j].end_j = (j + 1) * (C->column / num_thread);
        pthread_create(&thread_pool[j], NULL, all_techniques_worker_func, &threads_args[j]);
    }
    // TODO: Join threads
    for (int j = 0; j < num_thread; j++){
        pthread_join(thread_pool[j], NULL);
    }
};
}  // namespace matmul
