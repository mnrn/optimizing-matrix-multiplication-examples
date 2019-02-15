#include <iostream>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>



// N x Nの行列を扱う
static constexpr size_t N = 1024;

// BLOCK_SIZE x BLOCK_SIZEの部分行列を考えることになる
static constexpr size_t BLOCK_SIZE = 32;

// ループをUNROLL回展開する
static constexpr size_t UNROLL = 4;

// 行列の積を計算する関数
static void matrix_multiply(double*, double*, double*);

// メイン関数
int main(void)
{
    double* matrixA = static_cast<double*>(_mm_malloc(sizeof(double) * N * N, 32));
    double* matrixB = static_cast<double*>(_mm_malloc(sizeof(double) * N * N, 32));
    double* matrixC = static_cast<double*>(_mm_malloc(sizeof(double) * N * N, 32));

    for (int i = 0; i < N * N; i++) {
        matrixA[i] = static_cast<double>(i + 1);
        matrixB[i] = static_cast<double>(-i - 1);
        matrixC[i] = static_cast<double>(0);
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply(matrixA, matrixB, matrixC);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Calculator time is " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;
    std::cout << "Calculator result is " << matrixC[N * N - 1] << "." << std::endl;

    _mm_free(matrixC);
    _mm_free(matrixB);
    _mm_free(matrixA);
    
    return 0;
}


// BLOCK_SIZE単位で処理を行う
// NOTE: BLOCK_SIZE = 64の場合、64 * 64 * 8 * 3 > 8MiBとなりキャッシュに収まりきらない
static inline void do_block(size_t si, size_t sj, size_t sk, double* A, double* B, double* C)
{
    // ここでは1回の繰り返しで4 * UNROLLつの要素を処理しているので、iを4 * UNROLLずつ繰り上げる
    for (size_t i = si; i < si + BLOCK_SIZE; i += 4 * UNROLL) {

        for (size_t j = sj; j < sj + BLOCK_SIZE; j++) {

            // 4つの倍精度浮動小数点を並行して(_pd)、ループ展開する分だけ、行列Cからc[x]にロードする
            __m256d c[UNROLL];
            for (size_t x = 0; x < UNROLL; x++) {
                c[x] = _mm256_load_pd(C + i + 4 * x + N * j);
            }

            for (size_t k = sk; k < sk + BLOCK_SIZE; k++) {

                // ループを通じてB要素の4つが使えるため、コピーは1つあればいい
                __m256d b = _mm256_broadcast_sd(B + k + N * j);
                for (size_t x = 0; x < UNROLL; x++) {
                    c[x] = _mm256_add_pd(
                        c[x],
                        _mm256_mul_pd(_mm256_load_pd(A + i + 4 * x + N * k ), b));
                }

            }

            // 4つの倍精度浮動小数点を並行して、ループ展開する分だけ、c[x]から行列Cにストアする
            for (size_t x = 0; x < UNROLL; x++) {
                _mm256_store_pd(C + i + 4 * x + N * j, c[x]);
            }

        }
    }
}


// 行列の積を計算する
void matrix_multiply(double* A, double* B, double* C)
{
#pragma omp parallel for
    for (size_t sj = 0; sj < N; sj += BLOCK_SIZE) {

        for (size_t si = 0; si < N; si += BLOCK_SIZE) {

            for (size_t sk = 0; sk < N; sk += BLOCK_SIZE) {

                // BLOCK_SIZE単位で処理を行う
                do_block(si, sj, sk, A, B, C);
            }
        }
    }
}