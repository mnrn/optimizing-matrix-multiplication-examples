#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>
#include <omp.h>


// N x Nの行列を扱う
static constexpr int N = 1024;

// BLOCK_SIZE単位で処理を行う
static constexpr int BLOCK_SIZE = 32;
inline void do_block(int si, int sj, int sk, double* A, double* B, double* C)
{
    for (int i = si; i < si + BLOCK_SIZE; i++) {
        for (int j = sj; j < sj + BLOCK_SIZE; j++) {
            double cij = C[i + N * j];
            for (int k = sk; k < sk + BLOCK_SIZE; k++) {
                cij += A[i + N * k] * B[k + N * j];
            }
            C[i + N * j] = cij;
        }
    }
}

// 行列の積を計算する
void matrix_multiply(double* A, double* B, double* C)
{
#pragma omp parallel for
    for (int sj = 0; sj < N; sj += BLOCK_SIZE) {
        for (int si = 0; si < N; si += BLOCK_SIZE) {
            for (int sk = 0; sk < N; sk += BLOCK_SIZE) {
                // BLOCK_SIZE単位で処理を行う
                do_block(si, sj, sk, A, B, C);
            }
        }
    }
}

// メイン関数
int main(void)
{
    omp_set_num_threads(omp_get_num_procs());
    std::cout << "Use num threads: " << omp_get_num_procs() << std::endl;

    // 行列を定義
    static double matrixA[N * N];
    static double matrixB[N * N];
    static double matrixC[N * N];

    for (int i = 0; i < N * N; i++) {
        matrixA[i] = static_cast<double>(i + 1);
        matrixB[i] = static_cast<double>(-i - 1);
        matrixC[i] = static_cast<double>(0);
    }

    auto start = std::chrono::high_resolution_clock::now();
    matrix_multiply(matrixA, matrixB, matrixC);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Calculator time is " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;
    std::cout << "Calculator result is " << std::setprecision(std::numeric_limits<double>::max_digits10) << matrixC[N * N - 1] << "." << std::endl;
    
    return 0;
}
