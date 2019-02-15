#include <iostream>
#include <chrono>
#include <x86intrin.h>


// N x Nの行列を扱う
static constexpr size_t N = 1024;

// 行列の積を計算する関数
static void matrix_multiply(const double*, const double*, double*);

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


// 行列の積を計算する
void matrix_multiply(const double* A, const double* B, double* C)
{
    // ここでは1回の繰り返しで4つの要素を処理しているので、iを4ずつ繰り上げる
    for (size_t i = 0; i < N; i += 4) {

        for (size_t j = 0; j < N; j++) {

            // 4つの倍精度浮動小数点を並行して(_pd)、行列Cからcにロードする
            __m256d c0 = _mm256_load_pd(C + i + N * j);  // c0 = C[i][j]
            for (size_t k = 0; k < N; k++) {
                
                // c0 += A[i][k] * B[k][j] 
                c0 = _mm256_add_pd(
                    c0,
                    _mm256_mul_pd(_mm256_load_pd(A + i + N * k),  // 最初に4つの要素をロードし、それらの要素にBの1つの要素をかけるため
                        _mm256_broadcast_sd(B + k + N * j)        // Bの要素の同一のコピーを4つ生成し、_mm256_mul_pd()を使用することで、
                    )                                             // 4つの倍精度の結果を並行してかける. 最後に_mm256_add_pd()を使用し、
                );                                                // 4つの積をc0の4つの和に加算する. 
            }

            // 4つの倍精度浮動小数点をc0から行列Cにストアする
            _mm256_store_pd(C + i + N * j, c0);  // C[i][j] = c0;
        }
    }
}