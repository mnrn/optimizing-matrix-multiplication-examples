#include <iostream>
#include <chrono>
#include <iomanip>
#include <limits>


// N x Nの行列を扱う
static constexpr int N = 1024;

// 行列の積を計算する
void matrix_multiply(double* A, double* B, double* C)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double c = C[i + N * j];  
            for (int k = 0; k < N; k++) {
                c += A[i + N * k] * B[k + N * j];
            }
            C[i + N * j] = c;
        }
    }
}

// メイン関数
int main(void)
{
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
