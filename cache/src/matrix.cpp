#include <iostream>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>


// 行列の要素の型を決定する
using elem_t = double;

// N x Nの行列を扱う
constexpr std::size_t N = 1024;

// ホスト(CPU)側の行列を定義
static elem_t h_A[N * N];
static elem_t h_B[N * N];
static elem_t h_C[N * N];

// BLOCK_SIZE x BLOCK_SIZEの部分行列を考えることになる
constexpr std::size_t BLOCK_SIZE = 32;

// 行列の積を計算する関数
static void matrix_multiply(elem_t*, elem_t*, elem_t*);

// 高精度タイマー
struct timer {
public:
    inline void start() { start_ = std::chrono::high_resolution_clock::now(); }
    inline void end()   { end_   = std::chrono::high_resolution_clock::now(); }
    inline std::chrono::duration<double> duration() const { return end_ - start_; }
    inline double elapsed() const { return duration().count(); }

private:
    std::chrono::high_resolution_clock::time_point start_;
    std::chrono::high_resolution_clock::time_point end_;
};



// メイン関数
int main(void)
{
    for (std::size_t i = 0; i < N * N; i++) {
        h_A[i] = static_cast<elem_t>(i);
        h_B[i] = static_cast<elem_t>(i);
        h_C[i] = static_cast<elem_t>(0);
    }

    timer t;
    t.start();
    matrix_multiply(h_A, h_B, h_C);
    t.end();

    std::cout << "host calculator time is " << t.elapsed() << " seconds." << std::endl;
    std::cout << "host calculator result is " << h_C[N * N - 1] << "." << std::endl;

    return 0;
}


// BLOCK_SIZE単位で処理を行う
// NOTE: BLOCK_SIZE = 64の場合、64 * 64 * 8 * 3 > 8MiBとなりキャッシュに収まりきらない
static inline void do_block(std::size_t si, std::size_t sj, std::size_t sk, elem_t* A, elem_t* B, elem_t* C)
{
    for (std::size_t i = si; i < si + BLOCK_SIZE; i++) {

        for (std::size_t j = sj; j < sj + BLOCK_SIZE; j++) {

            elem_t cij = C[i + N * j];
            for (std::size_t k = sk; k < sk + BLOCK_SIZE; k++) {

                cij += A[i + N * k] * B[k + N * j];
            }
            C[i + N * j] = cij;
        }
    }
}


// 行列の積を計算する
void matrix_multiply(elem_t* A, elem_t* B, elem_t* C)
{
    for (std::size_t sj = 0; sj < N; sj += BLOCK_SIZE) {

        for (std::size_t si = 0; si < N; si += BLOCK_SIZE) {

            for (std::size_t sk = 0; sk < N; sk += BLOCK_SIZE) {

                // BLOCK_SIZE単位で処理を行う
                do_block(si, sj, sk, A, B, C);
            }
        }
    }

}