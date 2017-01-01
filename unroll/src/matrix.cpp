#include <iostream>
#include <chrono>
#include <omp.h>
#include <x86intrin.h>


// 行列の要素の型を決定する
using elem_t = double;

// N x Nの行列を扱う
constexpr std::size_t N = 1024;

// ループを4回展開する
constexpr std::size_t UNROLL = 4;

// ホスト(CPU)側の行列を定義
static elem_t h_A[N * N];
static elem_t h_B[N * N];
static elem_t h_C[N * N];

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


// 行列の積を計算する
void matrix_multiply(elem_t* A, elem_t* B, elem_t* C)
{
    // ここでは1回の繰り返しで4 * UNROLLつの要素を処理しているので、iを4 * UNROLLずつ繰り上げる
    for (std::size_t i = 0; i < N; i += 4 * UNROLL) {

        for (std::size_t j = 0; j < N; j++) {

            // 4つの倍精度浮動小数点を並行して(_pd)、ループ展開する分だけ、行列Cからc[x]にロードする
            __m256d c[UNROLL];
            for (std::size_t x = 0; x < UNROLL; x++) {
                c[x] = _mm256_load_pd(C + i + 4 * x + N * j);
            }

            for (std::size_t k = 0; k < N; k++) {
                
                // ループを通じてB要素の4つが使えるため、コピーは1つあればいい
                __m256d b = _mm256_broadcast_sd(B + k + N * j);
                for (std::size_t x = 0; x < UNROLL; x++) {
                    c[x] = _mm256_add_pd(
                        c[x],
                        _mm256_mul_pd(_mm256_load_pd(A + i + 4 * x + N * k ), b));
                }
            }

            // 4つの倍精度浮動小数点を並行して、ループ展開する分だけ、c[x]から行列Cにストアする
            for (std::size_t x = 0; x < UNROLL; x++) {
                _mm256_store_pd(C + i + 4 * x + N * j, c[x]);
            }
        }
    }

}