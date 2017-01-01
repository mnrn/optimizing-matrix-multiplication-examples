#include <iostream>
#include <chrono>
#include <cuda_runtime.h>



// N x Nの行列を扱う
constexpr std::size_t N = 1024;

// 1つのブロックでBLOCK_SIZE x BLOCK_SIZEのスレッドを管理する
constexpr std::size_t BLOCK_SIZE = 16;

// 行列の要素の型を決定する
using elem_t = double;

// ホスト(CPU)側の行列を定義
static elem_t h_A[N * N];
static elem_t h_B[N * N];
static elem_t h_C[N * N];

// デバイス(GPU)側の行列へのポインタ
static elem_t* d_A;
static elem_t* d_B;
static elem_t* d_C;

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

// 行列の積を計算する関数
__global__ void matrix_multiply(elem_t*, elem_t*, elem_t*);
__global__ void matrix_multiply_shared(elem_t*, elem_t*, elem_t*);



int main(void)
{
    // デバイス側に行列用の記憶領域を確保する
    cudaMalloc((void**)&d_A, sizeof(h_A));
    cudaMalloc((void**)&d_B, sizeof(h_B));
    cudaMalloc((void**)&d_C, sizeof(h_C));

    // ホスト側の行列に値を設定する
    for (std::size_t i = 0; i < N * N; i++) {
        h_A[i] = static_cast<elem_t>(i);
        h_B[i] = static_cast<elem_t>(i);
        h_C[i] = static_cast<elem_t>(0);
    }

    // タイマー開始
    timer t;
    t.start();

    // ホスト側の行列のデータをデバイス側の行列へ転送する
    cudaMemcpy(d_A, h_A, sizeof(h_A), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(h_B), cudaMemcpyHostToDevice);

    // グリッドおよびブロックの定義
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // GPU側の処理を起動させる
    matrix_multiply_shared <<< grid, block >>> (d_A, d_B, d_C);

    // d_Cに格納されている計算結果をh_Cへ転送する
    cudaMemcpy(h_C, d_C, sizeof(h_C), cudaMemcpyDeviceToHost);

    // タイマー終了
    t.end();

    // 計算結果発表
    std::cout << "device calculator time is " << t.elapsed() << " seconds." << std::endl;
    std::cout << "device calculator result is " << h_C[N * N - 1] << "." << std::endl;

    // デバイス側の記憶領域を解放する
    cudaFree(d_C);
    cudaFree(d_B);
    cudaFree(d_A);

    return 0;
}



// 行列の積を計算する関数
__global__ void matrix_multiply(elem_t* A, elem_t* B, elem_t* C)
{
    // 各スレッドが担当する行列の行yと列xを取得
    std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    elem_t c = 0;
    for (std::size_t i = 0; i < N; i++) {
        c += A[N * y + i] * B[N * i + x];
    }
    C[x + N * y] = c;
}


// シェアードメモリを利用した行列積計算関数
__global__ void matrix_multiply_shared(elem_t* A, elem_t* B, elem_t* C)
{
    // 各スレッドが担当する行列の行yと列xを取得
    std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ elem_t s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ elem_t s_B[BLOCK_SIZE][BLOCK_SIZE];

    elem_t c = 0;
    for (std::size_t i = 0; i < N; i += BLOCK_SIZE) {

        // シェアードメモリに行列の一部をコピー
        s_A[threadIdx.y][threadIdx.x] = A[N * y + i + threadIdx.x];
        s_B[threadIdx.y][threadIdx.x] = B[N * (i + threadIdx.y) + x];
        __syncthreads();

        // シェアードメモリで積を計算する
        for (std::size_t j = 0; j < BLOCK_SIZE; j++) {
            c += s_A[threadIdx.y][j] * s_B[j][threadIdx.x];
        }
        __syncthreads();
    }
    C[N * y + x] = c;
}
