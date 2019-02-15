#include <iostream>
#include <chrono>
#include <cuda_runtime.h>



// N x Nの行列を扱う
constexpr std::size_t N = 1024;

// 1つのブロックでBLOCK_SIZE x BLOCK_SIZEのスレッドを管理する
constexpr std::size_t BLOCK_SIZE = 16;

// ホスト(CPU)側の行列を定義
static double hostMatrixA[N * N];
static double hostMatrixB[N * N];
static double hostMatrixC[N * N];

// デバイス(GPU)側の行列へのポインタ
static double* deviceMatrixA;
static double* deviceMatrixB;
static double* deviceMatrixC;

// 行列の積を計算する関数
__global__ void matrix_multiply(double*, double*, double*);
__global__ void matrix_multiply_shared(double*, double*, double*);


int main(void)
{
    // デバイス側に行列用の記憶領域を確保する
    cudaMalloc((void**)&deviceMatrixA, sizeof(hostMatrixA));
    cudaMalloc((void**)&deviceMatrixB, sizeof(hostMatrixB));
    cudaMalloc((void**)&deviceMatrixC, sizeof(hostMatrixC));

    // ホスト側の行列に値を設定する
    for (std::size_t i = 0; i < N * N; i++) {
        hostMatrixA[i] = static_cast<double>(i + 1);
        hostMatrixB[i] = static_cast<double>(-i - 1);
        hostMatrixC[i] = static_cast<double>(0);
    }

    // タイマー開始
    auto start = std::chrono::high_resolution_clock::now();

    // ホスト側の行列のデータをデバイス側の行列へ転送する
    cudaMemcpy(deviceMatrixA, hostMatrixA, sizeof(hostMatrixA), cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, hostMatrixB, sizeof(hostMatrixB), cudaMemcpyHostToDevice);

    // グリッドおよびブロックの定義
    dim3 grid(N / BLOCK_SIZE, N / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);

    // GPU側の処理を起動させる
    matrix_multiply <<< grid, block >>> (deviceMatrixA, deviceMatrixB, deviceMatrixC);

    // deviceMatrixCに格納されている計算結果をhostMatrixCへ転送する
    cudaMemcpy(hostMatrixC, deviceMatrixC, sizeof(hostMatrixC), cudaMemcpyDeviceTohost);

    // タイマー終了
    auto end = std::chrono::high_resolution_clock::now();

    // 計算結果発表
    std::cout << "Calculator time is " << std::chrono::duration<double>(end - start).count() << " seconds." << std::endl;
    std::cout << "Calculator result is " << std::setprecision(std::numeric_limits<double>::max_digits10) << hostMatrixC[N * N - 1] << "." << std::endl;

    // デバイス側の記憶領域を解放する
    cudaFree(deviceMatrixC);
    cudaFree(deviceMatrixB);
    cudaFree(deviceMatrixA);

    return 0;
}



// 行列の積を計算する関数
__global__ void matrix_multiply(double* A, double* B, double* C)
{
    // 各スレッドが担当する行列の行yと列xを取得
    std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    double c = 0;
    for (std::size_t i = 0; i < N; i++) {
        c += A[N * y + i] * B[N * i + x];
    }
    C[x + N * y] = c;
}


// シェアードメモリを利用した行列積計算関数
__global__ void matrix_multiply_shared(double* A, double* B, double* C)
{
    // 各スレッドが担当する行列の行yと列xを取得
    std::size_t y = blockDim.y * blockIdx.y + threadIdx.y;
    std::size_t x = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ double s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double s_B[BLOCK_SIZE][BLOCK_SIZE];

    double c = 0;
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
