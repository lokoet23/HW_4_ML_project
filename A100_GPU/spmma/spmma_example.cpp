#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparseLt.h>       // cusparseLt header
#include <cublas_v2.h>        // cublas header
#include <cstdio>             // printf
#include <cstdlib>            // std::rand
#include <random>

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

constexpr int EXIT_UNSUPPORTED = 2;
// Host problem definition, row-major order
constexpr int m     = 4096; // bigger sizes may require dynamic allocations
constexpr int n     = 4096; // bigger sizes may require dynamic allocations
constexpr int k     = 4096; // bigger sizes may require dynamic allocations
auto          order = CUSPARSE_ORDER_ROW;
auto          opA   = CUSPARSE_OPERATION_NON_TRANSPOSE;
auto          opB   = CUSPARSE_OPERATION_NON_TRANSPOSE;
auto          type  = CUDA_R_16F;
auto          compute_type_cublas_tc = CUBLAS_COMPUTE_16F;  // for cublas with tensorcore
auto          compute_type_sparse = CUSPARSE_COMPUTE_16F;
// auto          type  = CUDA_R_32F;
// auto          compute_type_cublas_tc = CUBLAS_COMPUTE_32F_FAST_TF32;  // for cublas with tensorcore
// auto          compute_type_sparse = CUSPARSE_COMPUTE_TF32_FAST;

bool     is_rowmajor    = (order == CUSPARSE_ORDER_ROW);
bool     isA_transposed = (opA != CUSPARSE_OPERATION_NON_TRANSPOSE);
bool     isB_transposed = (opB != CUSPARSE_OPERATION_NON_TRANSPOSE);
auto     num_A_rows     = (isA_transposed) ? k : m;
auto     num_A_cols     = (isA_transposed) ? m : k;
auto     num_B_rows     = (isB_transposed) ? n : k;
auto     num_B_cols     = (isB_transposed) ? k : n;
auto     num_C_rows     = m;
auto     num_C_cols     = n;
unsigned alignment      = 16;
auto     lda            = (is_rowmajor) ? num_A_cols : num_A_rows;
auto     ldb            = (is_rowmajor) ? num_B_cols : num_B_rows;
auto     ldc            = (is_rowmajor) ? num_C_cols : num_C_rows;
auto     A_height       = (is_rowmajor) ? num_A_rows : num_A_cols;
auto     B_height       = (is_rowmajor) ? num_B_rows : num_B_cols;
auto     C_height       = (is_rowmajor) ? num_C_rows : num_C_cols;
auto     A_size         = A_height * lda * sizeof(__half);
auto     B_size         = B_height * ldb * sizeof(__half);
auto     C_size         = C_height * ldc * sizeof(__half);
// auto     A_size         = A_height * lda * sizeof(float);
// auto     B_size         = B_height * ldb * sizeof(float);
// auto     C_size         = C_height * ldc * sizeof(float);


int sparse_matmul(
    __half * hA, __half *hB, __half *hC, float alpha, float beta) {
    // float * hA, float *hB, float *hC, float alpha, float beta) {
    float elapse_time = 0;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC, *dD, *dA_compressed;
    // float *dA, *dB, *dC, *dD, *dA_compressed;
    int    *d_valid;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )
    CHECK_CUDA( cudaMalloc((void**) &d_valid, sizeof(int)) )
    dD = dC;

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------
    cusparseLtHandle_t             handle;
    cusparseLtMatDescriptor_t      matA, matB, matC;
    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;
    cusparseLtMatmulPlan_t         plan;
    cudaStream_t                   stream = nullptr;
    CHECK_CUSPARSE( cusparseLtInit(&handle) )
    // matrix descriptor initialization
    CHECK_CUSPARSE( cusparseLtStructuredDescriptorInit(
                                            &handle, &matA, num_A_rows,
                                            num_A_cols, lda, alignment,
                                            type, order,
                                            CUSPARSELT_SPARSITY_50_PERCENT) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matB, num_B_rows,
                                            num_B_cols, ldb, alignment,
                                            type, order) )
    CHECK_CUSPARSE( cusparseLtDenseDescriptorInit(
                                            &handle, &matC, num_C_rows,
                                            num_C_cols, ldc, alignment,
                                            type, order) )
    // matmul, algorithm selection, and plan initialization
    CHECK_CUSPARSE( cusparseLtMatmulDescriptorInit(
                                            &handle, &matmul, opA, opB,
                                            &matA, &matB, &matC, &matC,
                                            compute_type_sparse) )
    CHECK_CUSPARSE( cusparseLtMatmulAlgSelectionInit(
                                            &handle, &alg_sel, &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) )
    int alg = 0;
    CHECK_CUSPARSE( cusparseLtMatmulAlgSetAttribute(
                                            &handle, &alg_sel,
                                            CUSPARSELT_MATMUL_ALG_CONFIG_ID,
                                            &alg, sizeof(alg)))
    size_t workspace_size;
    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )
    //--------------------------------------------------------------------------
    // Prune the A matrix (in-place) and check the correctness
    CHECK_CUSPARSE( cusparseLtSpMMAPrune(&handle, &matmul, dA, dA,
                                         CUSPARSELT_PRUNE_SPMMA_TILE, stream) )
    CHECK_CUSPARSE( cusparseLtSpMMAPruneCheck(&handle, &matmul, dA,
                                              d_valid, stream) )
    int is_valid;
    CHECK_CUDA( cudaMemcpyAsync(&is_valid, d_valid, sizeof(int),
                                cudaMemcpyDeviceToHost, stream) )
    CHECK_CUDA( cudaStreamSynchronize(stream) )
    if (is_valid != 0) {
        std::printf("!!!! The matrix has been pruned in a wrong way. "
                    "cusparseLtMatmul will not provide correct results\n");
        return EXIT_FAILURE;
    }
    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size;

    CHECK_CUSPARSE( cusparseLtSpMMACompressedSize(&handle, &plan,
                                                  &compressed_size) )
    CHECK_CUDA( cudaMalloc((void**) &dA_compressed, compressed_size) )

    CHECK_CUSPARSE( cusparseLtSpMMACompress(&handle, &plan, dA,
                                            dA_compressed, stream) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Search the best kernel
    void*         d_workspace = nullptr;
    int           num_streams = 0;
    cudaStream_t* streams     = nullptr;

    CHECK_CUSPARSE( cusparseLtMatmulPlanInit(&handle, &plan, &matmul, &alg_sel,
                                             workspace_size) )

    CHECK_CUSPARSE( cusparseLtMatmulGetWorkspace(&handle, &plan,
                                                 &workspace_size))

    CHECK_CUDA( cudaMalloc((void**)&d_workspace, workspace_size) )
    // Perform the matrix multiplication
    CHECK_CUSPARSE( cusparseLtMatmul(&handle, &plan, &alpha, dA_compressed, dB,
                                    &beta, dC, dD, d_workspace, streams,
                                    num_streams) )
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // destroy plan and handle
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matA) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matB) )
    CHECK_CUSPARSE( cusparseLtMatDescriptorDestroy(&matC) )
    CHECK_CUSPARSE( cusparseLtMatmulPlanDestroy(&plan) )
    CHECK_CUSPARSE( cusparseLtDestroy(&handle) )
    //--------------------------------------------------------------------------
    // device result check
    // matrix A has been pruned
    CHECK_CUDA( cudaMemcpy(hA, dA, A_size, cudaMemcpyDeviceToHost) )
    CHECK_CUDA( cudaMemcpy(hC, dC, C_size, cudaMemcpyDeviceToHost) )

    //--------------------------------------------------------------------------
    // device memory deallocation
    CHECK_CUDA( cudaFree(dA_compressed) )
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    CHECK_CUDA( cudaFree(d_valid) )
    CHECK_CUDA( cudaFree(d_workspace) )
    return 0;
}


int dense_tensor_core_matmul(
    __half * hA, __half *hB, __half *hC, __half alpha, __half beta) {
    // float * hA, float *hB, float *hC, float alpha, float beta) {
    float elapse_time = 0;
    //--------------------------------------------------------------------------
    // Device memory management
    __half *dA, *dB, *dC;
    // float *dA, *dB, *dC;
    CHECK_CUDA( cudaMalloc((void**) &dA, A_size) )
    CHECK_CUDA( cudaMalloc((void**) &dB, B_size) )
    CHECK_CUDA( cudaMalloc((void**) &dC, C_size) )

    CHECK_CUDA( cudaMemcpy(dA, hA, A_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dB, hB, B_size, cudaMemcpyHostToDevice) )
    CHECK_CUDA( cudaMemcpy(dC, hC, C_size, cudaMemcpyHostToDevice) )
    //--------------------------------------------------------------------------

    cublasHandle_t cublasHandle;
    cublasErrCheck(cublasCreate(&cublasHandle));
    cublasErrCheck(
        cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH)
    ); // Use tensor cores
    // Now using cuBLAS
    // Perform the matrix multiplication
    cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    m, n, k, 
                    &alpha,
                    dA, type, m,
                    dB, type, k,
                    &beta, 
                    dC, type, m,
                    compute_type_cublas_tc, CUBLAS_GEMM_DFALT_TENSOR_OP));
    CHECK_CUDA( cudaFree(dA) )
    CHECK_CUDA( cudaFree(dB) )
    CHECK_CUDA( cudaFree(dC) )
    return 0;
}


int main(void) {
    int major_cc, minor_cc;
    CHECK_CUDA( cudaDeviceGetAttribute(&major_cc,
                                       cudaDevAttrComputeCapabilityMajor, 0) )
    CHECK_CUDA( cudaDeviceGetAttribute(&minor_cc,
                                       cudaDevAttrComputeCapabilityMinor, 0) )
    if (!(major_cc == 8 && minor_cc == 0) &&
        !(major_cc == 8 && minor_cc == 6)) {
        std::printf("\ncusparseLt is supported only on GPU devices with"
                    " compute capability == 8.0, 8.6 current: %d.%d\n\n",
                     major_cc, minor_cc);
        return EXIT_UNSUPPORTED;
    }

    printf("Matmul size: %dx%dx%d\n", m, k, n);

    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> unifrom(0, 1);  // uniform distribution in [0, 1]
    std::normal_distribution<> normal{0, 1};  // normal distribution in mean 0, std 1

    // // allocate memory in host DRAM
    __half *hA, *hB, *hC;
    hA = (__half*) malloc(sizeof(__half)*m*k);
    hB = (__half*) malloc(sizeof(__half)*k*n);
    hC = (__half*) malloc(sizeof(__half)*m*n);  // cpu result
    memset(hC, 0, sizeof(__half)*m*n);

    for (int i = 0; i < m * k; i++) 
        hA[i] = static_cast<__half>(normal(gen));
    for (int i = 0; i < k * n; i++)
        hB[i] = static_cast<__half>(normal(gen));
    __half alpha = 1.0f;
    __half beta  = 0.0f;

    // // allocate memory in host DRAM
    // float *hA, *hB, *hC;
    // hA = (float*) malloc(sizeof(float)*m*k);
    // hB = (float*) malloc(sizeof(float)*k*n);
    // hC = (float*) malloc(sizeof(float)*m*n);  // cpu result
    // memset(hC, 0, sizeof(float)*m*n);

    // for (int i = 0; i < m * k; i++) 
    //     hA[i] = static_cast<float>(normal(gen));
    // for (int i = 0; i < k * n; i++)
    //     hB[i] = static_cast<float>(normal(gen));
    // float alpha = 1.0f;
    // float beta  = 0.0f;

#ifdef SPARSE_MATMUL
    printf("\nRunning with cuSPARSELt...\n");
    sparse_matmul(hA, hB, hC, static_cast<float>(alpha), static_cast<float>(beta));
#else
    printf("\nRunning with cuBLAS...\n");
    dense_tensor_core_matmul(hA, hB, hC, alpha, beta);
#endif
    return EXIT_SUCCESS;
}
