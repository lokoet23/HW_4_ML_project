
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H


// define size of final arrays
#define final_rowA 128
#define final_colA 128
#define final_colB 128


// define the matrices dimensions of single kernel
#define rowA 64
#define colA 64	// colA=rowB
#define colB 64


// define the number of blocks (thus kernels) based on single kernel
constexpr int block_M = final_rowA/rowA;
constexpr int block_K = final_colA/colA;
constexpr int block_N = final_colB/colB;


// define the matrix multiplication shapes from the AI Engine API
#define M 4
#define K 8
#define N 4


// define the blocks
constexpr int num_rowA = rowA/M;
constexpr int num_colA = colA/K;
constexpr int num_colB = colB/N;


// define shift right for output values after matrix mult
#define SHIFT 0


#endif
