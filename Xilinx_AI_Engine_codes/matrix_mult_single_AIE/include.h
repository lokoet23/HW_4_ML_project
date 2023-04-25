
#ifndef FUNCTION_INCLUDES_H
#define FUNCTION_INCLUDES_H

// define the matrices dimensions
#define rowA 128
#define colA 128	// colA=rowB
#define colB 128

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
