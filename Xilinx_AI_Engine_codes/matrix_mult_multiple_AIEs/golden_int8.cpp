/*
* g++ -o golden_int8_cpp golden_int8.cpp
*/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// define size of final arrays
#define final_rowA (128*10)
#define final_colA 128
#define final_colB (128*10)

// define the matrices dimensions of single kernel
#define rowA (64*10)
#define colA 64	// colA=rowB
#define colB (64*10)

// define the matrix multiplication shapes from the AI Engine API
#define M 4
#define K 8
#define N 4


// define the number of blocks (thus kernels) based on single kernel
constexpr int block_M = final_rowA/rowA;
constexpr int block_K = final_colA/colA;
constexpr int block_N = final_colB/colB;


// define the blocks
const int num_rowA = rowA/M;
const int num_colA = colA/K;
const int num_colB = colB/N;


// define shift right for output values after matrix mult
#define SHIFT 0



int main(){



	int8_t matA[rowA*colA][block_M * block_K], matB[colA*colB][block_K * block_N], matC[rowA*colB][block_K], matC_addition[rowA*colB][block_M * block_N];
	
	std::fstream a_file_array[block_M * block_K];
	std::fstream b_file_array[block_K * block_N];
	std::fstream c_file_array[block_M * block_N];


	for (int i = 0; i < block_M * block_K; i++){
		a_file_array[i].open("./matA" + std::to_string(i) + ".txt", std::ios::out);
	}

	for (int i = 0; i < block_K * block_N; i++){
		b_file_array[i].open("./matB" + std::to_string(i) + ".txt", std::ios::out);
	}

	for (int i = 0; i < block_M * block_N; i++){
		c_file_array[i].open("./matC" + std::to_string(i) + ".txt", std::ios::out);
	}
	

	// seed
	srand(time(NULL));


	// A matrix
	for (int kk = 0; kk < block_M * block_K; kk++){
		// generate matA in blocked format
		for (int i = 0; i < num_rowA; i++){
			for (int j = 0; j < num_colA; j++){
		
				for (int x = 0; x < M; x++){
					for (int y = 0; y < K; y++){
				
						matA[i*num_colA*M*K + j*M*K + x*K + y][kk] = rand();
//						printf("%d\n", i*num_colA*M*K + j*M*K + x*K + y);
					}
				}
			}
		}

		// write matA to matA.txt
		for (int i = 0; i < rowA*colA; i++){

			a_file_array[kk] << int(matA[i][kk]);
			if (i % 4 == 3){
				a_file_array[kk] << "\n";
			}
			else{
				a_file_array[kk] << " ";
			}
		}
	}
	

	// B matrix
	for (int kk = 0; kk < block_K * block_N; kk++){
		// generate matB in blocked format
		for (int i = 0; i < num_colA; i++){
			for (int j = 0; j < num_colB; j++){
		
				for (int x = 0; x < K; x++){
					for (int y = 0; y < N; y++){
				
						matB[i*num_colB*K*N + j*K*N + x*N + y][kk] = rand();
//						printf("%d\n", i*num_colB*K*N + j*K*N + x*N + y);
					}
				}
			}
		}


		// write matB to matB.txt
		for (int i = 0; i < colA*colB; i++){
		
			b_file_array[kk] << int(matB[i][kk]);
			if (i % 4 == 3){
				b_file_array[kk] << "\n";
			}
			else{
				b_file_array[kk] << " ";
			}
		}
	}




	for (int ii = 0; ii < block_M; ii++){
		for (int jj = 0; jj < block_N; jj++){
			for (int kk = 0; kk < block_K; kk++){


   				for (int i = 0; i < num_rowA; i++){
   						for (int j = 0; j < num_colB; j++){
   		
   								int8_t C_block[M][N];
   								// initialize block to 0
   								for (int x = 0; x < M; x++){
   									for (int y = 0; y < N; y++){
   										C_block[x][y] = 0;
   									}
   								}
   		
   								for (int k = 0; k < num_colA; k++){
   				
   									for (int x = 0; x < M; x++){
   										for (int y = 0; y < N; y++){
   					
   											for (int z = 0; z < K; z++){
   												C_block[x][y] += matA[i*num_colA*M*K + k*M*K + x*K + z][ii*block_K+ kk] * matB[j*K*N + k*num_colB*K*N + N*z + y][jj*block_K + kk];
   											}
   										}
   									}

   								}
   			
   								// save output to matC
   								for (int x = 0; x < M; x++){
   									for (int y = 0; y < N; y++){
 				
   										matC[i*num_colB*M*N + j*M*N + x*N + y][kk] = C_block[x][y];
   									}
   								}

   						}
   				}

   				// add the partial sums
   				if (kk == 0){
   					for (int i = 0; i < rowA*colB; i++){
   						matC_addition[i][ii*block_N + jj] = matC[i][0];
   					}
   				}
   				else {
   					for (int i = 0; i < rowA*colB; i++){
   						matC_addition[i][ii*block_N + jj] += matC[i][kk];

   					}
   				}



			}
		}
	}




	for (int kk = 0; kk < block_M * block_N; kk++){
   		// write to output after elementwise addition
		for (int i = 0; i < rowA*colB; i++){
		
			c_file_array[kk] << int(matC_addition[i][kk]);
			if (i % 4 == 3){
				c_file_array[kk] << "\n";
			}
			else{
				c_file_array[kk] << " ";
			}
		}
	}


   	// close files
   	for (int i = 0; i < block_M * block_K; i++){
   		a_file_array[i].close();
   	}

   	for (int i = 0; i < block_K * block_N; i++){
   		b_file_array[i].close();
   	}

   	for (int i = 0; i < block_M * block_N; i++){
   		c_file_array[i].close();
   	}


	
	return 0;

}
