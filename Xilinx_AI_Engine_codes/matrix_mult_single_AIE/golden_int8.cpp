/*
* g++ -o golden_int8_cpp golden_int8.cpp
*/

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>

// define the matrices dimensions
#define rowA (128*10)
#define colA 128	// colA=rowB
#define colB (128*10)

// define the matrix multiplication shapes from the AI Engine API
#define M 4
#define K 8
#define N 4


// define the blocks
const int num_rowA = rowA/M;
const int num_colA = colA/K;
const int num_colB = colB/N;


// define shift right for output values after matrix mult
#define SHIFT 0



int main(){


	int8_t matA[rowA*colA], matB[colA*colB], matC[rowA*colB];
	
	std::fstream a_file;
	std::fstream b_file;
	std::fstream c_file;

	// generate input and write to files
	a_file.open("./matA.txt", std::ios::out);
	b_file.open("./matB.txt", std::ios::out);
	c_file.open("./matC.txt", std::ios::out);
	
	// seed
	srand(time(NULL));

	// generate matA in blocked format
	for (int i = 0; i < num_rowA; i++){
		for (int j = 0; j < num_colA; j++){
		
			for (int x = 0; x < M; x++){
				for (int y = 0; y < K; y++){
				
					matA[i*num_colA*M*K + j*M*K + x*K + y] = rand();
//					printf("%d\n", i*num_colA*M*K + j*M*K + x*K + y);
				}
			}
		}
	}
	
	// generate matB in blocked format
	for (int i = 0; i < num_colA; i++){
		for (int j = 0; j < num_colB; j++){
		
			for (int x = 0; x < K; x++){
				for (int y = 0; y < N; y++){
				
					matB[i*num_colB*K*N + j*K*N + x*N + y] = rand();
//					printf("%d\n", i*num_colB*K*N + j*K*N + x*N + y);
				}
			}
		}
	}

	// write matA to matA.txt
	for (int i = 0; i < rowA*colA; i++){
		
		a_file << int(matA[i]);
		if (i % 4 == 3){
			a_file << "\n";
		}
		else{
			a_file << " ";
		}
	}
	
	// write matB to matB.txt
	for (int i = 0; i < colA*colB; i++){
		
		b_file << int(matB[i]);
		if (i % 4 == 3){
			b_file << "\n";
		}
		else{
			b_file << " ";
		}
	}

   	
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
							C_block[x][y] += matA[i*num_colA*M*K + k*M*K + x*K + z] * matB[j*K*N + k*num_colB*K*N + N*z + y];
						}
   					}
   				}
   				
   			}
   			
   			// save output to matC
   			for (int x = 0; x < M; x++){
 				for (int y = 0; y < N; y++){
 				
 					matC[i*num_colB*M*N + j*M*N + x*N + y] = C_block[x][y];
   				}
   			}
   		}
   	}



	// write matC to matC.txt
	for (int i = 0; i < rowA*colB; i++){
		
		c_file << int(matC[i]);
		if (i % 4 == 3){
			c_file << "\n";
		}
		else{
			c_file << " ";
		}
	}
	
	a_file.close();
	b_file.close();
	c_file.close();
	
	return 0;

}
