
#include <adf.h>
#include "kernels.h"
#include "include.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:


  // total number of kernels mapped to AI Engine Array
  // is the addition of the sizes of the following 2 matrices
  kernel mat_mul_k[block_K * block_M * block_N];
  kernel add_k[(block_K - 1) * block_M * block_N];

public:

  input_plio A[block_M * block_K];
  input_plio B[block_K * block_N];

  output_plio C[block_M * block_N];

  simpleGraph(){

	  // input and output plio creation below
	  // A: 0   1    2  ...
	  //    x  x+1  x+2 ...
	  for (int i = 0; i < block_M * block_K; i++){
		  A[i] = input_plio::create(plio_32_bits, "data/matA" + std::to_string(i) + ".txt");
	  }

	  // B: 0   y
	  //	1  y+1
	  //	2  y+2
	  //   ... ...
	  for (int i = 0; i < block_K * block_N; i++){
		  B[i] = input_plio::create(plio_32_bits, "data/matB" + std::to_string(i) + ".txt");
	  }

	  // C: 0   1    2  ...
	  //    x  x+1  x+2 ...
	  for (int i = 0; i < block_M * block_N; i++){
		  C[i] = output_plio::create(plio_32_bits, "data/matC" + std::to_string(i) + ".txt");
	  }



	  // kernels creation
	  for (int i = 0; i < block_K * block_M * block_N; i++){
		  mat_mul_k[i] = kernel::create(blocked_matrix_mult);
	  }

	  for (int i = 0; i < (block_K - 1) * block_M * block_N; i++){
		  add_k[i] = kernel::create(vectorized_addition);
	  }


	  // graph automated generation
	  for (int i = 0; i < block_M; i++){
		  for (int j = 0; j < block_N; j++){

			  for (int k = 0; k < block_K; k++){

				  // matrix multiplication kernels connection
				  connect< window<rowA*colA*1> >  (A[i*block_K + k].out[0], mat_mul_k[i*block_N*block_K + j*block_K + k].in[0]);
				  connect< window<colA*colB*1> >  (B[j*block_K + k].out[0], mat_mul_k[i*block_N*block_K + j*block_K + k].in[1]);

				  // first time, addition kernel connections
				  if (k == 1){
					  connect< window<rowA*colB*1> >  (mat_mul_k[i*block_N*block_K + j*block_K + k -1].out[0], add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-1].in[0]);
					  connect< window<rowA*colB*1> >  (mat_mul_k[i*block_N*block_K + j*block_K + k].out[0], add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-1].in[1]);
				  }
				  // remaining times
				  else if (k > 1){
					  connect< window<rowA*colB*1> >  (add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-2].out[0], add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-1].in[0]);
					  connect< window<rowA*colB*1> >  (mat_mul_k[i*block_N*block_K + j*block_K + k].out[0], add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-1].in[1]);
				  }

				  // reached the end, save to output
				  if (k == (block_K - 1)){
					  connect< window<rowA*colB*1> >  (add_k[i*block_N*(block_K-1) + j*(block_K-1) + k-1].out[0], C[i*block_N + j].in[0]);
				  }

			  }

		  }
	  }

    

	  // direct the source file of kernels
	  for (int i = 0; i < block_K * block_M * block_N; i++){
		  source(mat_mul_k[i]) = "kernels/kernels.cc";
	  }

	  for (int i = 0; i < (block_K - 1) * block_M * block_N; i++){
		  source(add_k[i]) = "kernels/kernels.cc";
	  }


	  // estimate runtime ratio
	  for (int i = 0; i < block_K * block_M * block_N; i++){
		  runtime<ratio>(mat_mul_k[i]) = 0.6;
	  }

	  for (int i = 0; i < (block_K - 1) * block_M * block_N; i++){
		  runtime<ratio>(add_k[i]) = 0.6;
	  }

  }
};
