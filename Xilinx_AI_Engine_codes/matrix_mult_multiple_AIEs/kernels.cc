#include <adf.h>
#include "aie_api/aie.hpp"
#include "aie_api/aie_adf.hpp"
#include "include.h"


/*
 *  All matrices should be in blocked format in memory, in the following order:
 * 	 ____________________________
 * 	|  1 |  2 |  3 | ...
 * 	|____|____|____|
 * 	|  x | x+1| x+2| ...
 * 	|____|____|____|
 * 	|.
 * 	|.
 * 	|.
 *
 */


void blocked_matrix_mult(input_window_int8 * __restrict matA, input_window_int8 * __restrict matB,
						output_window_int8 * __restrict matC) {

	// change M, K, N at include.h, based on AI Engine API
	using MMUL = aie::mmul<M, K, N, int8, int8>;

	// pointers of matrices
	const int8* __restrict pA = (int8*) matA->ptr;
	const int8* __restrict pB = (int8*) matB->ptr;
	int8* __restrict pC = (int8*) matC->ptr;

	// copy of pC for addition purposes
	int8 * __restrict pC1 = pC;


	for (unsigned i = 0; i < num_rowA; i++) {

		for (unsigned j = 0; j < num_colB; j++) {

			// MMUL::size_A = M*K
			const int8 * __restrict pA1 = pA + ( i * num_colA + 0) * MMUL::size_A;

			// MMUL::size_B = K*N
			const int8 * __restrict pB1 = pB + ( 0 * num_colB + j) * MMUL::size_B;


			aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;

			aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;

			MMUL C00;

			// matrix multiply by initializing to 0
			C00.mul(A0, B0);

			for (unsigned k = 0; k < num_colA-1; k++)
//			chess_prepare_for_pipelining
			{
				A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;

				B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;

				// matrix multiply and adding partial blocks
				C00.mac(A0, B0);
			}

			aie::store_v(pC1, C00.template to_vector<int8>(SHIFT)); pC1 +=MMUL::size_C;


		}

	}

}


void vectorized_addition(input_window_int8 * __restrict in_a, input_window_int8 * __restrict in_b,
						output_window_int8 * __restrict out_c) {


	for (unsigned i = 0; i < (rowA*colB/128); i++)
//	  chess_prepare_for_pipelining
	{

		aie::vector<int8, 128> v_a = window_readincr_v<128>(in_a); // read data from input
		aie::vector<int8, 128> v_b = window_readincr_v<128>(in_b);

		aie::vector<int8, 128> v_c = aie::add(v_a, v_b);

		window_writeincr(out_c, v_c);	// write data to output without right shift
	}
}



// optimized as example in AIE APIs
void opt_blocked_matrix_mult(input_window_int8 * __restrict matA, input_window_int8 * __restrict matB,
						output_window_int8 * __restrict matC) {

	// change M, K, N at include.h, based on AI Engine API
	using MMUL = aie::mmul<M, K, N, int8, int8>;

	// pointers of matrices
	const int8* __restrict pA = (int8*) matA->ptr;
	const int8* __restrict pB = (int8*) matB->ptr;
	int8* __restrict pC = (int8*) matC->ptr;


	// unroll the loops for more optimization

	for (unsigned i = 0; i < num_rowA; i+=2)
//	chess_loop_range(2,)
	{

		int8 * __restrict pC1 = pC + (i * num_colB) * MMUL::size_C;
		int8 * __restrict pC2 = pC + ((i+1) * num_colB) * MMUL::size_C;;

		for (unsigned j = 0; j < num_colB; j+=2)
//		chess_loop_range(2,)
		{

			const int8 * __restrict pA1 = pA + ( i * num_colA + 0) * MMUL::size_A;
			const int8 * __restrict pA2 = pA + ( (i+1) * num_colA + 0) * MMUL::size_A;

			const int8 * __restrict pB1 = pB + ( 0 * num_colB + j) * MMUL::size_B;
			const int8 * __restrict pB2 = pB + ( 0 * num_colB + (j+1)) * MMUL::size_B;


			aie::vector<int8, MMUL::size_A> A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
			aie::vector<int8, MMUL::size_A> A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;

			aie::vector<int8, MMUL::size_B> B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;
			aie::vector<int8, MMUL::size_B> B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * num_colB;

			MMUL C00;
			MMUL C01;
			MMUL C10;
			MMUL C11;

			// matrix multiply by initializing to 0
			C00.mul(A0, B0);
			C01.mul(A0, B1);
			C10.mul(A1, B0);
			C11.mul(A1, B1);

			for (unsigned k = 0; k < num_colA-1; k++)
//			chess_prepare_for_pipelining chess_loop_range(2,)
			{
				A0 = aie::load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
				A1 = aie::load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;

				B0 = aie::load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * num_colB;
				B1 = aie::load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * num_colB;

				// matrix multiply and adding partial blocks
				C00.mac(A0, B0);
				C01.mac(A0, B1);
				C10.mac(A1, B0);
				C11.mac(A1, B1);
			}

			aie::store_v(pC1, C00.template to_vector<int8>(SHIFT)); pC1 +=MMUL::size_C;
			aie::store_v(pC1, C01.template to_vector<int8>(SHIFT)); pC1 +=MMUL::size_C;
			aie::store_v(pC2, C10.template to_vector<int8>(SHIFT)); pC2 +=MMUL::size_C;
			aie::store_v(pC2, C11.template to_vector<int8>(SHIFT)); pC2 +=MMUL::size_C;


		}

	}

}

