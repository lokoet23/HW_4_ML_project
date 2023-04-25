
#ifndef FUNCTION_KERNELS_H
#define FUNCTION_KERNELS_H

  void blocked_matrix_mult(input_window_int8 * __restrict matA, input_window_int8 * __restrict matB,
						output_window_int8 * __restrict matC);

  void opt_blocked_matrix_mult(input_window_int8 * __restrict matA, input_window_int8 * __restrict matB,
  						output_window_int8 * __restrict matC);

#endif
