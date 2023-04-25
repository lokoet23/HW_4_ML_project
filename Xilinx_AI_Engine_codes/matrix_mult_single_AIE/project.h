
#include <adf.h>
#include "kernels.h"
#include "include.h"

using namespace adf;

class simpleGraph : public adf::graph {
private:
  kernel simlpe_k;
public:
  input_plio  in_1;
  input_plio  in_2;
  output_plio out;
  simpleGraph(){
    
    in_1  = input_plio::create(plio_32_bits, "data/matA.txt");
    in_2  = input_plio::create(plio_32_bits, "data/matB.txt");
    out = output_plio::create(plio_32_bits, "data/matC.txt");

    simlpe_k = kernel::create(blocked_matrix_mult);
//    simlpe_k = kernel::create(opt_blocked_matrix_mult);

    connect< window<rowA*colA*1> > net0 (in_1.out[0], simlpe_k.in[0]);
    connect< window<colA*colB*1> > net1 (in_2.out[0], simlpe_k.in[1]);
    connect< window<rowA*colB*1> > net2 (simlpe_k.out[0], out.in[0]);

    source(simlpe_k) = "kernels/kernels.cc";

    runtime<ratio>(simlpe_k) = 0.6;


  }
};
