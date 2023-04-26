// (C) 2001-2021 Intel Corporation. All rights reserved.
// Your use of Intel Corporation's design tools, logic functions and other 
// software and tools, and its AMPP partner logic functions, and any output 
// files from any of the foregoing (including device programming or simulation 
// files), and any associated documentation or information are expressly subject 
// to the terms and conditions of the Intel Program License Subscription 
// Agreement, Intel FPGA IP License Agreement, or other applicable 
// license agreement, including, without limitation, that your use is for the 
// sole purpose of programming logic devices manufactured by Intel and sold by 
// Intel or its authorized distributors.  Please refer to the applicable 
// agreement for further details.



// synopsys translate_off
`timescale 1 ps / 1 ps
// synopsys translate_on
module	tensor_test_dsp_prime_10_qbfmubi	(
			clk,
			feed_sel,
			load_bb_one,
			load_bb_two,
			load_buf_sel,
			data_in_1,
			data_in_2,
			data_in_3,
			data_in_4,
			data_in_5,
			data_in_6,
			data_in_7,
			data_in_8,
			data_in_9,
			data_in_10,
			side_in_1,
			side_in_2,
			cascade_weight_out,
			int25_col_1,
			int25_col_2,
			int25_col_3);

			input  clk;
			input [1:0] feed_sel;
			input  load_bb_one;
			input  load_bb_two;
			input  load_buf_sel;
			input [7:0] data_in_1;
			input [7:0] data_in_2;
			input [7:0] data_in_3;
			input [7:0] data_in_4;
			input [7:0] data_in_5;
			input [7:0] data_in_6;
			input [7:0] data_in_7;
			input [7:0] data_in_8;
			input [7:0] data_in_9;
			input [7:0] data_in_10;
			input [7:0] side_in_1;
			input [7:0] side_in_2;
			output [79:0] cascade_weight_out;
			output [24:0] int25_col_1;
			output [24:0] int25_col_2;
			output [24:0] int25_col_3;

			wire [79:0] cascade_weight_out_w ;
			wire [24:0] int25_col_1_w ;
			wire [24:0] int25_col_2_w ;
			wire [24:0] int25_col_3_w ;
			wire [79:0] cascade_weight_out = cascade_weight_out_w [79:0] ;
			wire [24:0] int25_col_1 = int25_col_1_w [24:0] ;
			wire [24:0] int25_col_2 = int25_col_2_w [24:0] ;
			wire [24:0] int25_col_3 = int25_col_3_w [24:0] ;

			fourteennm_dsp_prime		fourteennm_dsp_prime_component (
						 .clk (clk),
						 .feed_sel (feed_sel),
						 .load_bb_one (load_bb_one),
						 .load_bb_two (load_bb_two),
						 .load_buf_sel (load_buf_sel),
						 .cascade_weight_out (cascade_weight_out_w),
						 .data_in({side_in_2,side_in_1,data_in_10,data_in_9,data_in_8,data_in_7,data_in_6,data_in_5,data_in_4,data_in_3,data_in_2,data_in_1}),
						 .result_l({int25_col_2_w[24:12],int25_col_1_w[24:0]}),
						 .result_h({int25_col_3_w[24:0],int25_col_2_w[11:0]}));
			defparam
						fourteennm_dsp_prime_component.dsp_mode = "tensor_fxp",
						fourteennm_dsp_prime_component.dsp_sel_int4 = "select_int8",
						fourteennm_dsp_prime_component.dsp_cascade = "cascade_disabled",
						fourteennm_dsp_prime_component.dsp_fp32_sub_en = "float_sub_disabled";


endmodule




