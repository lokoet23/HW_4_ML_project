module top(
	clk,
	load_bank_1,
	load_bank_2,
	cascade_in,
	data_out_colm1,
	data_out_colm2,
	data_out_colm3
	);
	
input load_bank_1, load_bank_2;
input clk;
input [79:0] cascade_in;
output [24:0] data_out_colm1, data_out_colm2, data_out_colm3;

wire [79:0] cascade_prev_out_tb0_in;

tensor_test initial (
                .clk                (clk),                //   input,   width = 1,                clk.clk
                .feed_sel           (2'h0),           //   input,   width = 2,           feed_sel.feed_sel
                .load_bb_one        (load_bank_1),        //   input,   width = 1,        load_bb_one.load_bb_one
                .load_bb_two        (load_bank_2),        //   input,   width = 1,        load_bb_two.load_bb_two
                .load_buf_sel       (1'h0),       //   input,   width = 1,       load_buf_sel.load_buf_sel
					 .data_in_1   (cascade_in[7:0]),   //   input,   width = 8,   data_in_1.data_in
                .data_in_2   (cascade_in[15:8]),   //   input,   width = 8,   data_in_2.data_in
                .data_in_3   (cascade_in[23:16]),   //   input,   width = 8,   data_in_3.data_in
                .data_in_4   (cascade_in[31:24]),   //   input,   width = 8,   data_in_4.data_in
                .data_in_5   (cascade_in[39:32]),   //   input,   width = 8,   data_in_5.data_in
                .data_in_6   (cascade_in[47:40]),   //   input,   width = 8,   data_in_6.data_in
                .data_in_7   (cascade_in[55:48]),   //   input,   width = 8,   data_in_7.data_in
                .data_in_8   (cascade_in[63:56]),   //   input,   width = 8,   data_in_8.data_in
                .data_in_9   (cascade_in[71:64]),   //   input,   width = 8,   data_in_9.data_in
                .data_in_10  (cascade_in[79:72]),  //   input,   width = 8,  data_in_10.data_in
                .side_in_1          (8'h0),          //   input,   width = 8,          side_in_1.data_in
                .side_in_2          (8'h0),          //   input,   width = 8,          side_in_2.data_in
                .cascade_weight_out (cascade_prev_out_tb0_in), //  output,  width = 80, cascade_weight_out.cascade_weight_out
                .int25_col_1        (data_out_colm1),        //  output,  width = 25,        int25_col_1.result_l
                .int25_col_2        (data_out_colm2),        //  output,  width = 25,        int25_col_2.result_l,result_h
                .int25_col_3        (data_out_colm3)         //  output,  width = 25,        int25_col_3.result_h
        );

endmodule