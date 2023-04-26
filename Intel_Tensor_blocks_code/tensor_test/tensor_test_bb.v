module tensor_test (
		input  wire        clk,                //                clk.clk
		input  wire [1:0]  feed_sel,           //           feed_sel.feed_sel
		input  wire        load_bb_one,        //        load_bb_one.load_bb_one
		input  wire        load_bb_two,        //        load_bb_two.load_bb_two
		input  wire        load_buf_sel,       //       load_buf_sel.load_buf_sel
		input  wire [7:0]  data_in_1,          //          data_in_1.data_in
		input  wire [7:0]  data_in_2,          //          data_in_2.data_in
		input  wire [7:0]  data_in_3,          //          data_in_3.data_in
		input  wire [7:0]  data_in_4,          //          data_in_4.data_in
		input  wire [7:0]  data_in_5,          //          data_in_5.data_in
		input  wire [7:0]  data_in_6,          //          data_in_6.data_in
		input  wire [7:0]  data_in_7,          //          data_in_7.data_in
		input  wire [7:0]  data_in_8,          //          data_in_8.data_in
		input  wire [7:0]  data_in_9,          //          data_in_9.data_in
		input  wire [7:0]  data_in_10,         //         data_in_10.data_in
		input  wire [7:0]  side_in_1,          //          side_in_1.data_in
		input  wire [7:0]  side_in_2,          //          side_in_2.data_in
		output wire [79:0] cascade_weight_out, // cascade_weight_out.cascade_weight_out
		output wire [24:0] int25_col_1,        //        int25_col_1.result_l
		output wire [24:0] int25_col_2,        //        int25_col_2.result_l,result_h
		output wire [24:0] int25_col_3         //        int25_col_3.result_h
	);
endmodule

