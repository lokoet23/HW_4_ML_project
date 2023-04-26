	tensor_in u0 (
		.clk                  (_connected_to_clk_),                  //   input,   width = 1,                  clk.clk
		.clr0                 (_connected_to_clr0_),                 //   input,   width = 1,                 clr0.clr
		.clr1                 (_connected_to_clr1_),                 //   input,   width = 1,                 clr1.clr
		.ena                  (_connected_to_ena_),                  //   input,   width = 1,                  ena.ena
		.feed_sel             (_connected_to_feed_sel_),             //   input,   width = 2,             feed_sel.feed_sel
		.load_bb_one          (_connected_to_load_bb_one_),          //   input,   width = 1,          load_bb_one.load_bb_one
		.load_bb_two          (_connected_to_load_bb_two_),          //   input,   width = 1,          load_bb_two.load_bb_two
		.load_buf_sel         (_connected_to_load_buf_sel_),         //   input,   width = 1,         load_buf_sel.load_buf_sel
		.data_in_1            (_connected_to_data_in_1_),            //   input,   width = 8,            data_in_1.data_in
		.data_in_2            (_connected_to_data_in_2_),            //   input,   width = 8,            data_in_2.data_in
		.data_in_3            (_connected_to_data_in_3_),            //   input,   width = 8,            data_in_3.data_in
		.data_in_4            (_connected_to_data_in_4_),            //   input,   width = 8,            data_in_4.data_in
		.data_in_5            (_connected_to_data_in_5_),            //   input,   width = 8,            data_in_5.data_in
		.data_in_6            (_connected_to_data_in_6_),            //   input,   width = 8,            data_in_6.data_in
		.data_in_7            (_connected_to_data_in_7_),            //   input,   width = 8,            data_in_7.data_in
		.data_in_8            (_connected_to_data_in_8_),            //   input,   width = 8,            data_in_8.data_in
		.data_in_9            (_connected_to_data_in_9_),            //   input,   width = 8,            data_in_9.data_in
		.data_in_10           (_connected_to_data_in_10_),           //   input,   width = 8,           data_in_10.data_in
		.side_in_1            (_connected_to_side_in_1_),            //   input,   width = 8,            side_in_1.data_in
		.side_in_2            (_connected_to_side_in_2_),            //   input,   width = 8,            side_in_2.data_in
		.shared_exponent_data (_connected_to_shared_exponent_data_), //   input,   width = 8, shared_exponent_data.shared_exponent
		.cascade_weight_out   (_connected_to_cascade_weight_out_),   //  output,  width = 88,   cascade_weight_out.cascade_weight_out
		.bf24_col_1           (_connected_to_bf24_col_1_),           //  output,  width = 24,           bf24_col_1.result_l
		.bf24_col_2           (_connected_to_bf24_col_2_),           //  output,  width = 24,           bf24_col_2.result_l,result_h
		.bf24_col_3           (_connected_to_bf24_col_3_)            //  output,  width = 24,           bf24_col_3.result_h
	);

