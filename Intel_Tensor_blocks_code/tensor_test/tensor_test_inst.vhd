	component tensor_test is
		port (
			clk                : in  std_logic                     := 'X';             -- clk
			feed_sel           : in  std_logic_vector(1 downto 0)  := (others => 'X'); -- feed_sel
			load_bb_one        : in  std_logic                     := 'X';             -- load_bb_one
			load_bb_two        : in  std_logic                     := 'X';             -- load_bb_two
			load_buf_sel       : in  std_logic                     := 'X';             -- load_buf_sel
			data_in_1          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_2          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_3          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_4          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_5          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_6          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_7          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_8          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_9          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			data_in_10         : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			side_in_1          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			side_in_2          : in  std_logic_vector(7 downto 0)  := (others => 'X'); -- data_in
			cascade_weight_out : out std_logic_vector(79 downto 0);                    -- cascade_weight_out
			int25_col_1        : out std_logic_vector(24 downto 0);                    -- result_l
			int25_col_2        : out std_logic_vector(24 downto 0);                    -- result_l,result_h
			int25_col_3        : out std_logic_vector(24 downto 0)                     -- result_h
		);
	end component tensor_test;

	u0 : component tensor_test
		port map (
			clk                => CONNECTED_TO_clk,                --                clk.clk
			feed_sel           => CONNECTED_TO_feed_sel,           --           feed_sel.feed_sel
			load_bb_one        => CONNECTED_TO_load_bb_one,        --        load_bb_one.load_bb_one
			load_bb_two        => CONNECTED_TO_load_bb_two,        --        load_bb_two.load_bb_two
			load_buf_sel       => CONNECTED_TO_load_buf_sel,       --       load_buf_sel.load_buf_sel
			data_in_1          => CONNECTED_TO_data_in_1,          --          data_in_1.data_in
			data_in_2          => CONNECTED_TO_data_in_2,          --          data_in_2.data_in
			data_in_3          => CONNECTED_TO_data_in_3,          --          data_in_3.data_in
			data_in_4          => CONNECTED_TO_data_in_4,          --          data_in_4.data_in
			data_in_5          => CONNECTED_TO_data_in_5,          --          data_in_5.data_in
			data_in_6          => CONNECTED_TO_data_in_6,          --          data_in_6.data_in
			data_in_7          => CONNECTED_TO_data_in_7,          --          data_in_7.data_in
			data_in_8          => CONNECTED_TO_data_in_8,          --          data_in_8.data_in
			data_in_9          => CONNECTED_TO_data_in_9,          --          data_in_9.data_in
			data_in_10         => CONNECTED_TO_data_in_10,         --         data_in_10.data_in
			side_in_1          => CONNECTED_TO_side_in_1,          --          side_in_1.data_in
			side_in_2          => CONNECTED_TO_side_in_2,          --          side_in_2.data_in
			cascade_weight_out => CONNECTED_TO_cascade_weight_out, -- cascade_weight_out.cascade_weight_out
			int25_col_1        => CONNECTED_TO_int25_col_1,        --        int25_col_1.result_l
			int25_col_2        => CONNECTED_TO_int25_col_2,        --        int25_col_2.result_l,result_h
			int25_col_3        => CONNECTED_TO_int25_col_3         --        int25_col_3.result_h
		);

