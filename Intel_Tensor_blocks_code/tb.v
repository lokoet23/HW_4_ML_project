`timescale 1 ps / 1 ps
module tb();

reg clk;
reg load_bank_1, load_bank_2;
reg [79:0] cascade_in;
//reg [79:0] data_in_all;
wire [24:0] data_out_colm1, data_out_colm2, data_out_colm3;

top top_inst(.clk(clk), .load_bank_1(load_bank_1), .load_bank_2(load_bank_2), 
		.cascade_in(cascade_in),
		.data_out_colm1(data_out_colm1), .data_out_colm2(data_out_colm2), .data_out_colm3(data_out_colm3));		
		
// clock generator
always #10 clk=~clk;
		
initial
begin		
	
	clk = 0;
	
	#20					// bank 1 load, remember load the data of the next clock cycle
	load_bank_1 = 1;
	load_bank_2 = 0;
	
	//data_in_all = {10{8'd0}};
	
	#20				// first data
	cascade_in = {10{8'd1}};
	
	#20				// second data
	cascade_in = {10{8'd2}};
	
	#20				// third data
	load_bank_1 = 0;
	load_bank_2 = 1;	// bank 2 load, next cc
	
	cascade_in = {10{8'd3}};
	
	#20					// first data, bank 2
	cascade_in = {10{8'd4}};
	
	#20				// second data, bank 2
	cascade_in = {10{8'd4}};
	
	#20			// third data, bank 2
	load_bank_1 = 1; // 
	load_bank_2 = 0;
	
	cascade_in = {10{8'd4}};
	
	
	#20				// first data
	cascade_in = {10{8'd6}};
	
	#20				// second data
	cascade_in = {10{8'd7}};
	
	#20				// third data
	load_bank_1 = 0;
	load_bank_2 = 1;	// bank 2 load, next cc
	
	cascade_in = {10{8'd8}};
	
	
	#20					// first data, bank 2
	cascade_in = {10{8'd5}};
	
	#20				// second data, bank 2
	cascade_in = {10{8'd5}};
	
	#20			// third data, bank 2
	load_bank_1 = 0; // 
	load_bank_2 = 0;
	cascade_in = {10{8'd5}};
	
	
end

initial #1000 $finish;
		
		
endmodule