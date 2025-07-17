`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date:    14:01:51 10/27/2021 
// Design Name: 
// Module Name:    Line_Buffer 
// Project Name: 
// Target Devices: 
// Tool versions: 
// Description: 
//
// Dependencies: 
//
// Revision: 
// Revision 0.01 - File Created
// Additional Comments: 
//
//////////////////////////////////////////////////////////////////////////////////
module Line_Buffer #(
	parameter 			H_ALL = 680,
	parameter 			V_ALL = 520 )
(
input clk_i,
input rst_n,

input HS_in,
input VS_in,
input [13:0]Data_in,

output reg [10:0]h_cnt,
output reg [10:0]v_cnt,
output reg [13:0]line1_data1,
output reg [13:0]line1_data2,
output reg [13:0]line1_data3,
output reg [13:0]line1_data4,
output reg [13:0]line1_data5,
output reg [13:0]line2_data1,
output reg [13:0]line2_data2,
output reg [13:0]line2_data3,
output reg [13:0]line2_data4,
output reg [13:0]line2_data5,
output reg [13:0]line3_data1,
output reg [13:0]line3_data2,
output reg [13:0]line3_data3,
output reg [13:0]line3_data4,
output reg [13:0]line3_data5,
output reg [13:0]line4_data1,
output reg [13:0]line4_data2,
output reg [13:0]line4_data3,
output reg [13:0]line4_data4,
output reg [13:0]line4_data5,
output reg [13:0]line5_data1,
output reg [13:0]line5_data2,
output reg [13:0]line5_data3,
output reg [13:0]line5_data4,
output reg [13:0]line5_data5
    );


reg HS_in_r,HS_in_rr;	
reg VS_in_r,VS_in_rr;	
reg [13:0]data_in_dly1;
reg [13:0]data_in_dly2;
reg [13:0]data_in_dly3;
reg [13:0]data_in_dly4;
always @ (posedge clk_i or negedge rst_n)
begin
	if(!rst_n)
		begin
			HS_in_r  <=	1'd0;
			HS_in_rr	<=	1'd0;
			VS_in_r  <=	1'd0;
			VS_in_rr	<=	1'd0;
			data_in_dly1	<=	1'd0;
			data_in_dly2 	<=	1'd0;
			data_in_dly3	<=	1'd0;
			data_in_dly4	<=	1'd0;
		end
	else
		begin
			HS_in_r	<=	HS_in;
			HS_in_rr	<=	HS_in_r;
			VS_in_r	<=	VS_in;
			VS_in_rr	<=	VS_in_r;
			data_in_dly1   <=  Data_in;
			data_in_dly2   <=  data_in_dly1;
			data_in_dly3   <=  data_in_dly2;
			data_in_dly4   <=  data_in_dly3;
		end
end
		
//reg [9:0]h_cnt;
//reg [9:0]v_cnt;
reg Cnt_run; 
always @ (posedge clk_i or negedge rst_n)
begin
	if(!rst_n)
		begin
			h_cnt	<=	1'd0;
			v_cnt	<=	1'd0;
			Cnt_run	<=	1'd0;
		end
	else
		begin		
			if(!VS_in_rr && VS_in_r)
				Cnt_run <= 0;
			else if(!HS_in_rr && HS_in_r)
				Cnt_run <= 1;
            else ;

			if(Cnt_run)
				begin
					if(h_cnt == H_ALL-1)
						begin
							h_cnt <= 0;
							if(v_cnt == V_ALL)
								begin
									v_cnt <= 0;
									Cnt_run <= 0;
								end
							else
								v_cnt <= v_cnt + 1'b1;
						end
					else
						h_cnt <= h_cnt + 1;
				end
			else
				begin
					h_cnt <= 0;
					v_cnt <= 0;
				end
		end
end

wire [13:0]line5_dout;
wire [13:0]line4_dout;
wire [13:0]line3_dout;
wire [13:0]line2_dout;
wire [13:0]line1_dout;
assign line5_dout = data_in_dly4;

reg line4_wr;
reg line4_rd;
reg line3_wr;
reg line3_rd;
reg line2_wr;
reg line2_rd;
reg line1_wr;
reg line1_rd;
always @(posedge clk_i)
begin
	if(!rst_n)
		begin
			line4_wr <= 0;
			line4_rd <= 0;
			line3_wr <= 0;
			line3_rd <= 0;
			line2_wr <= 0;
			line2_rd <= 0;
			line1_wr <= 0;
			line1_rd <= 0;
		end
	else 
		begin
			line4_wr <= v_cnt >= 0 && v_cnt < 512 && h_cnt >= 1 && h_cnt < 1 + 640;
			line4_rd <= v_cnt >= 1 && v_cnt < 513 && h_cnt >= 0 && h_cnt < 0 + 640;
			line3_wr <= v_cnt >= 1 && v_cnt < 513 && h_cnt >= 1 && h_cnt < 1 + 640;
			line3_rd <= v_cnt >= 2 && v_cnt < 514 && h_cnt >= 0 && h_cnt < 0 + 640;
			line2_wr <= v_cnt >= 2 && v_cnt < 514 && h_cnt >= 1 && h_cnt < 1 + 640;
			line2_rd <= v_cnt >= 3 && v_cnt < 515 && h_cnt >= 0 && h_cnt < 0 + 640;
			line1_wr <= v_cnt >= 3 && v_cnt < 515 && h_cnt >= 1 && h_cnt < 1 + 640;
			line1_rd <= v_cnt >= 4 && v_cnt < 516 && h_cnt >= 0 && h_cnt < 0 + 640;
		end
end

always @(posedge clk_i)
begin
	if(!rst_n)
		begin
			line5_data1 <= 0;
			line5_data2 <= 0;
			line5_data3 <= 0;
			line5_data4 <= 0;
			line5_data5 <= 0;
			line4_data1 <= 0;
			line4_data2 <= 0;
			line4_data3 <= 0;
			line4_data4 <= 0;
			line4_data5 <= 0;
			line3_data1 <= 0;
			line3_data2 <= 0;
			line3_data3 <= 0;
			line3_data4 <= 0;
			line3_data5 <= 0;
			line2_data1 <= 0;
			line2_data2 <= 0;
			line2_data3 <= 0;
			line2_data4 <= 0;
			line2_data5 <= 0;
			line1_data1 <= 0;
			line1_data2 <= 0;
			line1_data3 <= 0;
			line1_data4 <= 0;
			line1_data5 <= 0;
		end
	else 
		begin
             if(h_cnt == 4)
                begin
                    line5_data1 <= line5_data4;	
                    line5_data2 <= line5_data4;
                    line4_data1 <= line4_data4;	
                    line4_data2 <= line4_data4;
                    line3_data1 <= line3_data4;	
                    line3_data2 <= line3_data4;
                    line2_data1 <= line2_data4;	
                    line2_data2 <= line2_data4;
                    line1_data1 <= line1_data4;	
                    line1_data2 <= line1_data4;
                end
            else
                begin
                    line5_data1 <= line5_data2;	
                    line5_data2 <= line5_data3;
                    line4_data1 <= line4_data2;	
                    line4_data2 <= line4_data3;
                    line3_data1 <= line3_data2;	
                    line3_data2 <= line3_data3;
                    line2_data1 <= line2_data2;	
                    line2_data2 <= line2_data3;
                    line1_data1 <= line1_data2;	
                    line1_data2 <= line1_data3;
                end
                
            line5_data3 <= line5_data4;
            line5_data4 <= line5_data5;
            if(v_cnt == 512)
                line5_data5 <= line4_dout;
            else if(v_cnt == 513)
                line5_data5 <= line3_dout;
            else
                line5_data5 <= line5_dout;
            
            line4_data3 <= line4_data4;
            line4_data4 <= line4_data5;
            if(v_cnt == 513)
                line4_data5 <= line3_dout;
            else
                line4_data5 <= line4_dout;

            line3_data3 <= line3_data4;
            line3_data4 <= line3_data5;
            line3_data5 <= line3_dout;	

            line2_data3 <= line2_data4;
            line2_data4 <= line2_data5;
            if(v_cnt == 2)
                line2_data5 <= line3_dout;	
            else
                line2_data5 <= line2_dout;	

            line1_data3 <= line1_data4;
            line1_data4 <= line1_data5;
            if(v_cnt == 2)
                line1_data5 <= line3_dout;	
            else if(v_cnt == 3)
                line1_data5 <= line2_dout;	
            else
                line1_data5 <= line1_dout;	
		end
end

wire [9:0]line4_fifo_cnt;
wire [9:0]line3_fifo_cnt;
wire [9:0]line2_fifo_cnt;
wire [9:0]line1_fifo_cnt;


line_fifo line4_fifo (

  .rst(!rst_n || !Cnt_run),                  // input wire rst
  .wr_clk(clk_i),            // input wire wr_clk
  .rd_clk(clk_i),            // input wire rd_clk
  .din(line5_dout),                  // input wire [13 : 0] din
  .wr_en(line4_wr),              // input wire wr_en
  .rd_en(line4_rd),              // input wire rd_en
  .dout(line4_dout),                // output wire [13 : 0] dout
  .full(),                // output wire full
  .empty(),              // output wire empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);

line_fifo line3_fifo (
  .rst(!rst_n || !Cnt_run),                  // input wire rst
  .wr_clk(clk_i),            // input wire wr_clk
  .rd_clk(clk_i),            // input wire rd_clk
  .din(line4_dout),                  // input wire [13 : 0] din
  .wr_en(line3_wr),              // input wire wr_en
  .rd_en(line3_rd),              // input wire rd_en
  .dout(line3_dout),                // output wire [13 : 0] dout
  .full(),                // output wire full
  .empty(),              // output wire empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);
line_fifo line2_fifo (
  .rst(!rst_n || !Cnt_run),                  // input wire rst
  .wr_clk(clk_i),            // input wire wr_clk
  .rd_clk(clk_i),            // input wire rd_clk
  .din(line3_dout),                  // input wire [13 : 0] din
  .wr_en(line2_wr),              // input wire wr_en
  .rd_en(line2_rd),              // input wire rd_en
  .dout(line2_dout),                // output wire [13 : 0] dout
  .full(),                // output wire full
  .empty(),              // output wire empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);

line_fifo line1_fifo (
  .rst(!rst_n || !Cnt_run),                  // input wire rst
  .wr_clk(clk_i),            // input wire wr_clk
  .rd_clk(clk_i),            // input wire rd_clk
  .din(line2_dout),                  // input wire [13 : 0] din
  .wr_en(line1_wr),              // input wire wr_en
  .rd_en(line1_rd),              // input wire rd_en
  .dout(line1_dout),                // output wire [13 : 0] dout
  .full(),                // output wire full
  .empty(),              // output wire empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);

endmodule
