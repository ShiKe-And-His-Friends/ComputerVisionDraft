`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: one one
// 
// Create Date: 2024/10/17 10:47:19
// Design Name: 
// Module Name: zdr_cal_hist
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////
module hist_equal_plus
#(
parameter DW = 13,
parameter HIST_RAM_DW = 19,
parameter HIST_RAM_AW = 13,
parameter ACC_RAM_DW = 9,
parameter ACC_RAM_AW = 11,
parameter HIST_SUM_DW = 19


)
(
 input i_clk,
 input i_clk_2x,
 input i_rst,
 
 input i_field_vld,
 input i_line_vld,
 input [DW-1:0] i_img_data, 

 input i_hist_rd_vld,        //读直方图数据有效
 input [11:0] i_hist_rd_data,//读出的直方图统计数据
 input [7:0] i_gray_mid,  //均值配置
 input [7:0] i_hist_gain, //增益配置
 
 input [HIST_RAM_AW - 1:0] i_sum_min_aver, 
 input [HIST_RAM_AW - 1:0] i_sum_max_aver,
 input [DW - 1:0]  i_data_aver,  //14bit图像均值
 input [DW - 1:0] i_hist_min,
 input [DW - 1:0] i_hist_max,

 output o_field_vld,
 output o_line_vld,
 output [7:0] o_img_data 



);


localparam GRAY_LOW = 8'd0;
localparam GRAY_UP = 8'd255;

reg [HIST_RAM_DW - 1:0] hist_rd_data_latch;
reg [HIST_RAM_DW - 1:0] hist_data_limit;
reg 					hist_rd_vld_dly1;
reg                     hist_rd_vld_dly2;
reg                     hist_rd_vld_dly3;

reg [HIST_SUM_DW-1:0]   acc_data;
reg [ACC_RAM_AW-1:0]    addr_cur_wr;
reg [HIST_SUM_DW-1:0]   sum_cur_cnt,sum_cur_cnt_dly1,sum_cur_cnt_dly2,sum_cur_cnt_dly3;
reg [HIST_SUM_DW-1:0]   sum_cur_left;  

reg [7:0]               gray_cur;
reg [7:0]				sub_gray_cur;
reg [HIST_SUM_DW-1:0]   step_low;
reg [HIST_SUM_DW-1:0]   step_high;
reg [HIST_SUM_DW-1:0]   sub_aver_min;
reg [HIST_SUM_DW-1:0]   sub_aver_max;

wire [HIST_SUM_DW+7:0]  mult_sum_gray;
wire [HIST_SUM_DW+7:0]  mult_sum_gray_mid_high;
wire [HIST_SUM_DW+7:0]  mult_left_gain;

wire [HIST_SUM_DW+9:0]  mult_a,mult_b;   

reg  state;


always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		state <= 'b0;
	end
	else if(i_hist_rd_vld) begin
	    state <= ~state;
	end
	else begin
		state <= 'b0;
	end
end
//直方图灰度级min  max 与14bit均值之差
always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		step_low <= 'b0;
		step_high <= 'b0;
		sub_aver_min <= 'b0;
		sub_aver_max <= 'b0;
	end
	else begin
		sub_aver_min <= i_data_aver - i_hist_min;
		sub_aver_max <= i_hist_max - i_data_aver;
	end
end              


//multiplier
//#(
//  .LARENCY(1),
//  .WIDTH_A(HIST_SUM_DW+1),
//  .WIDTH_B(9)
//)sub_gray_mult_a
//(
// .i_rst(i_rst),
// .i_clk(i_clk_2x),
// .i_ce(1'b1),
 
// .i_dataa({1'b0,sum_cur_cnt}),
// .i_datab({1'b0,sub_gray_cur}),
// .o_dataout(mult_a)
//);


mult_gen_1 sub_gray_mult_a (
  .CLK(i_clk_2x),  // input wire CLK
  .A({1'b0,sum_cur_cnt}),      // input wire [19 : 0] A
  .B({1'b0,sub_gray_cur}),      // input wire [8 : 0] B
  .CE(1'b1),    // input wire CE
  .P(mult_a)      // output wire [28 : 0] P
);


assign mult_sum_gray = mult_a[HIST_SUM_DW+9:0];

//multiplier
//#(
//  .LARENCY(1),
//  .WIDTH_A(HIST_SUM_DW+1),
//  .WIDTH_B(9)
//)sub_gray_mult_b
//(
// .i_rst(i_rst),
// .i_clk(i_clk_2x),
// .i_ce(1'b1),
 
// .i_dataa({1'b0,sum_cur_cnt}),
// .i_datab({1'b0,i_hist_gain}),
// .o_dataout(mult_b)
//);

mult_gen_1 sub_gray_mult_b (
  .CLK(i_clk_2x),  // input wire CLK
  .A({1'b0,sum_cur_cnt}),      // input wire [19 : 0] A
  .B({1'b0,i_hist_gain}),      // input wire [8 : 0] B
  .CE(1'b1),    // input wire CE
  .P(mult_b)      // output wire [28 : 0] P
);


assign mult_left_gain = mult_b[HIST_SUM_DW+9:6];

reg[ACC_RAM_AW-1:0] addr_cur_wr_dly1;
reg[ACC_RAM_AW-1:0] addr_cur_wr_dly2;
reg[ACC_RAM_AW-1:0] addr_cur_wr_dly3;

always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		addr_cur_wr <= 'b0;
	end
	else begin
		addr_cur_wr_dly1 <= addr_cur_wr;
		addr_cur_wr_dly2 <= addr_cur_wr_dly1;
		if(i_hist_rd_vld)begin
			if(addr_cur_wr < i_data_aver)begin
				if(addr_cur_wr == 'b0)begin
					addr_cur_wr <= i_data_aver;
				end
				else begin
					addr_cur_wr <= addr_cur_wr - 1'b1;
				end
			end
			else if(addr_cur_wr <= {DW{1'b1}})begin
				addr_cur_wr <= addr_cur_wr + 1'b1;
			end
		end
		else begin
			addr_cur_wr <= i_data_aver - 1'b1;
		end
	end
	
end

always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		hist_rd_vld_dly1 <= 1'b0;
		hist_rd_vld_dly2 <= 1'b0;
		sum_cur_cnt_dly1 <= 'b0;
		sum_cur_cnt_dly2 <= 'b0;
		sum_cur_cnt_dly3 <= 'b0;
	end
	else begin
		sum_cur_cnt_dly1 <= sum_cur_cnt;
		sum_cur_cnt_dly2 <= sum_cur_cnt_dly1;
		sum_cur_cnt_dly3 <= sum_cur_cnt_dly2;
		
		hist_rd_vld_dly1 <= i_hist_rd_vld;
		hist_rd_vld_dly2 <= hist_rd_vld_dly1;
	end
end

always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		sum_cur_cnt <= 'b0;
    end
	else begin
		if(i_hist_rd_vld)begin
			if(state == 1'b0)begin
				if((addr_cur_wr == i_data_aver)||(mult_sum_gray > sum_cur_left))begin
					sum_cur_cnt <= i_hist_rd_data;
				end
				else begin
					sum_cur_cnt <= sum_cur_cnt + i_hist_rd_data;
				end
			end
		end
		else begin
			sum_cur_cnt <= 'b0;
		end
	end
end


wire [HIST_SUM_DW:0] sum_cur_cnt_buf;
assign sum_cur_cnt_buf = sum_cur_cnt + i_hist_rd_data;

always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		sum_cur_left <= 'b0;
	end
	else begin
	      if(i_hist_rd_vld)begin
	          if(state == 1'b0)begin
		    	  if(addr_cur_wr < i_data_aver)begin
		    		 if(mult_sum_gray > sum_cur_left)begin
		    			 if(sum_cur_left>= sum_cur_cnt)begin
		    				 sum_cur_left <= sum_cur_left - sum_cur_cnt;
		    			 end
		    		 end
		    	  end
		    	  else begin
		    		   if(addr_cur_wr == i_data_aver)begin
		    			  sum_cur_left <= i_sum_max_aver;
		    		   end
		    		   else begin
		    		       if(mult_sum_gray > sum_cur_left)begin
		    			      if(sum_cur_left >= sum_cur_cnt)begin
		    				      sum_cur_left <= sum_cur_left - sum_cur_cnt;
		    			      end
		    		       end
		    	       end
		          end
	          end
	      end
	      else begin
		     sum_cur_left <= i_sum_min_aver;
	      end	   
   end
end




always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		gray_cur <= 'b0;
	end
	else begin
		if(i_hist_rd_vld)begin
			if(state == 1'b0)begin
				if(addr_cur_wr == i_data_aver)begin
					gray_cur <= i_gray_mid - 1'b1;
					sub_gray_cur <= GRAY_UP - i_gray_mid - 1;
				end
				else if(addr_cur_wr < i_data_aver)begin
					if((mult_sum_gray > sum_cur_left) && (gray_cur > GRAY_LOW))begin
						gray_cur <= gray_cur - 1'b1;
						sub_gray_cur <= gray_cur - GRAY_LOW - 1'b1;
					end
				end
				else begin
					if((mult_sum_gray > sum_cur_left) &&(gray_cur < GRAY_UP))begin
						gray_cur <= gray_cur + 1'b1;
						sub_gray_cur <= GRAY_UP - gray_cur -1'b1;
					end
				end
			end
		end
		else begin
			gray_cur <= i_gray_mid - 1'b1;
			sub_gray_cur <= i_gray_mid - 1'b1;
		end
	end
end

reg [7:0] sub_gray_cur_dly1;
reg [7:0] gray_cur_dly1;
always@(posedge i_clk_2x or posedge i_rst)begin
	if(i_rst)begin
		sub_gray_cur_dly1 <= 'd0;
	end
	else begin
	    sub_gray_cur_dly1 <=sub_gray_cur;	
	end
end 

always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		gray_cur_dly1 <= 'b0;
	end
	else begin
		gray_cur_dly1 <= gray_cur;
	end
end

reg [4:0] dly_cnt;
reg       ram_wr;
reg [ACC_RAM_DW-1:0] ram_wr_data;
reg [ACC_RAM_AW-1:0] ram_wr_addr;

always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		ram_wr <= 'b0;
		ram_wr_addr <= 'b0;
		ram_wr_data <= 'b0;
	end
	else begin
		if(hist_rd_vld_dly1)begin
			ram_wr <= 1'b1;
		end
		else begin 
		    ram_wr <= 1'b0;
		end
		ram_wr_addr <= addr_cur_wr_dly1;
		ram_wr_data <= gray_cur_dly1;
	end
end

wire [ACC_RAM_AW-1:0] ram_rd_addr;
wire				  ram_rd;
wire [ACC_RAM_DW-1:0] ram_rd_q;

//single_dualport_ram
//#(
//	.ADDR_WIDTH_A(ACC_RAM_AW),
//	.ADDR_WIDTH_B(ACC_RAM_AW),
//	.WRITE_WIDTH_A(),
//	.READ_WIDTH_B ()
//)acc_ram
//(
//	.i_clka(i_clk),
//	.i_ena(1'b1),
//	.i_wea(ram_wr),
//	.i_addra(ram_wr_addr),
//	.i_dina(ram_wr_data),
	
//	.i_clkb(i_clk),
//	.i_addrb(ram_rd_addr),
//	.o_doutb(ram_rd_q)
	
//);



blk_mem_equalize acc_ram (
  .clka(i_clk),    // input wire clka
  .ena(1'b1),      // input wire ena
  .wea(ram_wr),      // input wire [0 : 0] wea
  .addra(ram_wr_addr),  // input wire [10 : 0] addra
  .dina(ram_wr_data),    // input wire [8 : 0] dina
  //
  .clkb(i_clk),    // input wire clkb
  .addrb(ram_rd_addr),  // input wire [10 : 0] addrb
  .doutb(ram_rd_q)  // output wire [8 : 0] doutb
);


assign ram_rd_addr = i_img_data[ACC_RAM_AW-1:0];
assign ram_rd = i_line_vld;

reg [7:0] data_out;
reg       field_vld_dly1,field_vld_dly2,field_vld_dly3;
reg       line_vld_dly1,line_vld_dly2,line_vld_dly3;

always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		data_out <= 'b0;
	end
	else if(line_vld_dly1)begin
		data_out <= ram_rd_q[7:0];
	end
	else begin
		data_out<= 8'b0;
	end
end

always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		field_vld_dly1 <= 'b0;
		field_vld_dly2 <= 'b0;
		field_vld_dly3 <= 'b0;
	end
	else begin
		field_vld_dly1 <= i_field_vld;
		field_vld_dly2 <= field_vld_dly1;
		field_vld_dly3 <= field_vld_dly2;
	end
end
always@(posedge i_clk or posedge i_rst)begin
	if(i_rst)begin
		line_vld_dly1 <= 'b0;
		line_vld_dly2 <= 'b0;
		line_vld_dly3 <= 'b0;
	end
	else begin
		line_vld_dly1 <= i_line_vld;
		line_vld_dly2 <= line_vld_dly1;
		line_vld_dly3 <= line_vld_dly2;
	end
end

assign o_img_data = data_out;
assign o_line_vld = line_vld_dly2;
assign o_field_vld = field_vld_dly2;


endmodule