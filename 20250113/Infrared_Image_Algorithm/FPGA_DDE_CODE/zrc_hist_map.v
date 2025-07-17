`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company:  
// Engineer: hurdy
// 
// Create Date: 2024/10/21 21:38:38
// Design Name: 
// Module Name: zrc_hist_map
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


module zrc_hist_map
#(
   parameter DW  = 14,
   parameter ACC_RAM_DW = 10,
   parameter ACC_RAM_AW = 14,
   parameter HIST_RAM_AW =  14,
   parameter HIST_RAM_DW = 12,
   parameter HIST_SUM_DW = 19
)
(

    input                      i_clk ,
    input                      i_rst ,
    
    input                      i_field_vld,
    input                      i_line_vld,
    input [DW - 1:0]           i_img_data,
    
    input                            i_freeze,
    input                            i_hist_wr_field_vld,
    input                            i_hist_wr_line_vld,
    input [HIST_RAM_DW - 1:0]        i_hist_wr_data,
    
    input                            i_hist_rd_vld,
    input [HIST_RAM_DW - 1:0]        i_hist_rd_data,
    input [7:0]                      i_gray_limit,
    input [DW - 1:0]                 i_data_range,
    input [DW - 1:0]                 i_acc_th,
    
    output                           o_field_vld,
    output                           o_line_vld,
    output [7:0]                     o_img_data

);

reg [HIST_RAM_DW - 1:0]     hist_wr_data_latch;
reg                         hist_wr_field_vld_d1;
reg                         hist_wr_field_vld_d2;
reg                         hist_wr_line_vld_d1;
reg                         hist_wr_line_vld_d2;

reg [HIST_SUM_DW - 1:0]     hist_data_sum;
reg [HIST_SUM_DW - 1:0]     hist_data_sum_latch;




always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        hist_wr_data_latch <= {HIST_RAM_DW{1'b0}};
        hist_data_sum_latch <= {HIST_SUM_DW{1'b0}};
        hist_data_sum <= {HIST_SUM_DW{1'b0}};
        hist_wr_field_vld_d1 <= 1'b0;
        hist_wr_field_vld_d2 <= 1'b0;
        hist_wr_line_vld_d1  <= 1'b0;
        hist_wr_line_vld_d2  <= 1'b0;  
    end
    else begin
        if(i_hist_wr_line_vld)begin
            hist_wr_data_latch <= i_hist_wr_data;
        end
        else begin
            hist_wr_data_latch <= {HIST_RAM_DW{1'b0}};
        end
        hist_wr_field_vld_d1 <= i_hist_wr_field_vld;
        hist_wr_field_vld_d2 <= hist_wr_field_vld_d1;
        
        hist_wr_line_vld_d1 <= i_hist_wr_line_vld;
        hist_wr_line_vld_d2 <= hist_wr_line_vld_d1;
        if(hist_wr_line_vld_d1)begin
            if(hist_wr_data_latch <= i_acc_th[HIST_RAM_DW - 1:0])begin
                 hist_data_sum <= hist_data_sum + 1'b1;
            end
        end
        else if(hist_wr_line_vld_d2 & (~hist_wr_line_vld_d1))begin
            hist_data_sum_latch <= hist_data_sum;
            hist_data_sum <= {HIST_SUM_DW{1'b0}};
        end
    end 
end


reg [HIST_RAM_DW - 1:0]  hist_rd_data_latch;
reg [HIST_RAM_DW - 1:0]  hist_data_limit;
reg                      hist_rd_vld_d1;
reg                      hist_rd_vld_d2;
reg [HIST_SUM_DW - 1:0]  acc_data;

always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        hist_rd_data_latch <= {HIST_RAM_DW{1'b0}};
        hist_data_limit <= {HIST_RAM_DW{1'b0}};
        hist_rd_vld_d1 <= 1'd0;
        acc_data <= {HIST_SUM_DW{1'b0}};
    end
    else begin
        hist_rd_vld_d1 <= i_hist_rd_vld;
        hist_rd_vld_d2 <= hist_rd_vld_d1;
        if(i_hist_rd_vld)begin
            hist_rd_data_latch <= i_hist_rd_data;
        end
        else begin
            hist_rd_data_latch <= {HIST_RAM_DW{1'b0}};
        end
        //
        if(hist_rd_vld_d1)begin
            if(hist_rd_data_latch <= i_acc_th[HIST_RAM_DW - 1:0])begin
                hist_data_limit <= hist_rd_data_latch;
            end
            else begin
                hist_data_limit <= i_acc_th[HIST_RAM_DW - 1:0];
            end
        end
        else begin
            hist_data_limit <= {HIST_RAM_DW{1'b0}};
        end
        //
        if(hist_rd_vld_d2)begin
            acc_data <= acc_data + hist_data_limit;
        end
        else begin
            acc_data <= {HIST_SUM_DW{1'b0}};
        end
    end
end

wire [23:0] hist_div_denom;
wire [31:0] hist_div_numer;
wire [31:0] hist_div_quotient;
wire [55:0] hist_div_quotient_temp;

assign hist_div_denom = {2'd0,hist_data_sum_latch};
assign hist_div_numer = {2'd0,acc_data,{ACC_RAM_DW{1'b0}}};

div_32_24 u_hist_div (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(hist_div_denom),      // input wire [23 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(hist_div_numer),    // input wire [31 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(hist_div_quotient_temp)            // output wire [55 : 0] m_axis_dout_tdata
);

assign hist_div_quotient = hist_div_quotient_temp[HIST_SUM_DW + ACC_RAM_DW + 23:24];

reg [4:0]              dly_cnt;
reg                    ram_wr;
reg [ACC_RAM_DW - 1:0] ram_wr_data;
reg [ACC_RAM_AW - 1:0] ram_wr_addr;
reg                    hist_rd_vld_d3;

always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        dly_cnt <= 5'd0;
        ram_wr <= 1'b0;
        ram_wr_data <= {ACC_RAM_DW{1'b0}}; 
        ram_wr_addr <= {ACC_RAM_AW{1'b0}};
        hist_rd_vld_d3 <= 1'b0;
    end
    else begin
        hist_rd_vld_d3 <= hist_rd_vld_d2;
        if(hist_rd_vld_d3)begin
            if(dly_cnt < 5'd31)begin
                dly_cnt <= dly_cnt + 1;
                ram_wr <= 1'b0;
            end
            else begin
                ram_wr <= 1'b1;
            end
        end
        else begin
            if(dly_cnt == 0)begin
                ram_wr <= 1'b0;
            end
            else begin
                dly_cnt <= dly_cnt - 1;
                ram_wr <= 1'b1;
            end
        end
        //
        if(hist_div_quotient[HIST_SUM_DW + ACC_RAM_DW - 1:ACC_RAM_DW] == 0)begin
            ram_wr_data <= hist_div_quotient[ACC_RAM_DW - 1:0];
        end
        else begin
            ram_wr_data <= {ACC_RAM_DW{1'b1}};
        end
        //
        if(ram_wr)begin
            ram_wr_addr <= ram_wr_addr + 1;
        end
        else begin
            ram_wr_addr <= {ACC_RAM_AW{1'b0}};
        end
    end
 end
 
 wire [ACC_RAM_AW - 1:0] ram_rd_addr;
 wire                    ram_rd;
 wire [ACC_RAM_DW - 1:0] ram_rd_q;
 
ad_drc_hist_ram u_ad_drc_hist_ram (
  .clka(i_clk),            // input wire clka
  .rsta(i_rst),            // input wire rsta
  .ena(1'b1),              // input wire ena
  .wea(ram_wr),              // input wire [0 : 0] wea
  .addra(ram_wr_addr),          // input wire [13 : 0] addra
  .dina(ram_wr_data),            // input wire [9 : 0] dina
  .douta(),          // output wire [9 : 0] douta
  //
  .clkb(i_clk),            // input wire clkb
  .rstb(i_rst),            // input wire rstb
  .web(1'b0),              // input wire [0 : 0] web
  .addrb(ram_rd_addr),          // input wire [13 : 0] addrb
  .dinb({ACC_RAM_DW{1'b0}}),            // input wire [9 : 0] dinb
  .doutb(ram_rd_q)          // output wire [9 : 0] doutb
);

assign ram_rd_addr = i_img_data[ACC_RAM_AW - 1:0];
assign ram_rd = i_line_vld;

wire [8:0]                    hist_mult_data_a;
wire [ACC_RAM_DW - 1:0]       hist_mult_data_b;
wire [ACC_RAM_DW + 9 - 1:0]   hist_mult_result;
reg  [8:0]                    gray_range_tmp;
wire [8:0]                    gray_start_w;
reg  [7:0]                    gray_start;
wire                          field_d;
wire                          line_d;
wire                          gener_field_vld;
reg  [9:0]                    data_out_tmp;   
reg  [7:0]                    data_out; 

mult_sun_gray u_mult_sun_gray (
  .CLK(i_clk),  // input wire CLK
  .A(hist_mult_data_a),      // input wire [8 : 0] A
  .B({1'b0,hist_mult_data_b}),      // input wire [8 : 0] B
  .P(hist_mult_result)      // output wire [17 : 0] P
);

assign hist_mult_data_a = (i_line_vld | line_d)? gray_range_tmp:{1'b0,i_gray_limit};
assign hist_mult_data_b = (i_line_vld | line_d)? {1'b0,ram_rd_q} : {1'b0,i_data_range[ACC_RAM_DW - 1:0]};

assign gray_start_w = {1'b0,i_gray_limit} - gray_range_tmp;

always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        gray_range_tmp <= 9'd0;
        gray_start <= 8'd0;
    end
    else begin
        if((i_line_vld | line_d) == 1'b0)begin
            if(i_data_range <= $unsigned(128))begin
                gray_range_tmp <= hist_mult_result[15:7];
            end
            else begin
                gray_range_tmp <= {1'b0,i_gray_limit};
            end
            gray_start <= gray_start_w[8:1];
        end
    end
end

 data_delay #(
    .DATA_WIDTH(2),
    .DLY_NUM(5)
    )
 u_field_vld_dly640 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data({i_field_vld,i_line_vld}),
      .o_data({field_d,line_d})
    );
 
 always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        data_out_tmp <= {(9+1){1'b0}};
        data_out     <= {8{1'b0}};
    end
    else begin
        data_out_tmp <= {1'b0,hist_mult_result[ACC_RAM_DW + 9 - 1:ACC_RAM_DW]} + gray_start;
        if(data_out_tmp < i_gray_limit)begin
            data_out <= data_out_tmp[7:0];
        end
        else begin
            data_out <= i_gray_limit;
        end
    end
 end
 
 
 assign o_img_data  = data_out;
 assign o_field_vld = field_d;
 assign o_line_vld  = line_d;


endmodule
