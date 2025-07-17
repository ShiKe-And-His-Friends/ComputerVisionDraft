`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2024/10/27 16:09:40
// Design Name: 
// Module Name: zrc_top
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


module zrc_top
#(
   parameter DW            = 14,
   parameter IMAGE_PIX_NUM = 21'd327680,
   parameter HIST_RAM_AW =  14,
   parameter HIST_RAM_DW = 14,
   parameter ACC_RAM_DW = 8,
   parameter ACC_RAM_AW = 16,
   parameter HIST_SUM_DW = 21
)


(

    input                           i_clk,
    input                           i_rst,

    input                           i_field_vld,
    input                           i_line_vld,
    input [DW - 1:0]                i_img_data,
    input [7:0]                     i_img_data_h,
    
    output                          o_field_vld,
    output                          o_line_vld,
    input [7:0]                     o_img_data,      
    
    input [HIST_RAM_DW - 1:0]       i_drc_th,
    input [HIST_RAM_DW - 1:0]       i_drc_limit_up,    
    input [HIST_RAM_DW - 1:0]       i_drc_limit_down  

    );
    
wire                                field_limit_w;
wire                                line_limit_w;
wire   [DW - 1:0]                   data_limit_w;

wire                                bc_cal_en_w;
wire                                hist_wr_field_vld_w;
wire                                hist_wr_line_vld_w;
wire  [HIST_RAM_DW - 1:0]           hist_wr_data_w;

wire                                hist_rd_vld_w;
wire  [HIST_RAM_DW - 1:0]           hist_rd_data_w; 

wire  [DW - 1:0]                    hist_max_w;
wire  [DW - 1:0]                    hist_min_w;
wire  [DW - 1:0]                    data_avg_w;
wire  [DW - 1:0]                    data_range_w;

wire                                field_vld_hist;
wire                                line_vld_hist;
wire  [ACC_RAM_DW - 1:0]            img_data_hist;

wire                                field_vld_line;
wire                                line_vld_line;
wire  [ACC_RAM_DW - 1:0]            img_data_line;


hist_gen 
#(
  .DW(14),
  .IMAGE_WIDTH(640),
  .IMAGE_HEIGHT(512),
  .HIST_RAM_AW(14),
  .HIST_RAM_DW(12),
  .HIST_SUM_DW(19)
) 
hist_gen_inst 
(
  .i_clk(i_clk),
  .i_rst(i_rst),
  
  .i_field_vld(i_field_vld),
  .i_line_vld(i_line_vld),
  .i_img_data(i_img_data),
//  .i_img_data_h(),
  .i_limit_up(20'd4096),
  .i_limit_down(20'd4096),
  .o_field_limit(field_limit_w),
  .o_line_limit(line_limit_w),
  .o_data_limit(data_limit_w),
  .o_bc_cal_en(bc_cal_en_w),
  .o_hist_wr_field_vld(hist_wr_field_vld_w),
  .o_hist_wr_line_vld(hist_wr_line_vld_w),
  .o_hist_wr_data(hist_wr_data_w),
  .o_hist_rd_vld(hist_rd_vld_w),
  .o_hist_rd_data(hist_rd_data_w),
  .o_hist_max(hist_max_w),
  .o_hist_min(hist_min_w),
  .o_data_avg(data_avg_w)
);

   zrc_hist_map #(
     .DW(14),
     .ACC_RAM_DW(10),
     .ACC_RAM_AW(14),
     .HIST_RAM_AW(14),
     .HIST_RAM_DW(12),
     .HIST_SUM_DW(21)
   ) inst_zrc_hist_map (
     .i_clk(i_clk),
     .i_rst(i_rst),
     
     .i_field_vld(field_limit_w),
     .i_line_vld(line_limit_w),
     .i_img_data(data_limit_w),
     
     .i_freeze(1'b0),
     .i_hist_wr_field_vld(hist_wr_field_vld_w),
     .i_hist_wr_line_vld(hist_wr_line_vld_w),
     .i_hist_wr_data(hist_wr_data_w),
     .i_hist_rd_vld(hist_rd_vld_w),
     .i_hist_rd_data(hist_rd_data_w),
     
     .i_gray_limit(8'd255),
     .i_data_range(14'd1023),
     .i_acc_th(i_drc_th) ,
     .o_field_vld(o_field_vld),
     .o_line_vld(o_line_vld),
     .o_img_data(o_img_data)     
   );

//line_map
//#(
//    .DW (14),
//    .COMP_VALUE(128),
//    .FILTER_N(1)
//)
//(
//  .i_clk(i_clk),
//  .i_rst(i_rst),
  
//  .i_freeze(1'b0),
//  .i_bc_mode(2'b0),
//  .i_b_exp (8'd128),
//  .i_c_exp (8'd128),
//  .i_b_manual (24'd1000),
//  .i_b_manual (16'd500),  
//  .i_gray_limit(8'd255),
//  .i_c_limit    ( 16'd614),
  
//  .i_hist_max  (hist_max_w),
//  .i_hist_min  (hist_min_w),  
//  .i_data_avg  (data_avg_w),
//  .i_bc_enable(bc_cal_en_w),
  
//  .i_field_vld(field_limit_w),
//  .i_line_vld(line_limit_w),
//  .i_img_data(data_limit_w),
  
//  .o_field_vld(field_vld_line),
//  .o_line_vld(line_vld_line),
//  .o_img_data(img_data_line),
  
//  .o_freeze_sync( ),
//  .o_gain (),
//  .o_offset(),
//  .o_data_range(data_range_w)

//);

//mix_map
//#(
//    .DW(14)

//)
//(
//  .i_clk(i_clk),
//  .i_rst(i_rst),
  
//  .i_mix_ctrl(2'd2),
//  .i_data_range(data_range_w),
  
//  .i_field_vld_l(field_vld_line),
//  .i_line_vld_l(line_vld_line),
//  .i_img_data_l(img_data_line),
  
//   .i_field_vld_h(field_vld_hist),
//   .i_line_vld_h(line_vld_hist),
//   .i_img_data_h(img_data_hist),
   
//   .o_field_vld(o_field_vld),
//   .o_line_vld(o_line_vld),
//   .o_img_data(o_img_data)

//);


endmodule
