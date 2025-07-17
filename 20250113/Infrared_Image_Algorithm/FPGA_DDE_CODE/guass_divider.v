`timescale 1ns / 1ps
`include "top_define.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: Hurdy
// 
// Create Date: 2024/09/23 13:28:51
// Design Name: 
// Module Name: guass_divider
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
//guass_demo_matlab
//0.0673  0.1248  0.0673     //0.075  0.123  0.075 
//0.1248  0.2314  0.1248     //0.123  0.204  0.123
//0.0673  0.1248  0.0673     //0.075  0.123  0.075

//guass_demo_FPGA_vital
//298  552   298
//552  1024  552
//298  552   298

//guass_demo_FPGA_AD
//377  621   377
//552  1024  552
//377  621   377

//////////////////////////////////////////////////////////////////////////////////


module guass_divider(

input                  i_clk,
input                  i_rst,

input                  i_field_vld,
input                  i_line_vld,
input [`Y16_DW - 1:0]  i_img_data,

output                 o_field_vld,
output                 o_line_vld,
input [`Y16_DW - 1:0]  o_img_data,

input [2:0]            i_gauss_level


    );
    
//*************************reg and wire*********************
//field_vld_dlys
reg field_vld_dly1;
reg field_vld_dly2;

wire  field_vld_line_1;
wire  field_vld_line_2;

//line_vld_dlys
reg  line_vld_dly1;
reg  line_vld_dly2;

wire  line_vld_line_1;
reg  line_vld_line_1_dly1;
reg  line_vld_line_1_dly2;
reg line_vld_line_1_dly3;

wire  line_vld_line_2;
//reg  line_vld_line_2_dly1;
//reg  line_vld_line_2_dly2;

//img_data_dlys
reg [`Y16_DW - 1:0] img_data_dly1;
reg [`Y16_DW - 1:0] img_data_dly2;

wire [`Y16_DW - 1:0] img_data_line_1;
reg [`Y16_DW - 1:0] img_data_line_1_dly1;
reg [`Y16_DW - 1:0] img_data_line_1_dly2;

wire [`Y16_DW - 1:0] img_data_line_2;
reg [`Y16_DW - 1:0] img_data_line_2_dly1;
reg [`Y16_DW - 1:0] img_data_line_2_dly2;


reg [10:0] gauss_table_corner;  //四角
reg [10:0] gauss_table_edge;    //边缘
reg [10:0] gauss_table_center;  //中心
reg [18:0] gauss_table_sum;     //模板和

always @(posedge i_clk or posedge i_rst) begin
    if (i_rst)begin
        gauss_table_corner <= `DW_GAUSS_TABLE 'd298;
        gauss_table_edge   <= `DW_GAUSS_TABLE 'd552;
        gauss_table_center <= `DW_GAUSS_TABLE 'd1024; 
        gauss_table_sum    <= `DW_GAUSS_SUM_TABLE'd4424;  
    end
    else begin
        case (i_gauss_level)
            3'd0: begin
                gauss_table_corner <= `DW_GAUSS_TABLE 'd298;
                gauss_table_edge   <= `DW_GAUSS_TABLE 'd552;
                gauss_table_center <= `DW_GAUSS_TABLE 'd1024; 
                gauss_table_sum    <= `DW_GAUSS_SUM_TABLE'd4424;
            end 
            //
            3'd1: begin
                gauss_table_corner <= `DW_GAUSS_TABLE 'd377;
                gauss_table_edge   <= `DW_GAUSS_TABLE 'd621;
                gauss_table_center <= `DW_GAUSS_TABLE 'd1024; 
                gauss_table_sum    <= `DW_GAUSS_SUM_TABLE'd5016;
            end
            default: ;
    endcase
    end
end


//field_vld_dly
 data_delay #(
    .DATA_WIDTH(1),
    .DLY_NUM(`IMAGE_WIDTH)
    )
 u_field_vld_dly640 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(i_field_vld),
      .o_data(field_vld_line_1)
    );

 data_delay #(
    .DATA_WIDTH(1),
    .DLY_NUM(`IMAGE_WIDTH)
    )
 u_field_vld_dly1280 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(field_vld_line_1),
      .o_data(field_vld_line_2)
    );


//line_vld_dly
 data_delay #(
    .DATA_WIDTH(1),
    .DLY_NUM(`IMAGE_WIDTH)
    )
 u_line_vld_dly640 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(i_line_vld),
      .o_data(line_vld_line_1)
    );

 data_delay #(
    .DATA_WIDTH(1),
    .DLY_NUM(`IMAGE_WIDTH )
    )
 u_line_vld_dly1280 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(line_vld_line_1),
      .o_data(line_vld_line_2)
    );
    
//img_data_dly
 data_delay #(
    .DATA_WIDTH(`Y16_DW),
    .DLY_NUM(`IMAGE_WIDTH )
    )
 u_img_data_dly640 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(i_img_data),
      .o_data(img_data_line_1)
    );

 data_delay #(
    .DATA_WIDTH(`Y16_DW),
    .DLY_NUM(`IMAGE_WIDTH )
    )
 u_img_data_dly1280 (
      .i_clk(i_clk),
      .i_rst(i_rst),
      .i_data(img_data_line_1),
      .o_data(img_data_line_2)
    );

reg [`Y16_DW - 1:0] line_cnt;
reg [`Y16_DW - 1:0] pix_cnt;
reg [2:0] field_vld_buff;


always @(posedge i_clk or posedge i_rst) begin
    if (i_rst)begin
        pix_cnt <= {`Y16_DW{1'b0}};
        line_cnt <= {`Y16_DW{1'b0}};
        field_vld_buff <= {`Y16_DW{1'b0}};
    end
    else begin
        field_vld_buff <= {field_vld_buff[2:0],i_field_vld};
        if(field_vld_buff[1])begin //帧起始后清零状态
            pix_cnt <= {`Y16_DW{1'b0}};
            line_cnt <= {`Y16_DW{1'b0}};
        end
        else begin
            if(line_vld_line_1_dly1)begin //从起始后第二行第二个像素开始处理
                if(pix_cnt < `IMAGE_WIDTH  - 1)begin
                    pix_cnt <= pix_cnt + 1'b1;
                end
                else begin
                    pix_cnt <= 16'd0;
                    line_cnt <= line_cnt + 1;
                end
            end
        end
    end
end


reg [24:0] mult_gauss_img_11;
reg [24:0] mult_gauss_img_12;
reg [24:0] mult_gauss_img_13;

reg [24:0] mult_gauss_img_21;
reg [24:0] mult_gauss_img_22;
reg [24:0] mult_gauss_img_23;

reg [24:0] mult_gauss_img_31;
reg [24:0] mult_gauss_img_32;
reg [24:0] mult_gauss_img_33;

//************************mult_gauss_img = img_3x3 * gauss_3x3 ***************** 1 dly
always @(posedge i_clk or posedge i_rst) begin
    if (i_rst)begin
        
        field_vld_dly1 <= 1'b0;
        field_vld_dly2 <= 1'b0;
        
        line_vld_dly1 <= 1'b0;
        line_vld_dly2 <= 1'b0;
        
        line_vld_line_1_dly1 <= 1'b0;
        line_vld_line_1_dly2 <= 1'b0;
        
        img_data_dly1 <= {`Y16_DW{1'b0}};
        img_data_dly2 <= {`Y16_DW{1'b0}};

        img_data_line_1_dly1 <= {`Y16_DW{1'b0}}; 
        img_data_line_1_dly2 <= {`Y16_DW{1'b0}}; 
        line_vld_line_1_dly3 <= {`Y16_DW{1'b0}}; 

        img_data_line_2_dly1 <=  {`Y16_DW{1'b0}}; 
        img_data_line_2_dly2 <=  {`Y16_DW{1'b0}}; 
    end
    else begin        
    
        //line dly
        line_vld_line_1_dly1 <= line_vld_line_1;
        line_vld_line_1_dly2 <= line_vld_line_1_dly1;
        line_vld_line_1_dly3 <=  line_vld_line_1_dly2;
        //img_data_dly
        img_data_dly1 <= i_img_data;
        img_data_dly2 <= img_data_dly1;
        
        img_data_line_1_dly1 <= img_data_line_1;
        img_data_line_1_dly2 <= img_data_line_1_dly1;
        
        img_data_line_2_dly1 <= img_data_line_2;
        img_data_line_2_dly2 <= img_data_line_2_dly1;
        
       if(line_cnt == 0)begin
           if(pix_cnt == 0)begin   //滑窗中心在左上角    
                mult_gauss_img_11 <= 25'd0;
                mult_gauss_img_12 <= 25'd0;
                mult_gauss_img_13 <= 25'd0;
                                   
                mult_gauss_img_21 <= 25'd0;
                mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                              
                mult_gauss_img_31 <= i_img_data * gauss_table_corner;
                mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                mult_gauss_img_33 <= 25'd0;
           end
           else if(pix_cnt > 0 & pix_cnt < `IMAGE_WIDTH - 1)begin// 第一行中间
                mult_gauss_img_11 <= 25'd0;
                mult_gauss_img_12 <= 25'd0;
                mult_gauss_img_13 <= 25'd0;
                                   
                mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                              
                mult_gauss_img_31 <= i_img_data * gauss_table_corner;
                mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                mult_gauss_img_33 <= img_data_dly2 * gauss_table_corner;
           end
           else begin //右上角
                mult_gauss_img_11 <= 25'd0;
                mult_gauss_img_12 <= 25'd0;
                mult_gauss_img_13 <= 25'd0;
                                   
                mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                mult_gauss_img_23 <= 25'd0;
                              
                mult_gauss_img_31 <= i_img_data * gauss_table_corner;
                mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                mult_gauss_img_33 <= 25'd0;
           end
       end
       else if(line_cnt > 0 & line_cnt < `IMAGE_HEIGHT - 1 )begin //左侧一列
                if(pix_cnt == 0)begin
                   mult_gauss_img_11 <= 25'd0;
                   mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                   mult_gauss_img_13 <= img_data_line_2_dly2 * gauss_table_corner;
                                      
                   mult_gauss_img_21 <= 25'd0;
                   mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                   mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                                 
                   mult_gauss_img_31 <= 25'd0;
                   mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                   mult_gauss_img_33 <= img_data_dly2 * gauss_table_corner; 
                end
                else if(pix_cnt > 0 & pix_cnt < `IMAGE_WIDTH - 1)begin // 中心区域  
                   mult_gauss_img_11 <= img_data_line_2 * gauss_table_corner;
                   mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                   mult_gauss_img_13 <= img_data_line_2_dly2 * gauss_table_corner;
                                      
                   mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                   mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                   mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                                 
                   mult_gauss_img_31 <= i_img_data * gauss_table_corner;
                   mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                   mult_gauss_img_33 <= img_data_dly2 * gauss_table_corner;
                end
                else begin // 右侧一列   
                   mult_gauss_img_11 <= img_data_line_2 * gauss_table_corner;
                   mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                   mult_gauss_img_13 <= 25'd0;
                                      
                   mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                   mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                   mult_gauss_img_23 <= 25'd0;
                                 
                   mult_gauss_img_31 <= i_img_data * gauss_table_corner;
                   mult_gauss_img_32 <= img_data_dly1 * gauss_table_edge;
                   mult_gauss_img_33 <= 25'd0;
                end
            end
            else begin
                 if(pix_cnt == 0)begin //左下角
                      mult_gauss_img_11 <= 25'd0;
                      mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                      mult_gauss_img_13 <= img_data_line_2_dly2 * gauss_table_corner;
                                         
                      mult_gauss_img_21 <= 25'd0;
                      mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                      mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                                    
                      mult_gauss_img_31 <= 25'd0;
                      mult_gauss_img_32 <= 25'd0;
                      mult_gauss_img_33 <= 25'd0;
                end
                else if(pix_cnt > 0 & pix_cnt < `IMAGE_WIDTH - 1)begin //最后一行中间 
                      mult_gauss_img_11 <= img_data_line_2 * gauss_table_corner;
                      mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                      mult_gauss_img_13 <= img_data_line_2_dly2 * gauss_table_corner;
                                         
                      mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                      mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                      mult_gauss_img_23 <= img_data_line_1 * gauss_table_edge;
                                    
                      mult_gauss_img_31 <= 25'd0;
                      mult_gauss_img_32 <= 25'd0;
                      mult_gauss_img_33 <= 25'd0;
                end
                else begin // 右下角
                      mult_gauss_img_11 <= img_data_line_2 * gauss_table_corner;
                      mult_gauss_img_12 <= img_data_line_2_dly1 * gauss_table_edge;
                      mult_gauss_img_13 <= 25'd0;
                                         
                      mult_gauss_img_21 <= img_data_line_1_dly2 * gauss_table_edge;
                      mult_gauss_img_22 <= img_data_line_1_dly1 * gauss_table_center;
                      mult_gauss_img_23 <= 25'd0;
                                    
                      mult_gauss_img_31 <= 25'd0;
                      mult_gauss_img_32 <= 25'd0;
                      mult_gauss_img_33 <= 25'd0;
                end
           end
    end
end

//gauss_table_sum (no dly)
wire [32:0] sum_mult_data;
assign sum_mult_data = mult_gauss_img_11 + mult_gauss_img_12 + mult_gauss_img_13
                            + mult_gauss_img_21 + mult_gauss_img_22 + mult_gauss_img_23
                            + mult_gauss_img_31 + mult_gauss_img_32 + mult_gauss_img_33;

//normalization -- 1dly
wire [32:0] norm_data;
wire [56:0] m_axis_dout_tdata;
wire [13:0] img_data_limit;
gauss_norm_div inst_gauss_norm_div (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(gauss_table_sum),      // input wire [23 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(sum_mult_data),    // input wire [39 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(m_axis_dout_tdata)            // output wire [63 : 0] m_axis_dout_tdata
);
assign norm_data = m_axis_dout_tdata[56:24];

//data limit
assign img_data_limit = (&norm_data[32:14])?14'd16383:norm_data[13:0];
assign o_img_data = (line_vld_line_1_dly3)?img_data_limit:14'd0;
 
assign o_line_vld = line_vld_line_1_dly3;




endmodule
