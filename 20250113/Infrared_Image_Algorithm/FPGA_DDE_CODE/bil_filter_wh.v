`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2022/10/04 05:24:01
// Design Name: 
// Module Name: tops
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


module guid_filter_wh(
input   wire         i_clk,
output wire          o_clk,
input   wire [13:0]  i_I0,            //14位
input   wire      ctrl,
input   wire [10:0]  PxlI_YS,
input   wire [10:0]  PxlI_XS,
input   wire [10:0]  PxlI_CntY,
input   wire [10:0]  PxlI_CntX,
output wire [10:0]  Pxlo_YS,
output wire [10:0]  Pxlo_XS,
output wire [10:0]  Pxlo_CntY,
output wire [10:0]  Pxlo_CntX,
//output reg [37:0]                  ww_Ifilter,
//input wire line_vld,
//output wire line_vld_o,  
output  wire [13:0] o_Ifilter
);
parameter e = 28'd15;


assign o_clk = i_clk;
assign Pxlo_YS=ctrl?PxlI_YS+1:PxlI_YS;
assign Pxlo_XS=ctrl?PxlI_XS+11:PxlI_XS;
assign Pxlo_CntY=PxlI_CntY;
assign Pxlo_CntX=PxlI_CntX;


 
 //总的延时计算

   
//wire line_vld_o;
//   data_delay 
//    #(
//      .DATA_WIDTH(1),
//      .DLY_NUM(691)
//    ) 
//    inst_data_delay_u1 (
//      .i_clk(i_clk),
//      //.i_rst(i_rst),
//      .i_data(line_vld),
//      .o_data(line_vld_o)
//    );  
//

/* parameter g11 = 8'd89;  
parameter g12 = 8'd249;  
parameter g13 = 8'd89;  
 
parameter g21 = 8'd249;  
parameter g22 = 10'd691;  
parameter g23 = 8'd249;  

parameter g31 = 8'd89;  
parameter g32 = 8'd249;  
parameter g33 = 8'd89;   */ 


 wire        PxlI_Clk;
   /*  wire [9:0]  PxlI_YS;
    wire [9:0]  PxlI_XS;
    wire [9:0]  PxlI_CntY;
    wire [9:0]  PxlI_CntX; */
    wire [13:0] PxlI_Dat;

    wire [13:0] line11_om;
    wire [13:0] line12_om;
    wire [13:0] line13_om;
    
    wire [13:0] line21_om;
    wire [13:0] line22_om;
    wire [13:0] line23_om;
   
    wire [13:0] line31_om;
    wire [13:0] line32_om;
    wire [13:0] line33_om;
    

    // 渚嬪寲 window_3x3 妯″潡,寤舵椂涓?琛屼袱鍒?
    window_3x3#(
	    .C_IMG_COL_WIDTH(640),
		.C_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(14),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_3x3 (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS),
      .PxlI_XS(PxlI_XS),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(i_I0),

      .line11_om(line11_om),
      .line12_om(line12_om),
      .line13_om(line13_om),

      .line21_om(line21_om),
      .line22_om(line22_om),
      .line23_om(line23_om),

      .line31_om(line31_om),
      .line32_om(line32_om),
      .line33_om(line33_om)
    );
//第一个时钟
wire [15:0]sumline_1;
wire [15:0]sumline_2;
wire [15:0]sumline_3;
always @(posedge i_clk )
begin
    sumline_1 <= line11_om+line12_om+line13_om;
    sumline_2 <= line21_om+line22_om+line23_om;
    sumline_3 <= line31_om+line32_om+line33_om; 
end 	

wire [27:0]	square_pp11;
wire [27:0]	square_pp12;
wire [27:0]	square_pp13;
wire [27:0]	square_pp21;
wire [27:0]	square_pp22;
wire [27:0]	square_pp23;
wire [27:0]	square_pp31;
wire [27:0]	square_pp32;
wire [27:0]	square_pp33;



mult_bil_1 pp11 (
  .CLK(i_clk),  // input wire CLK
  .A(line11_om),      // input wire [13 : 0] A
  .B(line11_om),      // input wire [13 : 0] B
  .P(square_pp11)      // output wire [27 : 0] P
);	
mult_bil_1 pp12 (
  .CLK(i_clk),  // input wire CLK
  .A(line12_om),      // input wire [13 : 0] A
  .B(line12_om),      // input wire [13 : 0] B
  .P(square_pp12)      // output wire [27 : 0] P
);		
mult_bil_1 pp13 (
  .CLK(i_clk),  // input wire CLK
  .A(line13_om),      // input wire [13 : 0] A
  .B(line13_om),      // input wire [13 : 0] B
  .P(square_pp13)      // output wire [27 : 0] P
);	
	
mult_bil_1 pp21 (
  .CLK(i_clk),  // input wire CLK
  .A(line21_om),      // input wire [13 : 0] A
  .B(line21_om),      // input wire [13 : 0] B
  .P(square_pp21)      // output wire [27 : 0] P
);	
mult_bil_1 pp22 (
  .CLK(i_clk),  // input wire CLK
  .A(line22_om),      // input wire [13 : 0] A
  .B(line22_om),      // input wire [13 : 0] B
  .P(square_pp22)      // output wire [27 : 0] P
);		
mult_bil_1 pp23 (
  .CLK(i_clk),  // input wire CLK
  .A(line23_om),      // input wire [13 : 0] A
  .B(line23_om),      // input wire [13 : 0] B
  .P(square_pp23)      // output wire [27 : 0] P
);	
mult_bil_1 pp31 (
  .CLK(i_clk),  // input wire CLK
  .A(line31_om),      // input wire [13 : 0] A
  .B(line31_om),      // input wire [13 : 0] B
  .P(square_pp31)      // output wire [27 : 0] P
);	
mult_bil_1 pp32 (
  .CLK(i_clk),  // input wire CLK
  .A(line32_om),      // input wire [13 : 0] A
  .B(line32_om),      // input wire [13 : 0] B
  .P(square_pp32)      // output wire [27 : 0] P
);	
mult_bil_1 pp33 (
  .CLK(i_clk),  // input wire CLK
  .A(line33_om),      // input wire [13 : 0] A
  .B(line33_om),      // input wire [13 : 0] B
  .P(square_pp33)      // output wire [27 : 0] P
);	
//第2个时钟
reg [17:0]sumline;

always @(posedge i_clk )
begin
    sumline <= sumline_1+sumline_2+sumline_3;
end 	
reg [29:0]sumline_square_pp1;
reg [29:0]sumline_square_pp2;
reg [29:0]sumline_square_pp3;



always @(posedge i_clk )
begin
    sumline_square_pp1 <= square_pp11+square_pp12+square_pp13;
    sumline_square_pp2 <= square_pp21+square_pp22+square_pp23;
    sumline_square_pp3 <= square_pp31+square_pp31+square_pp33; 
end 
//第3个时钟	，求pm
	reg [13:0] ave_p;
	
	always @(posedge i_clk )
begin
    ave_p <= (sumline*113)>>10;//代替和除以9
end 
	
reg [31:0]sumline_square;

always @(posedge i_clk )
begin
    sumline_square <= sumline_square_pp1+sumline_square_pp2+sumline_square_pp3;
end 
	
//第4个时钟 pm*pm,P的平方求和再除9求平均值得到ppm
wire [27:0] ave_pp;

mult_bil_2 square_avepp (
  .CLK(CLK),  // input wire CLK
  .A(ave_p),      // input wire [13 : 0] A
  .B(ave_p),      // input wire [13 : 0] B
  .P(ave_pp)      // output wire [27 : 0] P
);
	reg [27:0] ave_pppp;
	
	always @(posedge i_clk )
begin
    ave_pppp <= (sumline_square*113)>>10;//代替和除以9
end 
//第5-6个时钟,a=(ppm-pmpm)*1024/(ppm-pmpm+e)
reg [37:0]a_dividend;
reg [11:0]a_divisior;
always @(posedge i_clk )
begin
    a_dividend <= (ave_pppp-ave_pp)*1024;//(ppm-pmpm)*1024
	a_divisior <= (ave_pppp-ave_pp+e);//ppm-pmpm+e
end 
wire [55 : 0] m_axis_dout_tdata;
wire [9:0] a;
div_guid_1 your_instance_name (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(a_divisior),      // input wire [15 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(a_dividend),    // input wire [39 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(1'b1),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(m_axis_dout_tdata)            // output wire [55 : 0] m_axis_dout_tdata
);
assign a=m_axis_dout_tdata[25:16];


//第7个时钟b=(1024-a)*pm,a是第6个时钟，pm是第3个时钟
reg [23:0] i_b;
reg  [13:0]ave_p_dly1;
reg  [13:0]ave_p_dly2;
reg  [13:0]ave_p_dly3;
//pm是第3个时钟，a是第6个时钟
	always @(posedge i_clk )
begin
    ave_p_dly1 <= ave_p;//
	 ave_p_dly2 <= ave_p_dly1;//
	  ave_p_dly3 <= ave_p_dly2;//
end 



mult_guid_2 b_1 (
  .CLK(i_clk),  // input wire CLK
  .A(1024-a),      // input wire [9 : 0] A
  .B(ave_p_dly3),      // input wire [13 : 0] B
  .P(i_b)      // output wire [23 : 0] P
);
//把第六个时钟的a变成与b一样的第7个时钟
reg [9:0] i_a;
always @(posedge i_clk )
begin
    i_a <= a;//
	 
end 
//再把a和b缓存3行3列，从而求a和b的均值,目前延迟了一行两列，再加7个时钟

wire [9:0]line11_a;
wire [9:0]line12_a;
wire [9:0]line13_a;
wire [9:0]line21_a;
wire [9:0]line22_a;
wire [9:0]line23_a;
wire [9:0]line31_a;
wire [9:0]line32_a;
wire [9:0]line33_a;


wire [23:0]line11_b;
wire [23:0]line12_b;
wire [23:0]line13_b;
wire [23:0]line21_b;
wire [23:0]line22_b;
wire [23:0]line23_b;
wire [23:0]line31_b;
wire [23:0]line32_b;
wire [23:0]line33_b;

 window_3x3#(
	    .C_IMG_COL_WIDTH(640),
		.C_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(10),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_3x3_a (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS+9),
      .PxlI_XS(PxlI_XS+1),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(i_a),

      .line11_om(line11_a),
      .line12_om(line12_a),
      .line13_om(line13_a),

      .line21_om(line21_a),
      .line22_om(line22_a),
      .line23_om(line23_a),

      .line31_om(line31_a),
      .line32_om(line32_a),
      .line33_om(line33_a)
    );
	//b缓存
 window_3x3#(
	    .C_IMG_COL_WIDTH(640),
		.C_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(24),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_3x3_a (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS+9),
      .PxlI_XS(PxlI_XS+1),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(i_b),

      .line11_om(line11_b),
      .line12_om(line12_b),
      .line13_om(line13_b),

      .line21_om(line21_b),
      .line22_om(line22_b),
      .line23_om(line23_b),

      .line31_om(line31_b),
      .line32_om(line32_b),
      .line33_om(line33_b)
    );

//1行9列后+第1个时钟
wire [11:0]sumline_a1;
wire [11:0]sumline_a2;
wire [11:0]sumline_a3;
always @(posedge i_clk )
begin
    sumline_a1 <= line11_a+line12_a+line13_a;
    sumline_a2 <= line21_a+line22_a+line23_a;
    sumline_a3 <= line31_a+line32_a+line33_a; 
end 

wire [25:0]sumline_b1;
wire [25:0]sumline_b2;
wire [25:0]sumline_b3;
always @(posedge i_clk )
begin
    sumline_b1 <= line11_b+line12_b+line13_b;
    sumline_b2 <= line21_b+line22_b+line23_b;
    sumline_b3 <= line31_b+line32_b+line33_b; 
end 

//1行9列后+第2个时钟

reg [13:0]sumline_a;

always @(posedge i_clk )
begin
    sumline_a <= sumline_a1+sumline_a2+sumline_a3;
end

reg [27:0]sumline_b;

always @(posedge i_clk )
begin
    sumline_b <= sumline_b1+sumline_b2+sumline_b3;
end
//1行9列后+第3个时钟

	reg [9:0] ave_a;
	
	always @(posedge i_clk )
begin
    ave_a <= (sumline_a*113)>>10;//代替和除以9
end 

reg [23:0] ave_b;
reg [23:0] ave_b_dly;	
	always @(posedge i_clk )
begin
    ave_b <= (sumline_b*113)>>10;//代替和除以9
	ave_b_dly<=ave_b;
end 

//1行9列后+第4-5个时钟,q=aI+b

//先对输入像素延迟1行9列后+第3个时钟


wire[13:0] I_dly;
wire[23:0] q_1;
data_delay 
    #(
      .DATA_WIDTH(14),
      .DLY_NUM(652)
    ) 
    inst_data_delay_u2 (
      .i_clk(i_clk),

      .i_data(i_I0),
      .o_data(I_dly)
    );

//第4个
mult_guid_3 u_q (
  .CLK(i_clk),  // input wire CLK
  .A(ave_a),      // input wire [9 : 0] A
  .B(I_dly),      // input wire [13 : 0] B
  .P(q_1)      // output wire [23 : 0] P
);

//第5个

reg [23:0]o_q;
	
	always @(posedge i_clk )
begin
    o_q<=q_1+ave_b_dly;
end 
assign o_Ifilter=ctrl?o_q[23:10]:i_I0;//输出除以1024









    
    
endmodule

