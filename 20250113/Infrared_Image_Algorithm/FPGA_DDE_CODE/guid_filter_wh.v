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
input   wire [7:0]   e,
output wire [10:0]  Pxlo_YS,
output wire [10:0]  Pxlo_XS,
output wire [10:0]  Pxlo_CntY,
output wire [10:0]  Pxlo_CntX,
//
/*  input wire line_vld,
output wire line_vld_o,   */ 
output  wire [13:0] o_Ifilter
);
//parameter e = 28'd1;//最大不超过12位


assign o_clk = i_clk;
assign Pxlo_YS=ctrl?PxlI_YS+4:PxlI_YS;
assign Pxlo_XS=ctrl?PxlI_XS+16:PxlI_XS;
assign Pxlo_CntY=PxlI_CntY;
assign Pxlo_CntX=PxlI_CntX;


 
 //总的延时计算4行16列,一行660

  
   /* wire line_vld_o;
    data_delay 
     #(
       .DATA_WIDTH(1),
       .DLY_NUM(2656)
     ) 
     inst_data_delay_u1 (
       .i_clk(i_clk),
       //.i_rst(i_rst),
       .i_data(line_vld),
       .o_data(line_vld_o)
   );   
  */

/* parameter g11 = 8'd89;  
parameter g12 = 8'd249;  
parameter g13 = 8'd89;  
 
parameter g21 = 8'd249;  
parameter g22 = 10'd691;  
parameter g23 = 8'd249;  

parameter g31 = 8'd89;  
parameter g32 = 8'd249;  
parameter g33 = 8'd89;   */ 


 //wire        PxlI_Clk;
   /*  wire [9:0]  PxlI_YS;
    wire [9:0]  PxlI_XS;
    wire [9:0]  PxlI_CntY;
    wire [9:0]  PxlI_CntX; */
    //wire [13:0] PxlI_Dat;

    wire [13:0] line11_om;
    wire [13:0] line12_om;
    wire [13:0] line13_om;
	wire [13:0] line14_om;
    wire [13:0] line15_om;
	
    wire [13:0] line21_om;
    wire [13:0] line22_om;
    wire [13:0] line23_om;
	wire [13:0] line24_om;
    wire [13:0] line25_om;
   
    wire [13:0] line31_om;
    wire [13:0] line32_om;
    wire [13:0] line33_om;
    wire [13:0] line34_om;
    wire [13:0] line35_om;
	
	wire [13:0] line41_om;
    wire [13:0] line42_om;
    wire [13:0] line43_om;
    wire [13:0] line44_om;
    wire [13:0] line45_om;
	
	wire [13:0] line51_om;
    wire [13:0] line52_om;
    wire [13:0] line53_om;
    wire [13:0] line54_om;
    wire [13:0] line55_om;
	
	
	

    // 两行两列
    window_5x5#(
	    .G_IMG_COL_WIDTH(640),
		.G_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(14),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_5x5 (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS),
      .PxlI_XS(PxlI_XS),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(i_I0),

      .line11_om(line11_om),
      .line12_om(line12_om),
      .line13_om(line13_om),
      .line14_om(line14_om),
      .line15_om(line15_om),
	  
      .line21_om(line21_om),
      .line22_om(line22_om),
      .line23_om(line23_om),
      .line24_om(line24_om),
      .line25_om(line25_om),

      .line31_om(line31_om),
      .line32_om(line32_om),
      .line33_om(line33_om),
      .line34_om(line34_om),
      .line35_om(line35_om),
	  
      .line41_om(line41_om),
      .line42_om(line42_om),
      .line43_om(line43_om),
      .line44_om(line44_om),
      .line45_om(line45_om),	

      .line51_om(line51_om),
      .line52_om(line52_om),
      .line53_om(line53_om),
      .line54_om(line54_om),
      .line55_om(line55_om)	  
    );
//第一个时钟，求A0=P00+P10+P20;P00的平方
reg [16:0]sumline_1;
reg [16:0]sumline_2;
reg [16:0]sumline_3;
reg [16:0]sumline_4;
reg [16:0]sumline_5;

always @(posedge i_clk )
begin
    sumline_1 <={3'b0,line11_om}+{3'b0,line12_om}+{3'b0,line13_om}+{3'b0,line14_om}+{3'b0,line15_om};
    sumline_2 <={3'b0,line21_om}+{3'b0,line22_om}+{3'b0,line23_om}+{3'b0,line24_om}+{3'b0,line25_om};
    sumline_3 <={3'b0,line31_om}+{3'b0,line32_om}+{3'b0,line33_om}+{3'b0,line34_om}+{3'b0,line35_om}; 
	sumline_4 <={3'b0,line41_om}+{3'b0,line42_om}+{3'b0,line43_om}+{3'b0,line44_om}+{3'b0,line45_om};
    sumline_5 <={3'b0,line51_om}+{3'b0,line52_om}+{3'b0,line53_om}+{3'b0,line54_om}+{3'b0,line55_om}; 
	
end 	

wire [27:0]	square_pp11;
wire [27:0]	square_pp12;
wire [27:0]	square_pp13;
wire [27:0]	square_pp14;
wire [27:0]	square_pp15;

wire [27:0]	square_pp21;
wire [27:0]	square_pp22;
wire [27:0]	square_pp23;
wire [27:0]	square_pp24;
wire [27:0]	square_pp25;

wire [27:0]	square_pp31;
wire [27:0]	square_pp32;
wire [27:0]	square_pp33;
wire [27:0]	square_pp34;
wire [27:0]	square_pp35;

wire [27:0]	square_pp41;
wire [27:0]	square_pp42;
wire [27:0]	square_pp43;
wire [27:0]	square_pp44;
wire [27:0]	square_pp45;

wire [27:0]	square_pp51;
wire [27:0]	square_pp52;
wire [27:0]	square_pp53;
wire [27:0]	square_pp54;
wire [27:0]	square_pp55;


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
	
mult_bil_1 pp14 (
  .CLK(i_clk),  // input wire CLK
  .A(line14_om),      // input wire [13 : 0] A
  .B(line14_om),      // input wire [13 : 0] B
  .P(square_pp14)      // output wire [27 : 0] P
);		
mult_bil_1 pp15 (
  .CLK(i_clk),  // input wire CLK
  .A(line15_om),      // input wire [13 : 0] A
  .B(line15_om),      // input wire [13 : 0] B
  .P(square_pp15)      // output wire [27 : 0] P
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

mult_bil_1 pp24 (
  .CLK(i_clk),  // input wire CLK
  .A(line24_om),      // input wire [13 : 0] A
  .B(line24_om),      // input wire [13 : 0] B
  .P(square_pp24)      // output wire [27 : 0] P
);		
mult_bil_1 pp25 (
  .CLK(i_clk),  // input wire CLK
  .A(line25_om),      // input wire [13 : 0] A
  .B(line25_om),      // input wire [13 : 0] B
  .P(square_pp25)      // output wire [27 : 0] P
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
mult_bil_1 pp34 (
  .CLK(i_clk),  // input wire CLK
  .A(line34_om),      // input wire [13 : 0] A
  .B(line34_om),      // input wire [13 : 0] B
  .P(square_pp34)      // output wire [27 : 0] P
);	
mult_bil_1 pp35 (
  .CLK(i_clk),  // input wire CLK
  .A(line35_om),      // input wire [13 : 0] A
  .B(line35_om),      // input wire [13 : 0] B
  .P(square_pp35)      // output wire [27 : 0] P
);	

mult_bil_1 pp41 (
  .CLK(i_clk),  // input wire CLK
  .A(line41_om),      // input wire [13 : 0] A
  .B(line41_om),      // input wire [13 : 0] B
  .P(square_pp41)      // output wire [27 : 0] P
);	
mult_bil_1 pp42 (
  .CLK(i_clk),  // input wire CLK
  .A(line42_om),      // input wire [13 : 0] A
  .B(line42_om),      // input wire [13 : 0] B
  .P(square_pp42)      // output wire [27 : 0] P
);	
mult_bil_1 pp43 (
  .CLK(i_clk),  // input wire CLK
  .A(line43_om),      // input wire [13 : 0] A
  .B(line43_om),      // input wire [13 : 0] B
  .P(square_pp43)      // output wire [27 : 0] P
);	
mult_bil_1 pp44 (
  .CLK(i_clk),  // input wire CLK
  .A(line44_om),      // input wire [13 : 0] A
  .B(line44_om),      // input wire [13 : 0] B
  .P(square_pp44)      // output wire [27 : 0] P
);	
mult_bil_1 pp45 (
  .CLK(i_clk),  // input wire CLK
  .A(line45_om),      // input wire [13 : 0] A
  .B(line45_om),      // input wire [13 : 0] B
  .P(square_pp45)      // output wire [27 : 0] P
);

mult_bil_1 pp51 (
  .CLK(i_clk),  // input wire CLK
  .A(line51_om),      // input wire [13 : 0] A
  .B(line51_om),      // input wire [13 : 0] B
  .P(square_pp51)      // output wire [27 : 0] P
);	
mult_bil_1 pp52 (
  .CLK(i_clk),  // input wire CLK
  .A(line52_om),      // input wire [13 : 0] A
  .B(line52_om),      // input wire [13 : 0] B
  .P(square_pp52)      // output wire [27 : 0] P
);	
mult_bil_1 pp53 (
  .CLK(i_clk),  // input wire CLK
  .A(line53_om),      // input wire [13 : 0] A
  .B(line53_om),      // input wire [13 : 0] B
  .P(square_pp53)      // output wire [27 : 0] P
);	
mult_bil_1 pp54 (
  .CLK(i_clk),  // input wire CLK
  .A(line54_om),      // input wire [13 : 0] A
  .B(line54_om),      // input wire [13 : 0] B
  .P(square_pp54)      // output wire [27 : 0] P
);	
mult_bil_1 pp55 (
  .CLK(i_clk),  // input wire CLK
  .A(line55_om),      // input wire [13 : 0] A
  .B(line55_om),      // input wire [13 : 0] B
  .P(square_pp55)      // output wire [27 : 0] P
);



//第2个时钟 sum=A0+A1+A2+A3+A4+A5;B0=P00的平方+P01的平方+。。。。

reg [19:0]sumline;

always @(posedge i_clk )
begin
    sumline <= {3'b0,sumline_1}+{3'b0,sumline_2}+{3'b0,sumline_3}+{3'b0,sumline_4}+{3'b0,sumline_5};
end 	
reg [30:0]sumline_square_pp1;
reg [30:0]sumline_square_pp2;
reg [30:0]sumline_square_pp3;
reg [30:0]sumline_square_pp4;
reg [30:0]sumline_square_pp5;


always @(posedge i_clk )
begin
    sumline_square_pp1 <= {3'b0,square_pp11}+{3'b0,square_pp12}+{3'b0,square_pp13}+{3'b0,square_pp14}+{3'b0,square_pp15};
    sumline_square_pp2 <= {3'b0,square_pp21}+{3'b0,square_pp22}+{3'b0,square_pp23}+{3'b0,square_pp24}+{3'b0,square_pp25};
    sumline_square_pp3 <= {3'b0,square_pp31}+{3'b0,square_pp32}+{3'b0,square_pp33}+{3'b0,square_pp34}+{3'b0,square_pp35}; 
	sumline_square_pp4 <= {3'b0,square_pp41}+{3'b0,square_pp42}+{3'b0,square_pp43}+{3'b0,square_pp44}+{3'b0,square_pp45};
    sumline_square_pp5 <= {3'b0,square_pp51}+{3'b0,square_pp52}+{3'b0,square_pp53}+{3'b0,square_pp54}+{3'b0,square_pp55}; 
	
	
end 
//第3个时钟	，求pm=sum1/25,sum2=B0+B1+B2+B3+B4+B5;
	/*reg [25:0] ave_p_1;
	
	wire [13:0] ave_p;
	
	
	always @(posedge i_clk )
begin
    ave_p_1 <= {1'b0,sumline[19:0],5'b0}+{3'b0,sumline[19:0],3'b0}+{6'b0,sumline[19:0]};//代替和除以25，32+8+1。*41>>10
	
end 
assign	ave_p=ave_p_1[23:10];//除1024
*/
wire  [31:0] ave_p_1;
wire [13:0] ave_p;
div_p div_p_u (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata('d25),      // input wire [7 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata({sumline[19:0],4'b0}),    // input wire [23 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(ave_p_1)            // output wire [31 : 0] m_axis_dout_tdata
);
assign	ave_p=ave_p_1[27:8]>>4;//除16


reg [33:0]sumline_square;

always @(posedge i_clk )
begin
    sumline_square <= {3'b0,sumline_square_pp1}+{3'b0,sumline_square_pp2}+{3'b0,sumline_square_pp3}+{3'b0,sumline_square_pp4}+{3'b0,sumline_square_pp5};
end 
	
//第4个时钟 pm*pm,P的平方求和再除25求平均值得到ppm
/*
wire [27:0] ave_pp;

mult_bil_2 square_avepp (
  .CLK(i_clk),  // input wire CLK
  .A(ave_p),      // input wire [13 : 0] A
  .B(ave_p),      // input wire [13 : 0] B
  .P(ave_pp)      // output wire [27 : 0] P
);
	wire [27:0] ave_pppp;
	reg [39:0] ave_pppp1;
	always @(posedge i_clk )
begin
	ave_pppp1 <= {1'b0,sumline_square[33:0],5'b0}+{3'b0,sumline_square[33:0],3'b0}+{6'b0,sumline_square[33:0]};//代替和除以25，32+8+1。41>>10
end 
assign ave_pppp=ave_pppp1[37:10];
*/
wire [27:0] ave_pp;

mult_bil_2 square_avepp (
  .CLK(i_clk),  // input wire CLK
  .A(ave_p),      // input wire [13 : 0] A
  .B(ave_p),      // input wire [13 : 0] B
  .P(ave_pp)      // output wire [27 : 0] P
);
wire [47:0] ave_pppp1;
wire [27:0] ave_pppp;
div_pp div_pp_u (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata('d25),      // input wire [7 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata({sumline_square[33:0],4'b0}),    // input wire [39 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(ave_pppp1)            // output wire [47 : 0] m_axis_dout_tdata
);
assign ave_pppp=ave_pppp1[45:8]>>4;	
//第5-6个时钟,a=(ppm-pmpm)*1024/(ppm-pmpm+e)
/* reg [37:0]a_dividend;
reg [27:0]a_divisior;
reg signed [28:0] minus_value;
wire [27:0] minus_value_abs;
wire [37:0] mult_value;
always @(posedge i_clk )
begin
//   if(ave_pppp>=ave_pp)begin
   
    minus_value <= $signed({1'b0,ave_pppp}) - $signed({1'b0,ave_pp});//(ppm-pmpm)*1024
	a_divisior <= (ave_pppp-ave_pp+{18'b0,e});//ppm-pmpm+e
//	end
//   else begin
//    a_dividend <= (ave_pp-ave_pppp)<<10;//(ppm-pmpm)*1024
//	a_divisior <= (ave_pp-ave_pppp+{18'b0,e});//ppm-pmpm+e
//   end
end 

assign minus_value_abs = (minus_value[14])? ~minus_value + 1 : minus_value;
assign mult_value = minus_value_abs * 1024; */
reg  [27:0] a_dividend_diff;
reg [39:0] a_dividend;
reg [27:0] a_divisior;
always @(posedge i_clk )
begin
if(ave_pppp<=ave_pp)begin
    a_dividend <= {(ave_pp-ave_pppp),12'b0};//(ppm-pmpm)*1024
	a_divisior <= ave_pp-ave_pppp+{18'b0,e};//ppm-pmpm+e
   
	end
  else begin
   a_dividend <= {(ave_pppp-ave_pp),12'b0};//(ppm-pmpm)
	a_divisior <= ave_pppp-ave_pp+{18'b0,e};//ppm-pmpm+e
   end

end 
//assign a_dividend={a_dividend_diff,12'b0};//40
//assign a_divisior=a_dividend_diff+e;//28
wire [71: 0] m_axis_dout_tdata;
wire [11:0] a;

div_guid_1 u_a (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(a_divisior),      // input wire [31 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(a_dividend),    // input wire [39 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(m_axis_dout_tdata)            // output wire [71 : 0] m_axis_dout_tdata//shang[71:32]
);
assign a=m_axis_dout_tdata[43:32];//a最大只有12位，这里截位了。


//第7-8个时钟b=(1024-a)*pm,
wire [25:0] i_b;
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
reg [11:0] a_d;
//assign a_d=4095-a;
always @(posedge i_clk )
begin
    a_d <= 4095-a;//
	 
end 
mult_guid_2 b_1 (
  .CLK(i_clk),  // input wire CLK
  .A(a_d),      // input wire [11 : 0] A
  .B(ave_p_dly3),      // input wire [13 : 0] B
  .P(i_b)      // output wire [25 : 0] P
);
//把第六个时钟的a变成与b一样的第7个时钟
/* reg [11:0] i_a;
always @(posedge i_clk )
begin
    i_a <= a;//
	 
end  */
//再把a和b缓存5*5列，从而求a和b的均值,目前延迟了两行两列，再加7个时钟

wire [11:0]line11_a;
wire [11:0]line12_a;
wire [11:0]line13_a;
wire [11:0]line14_a;
wire [11:0]line15_a;

wire [11:0]line21_a;
wire [11:0]line22_a;
wire [11:0]line23_a;
wire [11:0]line24_a;
wire [11:0]line25_a;

wire [11:0]line31_a;
wire [11:0]line32_a;
wire [11:0]line33_a;
wire [11:0]line34_a;
wire [11:0]line35_a;

wire [11:0]line41_a;
wire [11:0]line42_a;
wire [11:0]line43_a;
wire [11:0]line44_a;
wire [11:0]line45_a;

wire [11:0]line51_a;
wire [11:0]line52_a;
wire [11:0]line53_a;
wire [11:0]line54_a;
wire [11:0]line55_a;


wire [25:0]line11_b;
wire [25:0]line12_b;
wire [25:0]line13_b;
wire [25:0]line14_b;
wire [25:0]line15_b;


wire [25:0]line21_b;
wire [25:0]line22_b;
wire [25:0]line23_b;
wire [25:0]line24_b;
wire [25:0]line25_b;

wire [25:0]line31_b;
wire [25:0]line32_b;
wire [25:0]line33_b;
wire [25:0]line34_b;
wire [25:0]line35_b;

wire [25:0]line41_b;
wire [25:0]line42_b;
wire [25:0]line43_b;
wire [25:0]line44_b;
wire [25:0]line45_b;

wire [25:0]line51_b;
wire [25:0]line52_b;
wire [25:0]line53_b;
wire [25:0]line54_b;
wire [25:0]line55_b;

 window_5x5#(
	    .G_IMG_COL_WIDTH(640),
		.G_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(12),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_5x5_a (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS+2),
      .PxlI_XS(PxlI_XS+8),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(a),

      .line11_om(line11_a),
      .line12_om(line12_a),
      .line13_om(line13_a),
      .line14_om(line14_a),
      .line15_om(line15_a),
	  
      .line21_om(line21_a),
      .line22_om(line22_a),
      .line23_om(line23_a),
      .line24_om(line24_a),
      .line25_om(line25_a),
	  
      .line31_om(line31_a),
      .line32_om(line32_a),
      .line33_om(line33_a),
      .line34_om(line34_a),
      .line35_om(line35_a),	 

      .line41_om(line41_a),
      .line42_om(line42_a),
      .line43_om(line43_a),
      .line44_om(line44_a),
      .line45_om(line45_a),	 

      .line51_om(line51_a),
      .line52_om(line52_a),
      .line53_om(line53_a),
      .line54_om(line54_a),
      .line55_om(line55_a)
  
	  
    );
	//b缓存
 window_5x5#(
	    .G_IMG_COL_WIDTH(640),
		.G_IMG_ROW_WIDTH(512),
		.G_COLSIZE(11),
		.G_ROWSIZE(11),
		.G_DATASIZE(26),
		//.G_M9K_MAX_LEN(256),
		.G_BUFFER_LEN(768)
	)
	inst_window_5x5_b (
      .PxlI_Clk(i_clk),
      .PxlI_YS(PxlI_YS+2),
      .PxlI_XS(PxlI_XS+10),
      .PxlI_CntY(PxlI_CntY),
      .PxlI_CntX(PxlI_CntX),
      .PxlI_Dat(i_b),

      .line11_om(line11_b),
      .line12_om(line12_b),
      .line13_om(line13_b),
      .line14_om(line14_b),
      .line15_om(line15_b),

      .line21_om(line21_b),
      .line22_om(line22_b),
      .line23_om(line23_b),
      .line24_om(line24_b),
      .line25_om(line25_b),
	  
      .line31_om(line31_b),
      .line32_om(line32_b),
      .line33_om(line33_b),
      .line34_om(line34_b),
      .line35_om(line35_b),	 

      .line41_om(line41_b),
      .line42_om(line42_b),
      .line43_om(line43_b),
      .line44_om(line44_b),
      .line45_om(line45_b),	 

      .line51_om(line51_b),
      .line52_om(line52_b),
      .line53_om(line53_b),
      .line54_om(line54_b),
      .line55_om(line55_b)
    );

//2行9列后+2行2列+第1个时钟
reg [14:0]sumline_a1;
reg [14:0]sumline_a2;
reg [14:0]sumline_a3;
reg [14:0]sumline_a4;
reg [14:0]sumline_a5;

always @(posedge i_clk )
begin
    sumline_a1 <= {3'b0,line11_a}+{3'b0,line12_a}+{3'b0,line13_a}+{3'b0,line14_a}+{3'b0,line15_a};
    sumline_a2 <= {3'b0,line21_a}+{3'b0,line22_a}+{3'b0,line23_a}+{3'b0,line24_a}+{3'b0,line25_a};
    sumline_a3 <= {3'b0,line31_a}+{3'b0,line32_a}+{3'b0,line33_a}+{3'b0,line34_a}+{3'b0,line35_a}; 
    sumline_a4 <= {3'b0,line41_a}+{3'b0,line42_a}+{3'b0,line43_a}+{3'b0,line44_a}+{3'b0,line45_a};
    sumline_a5 <= {3'b0,line51_a}+{3'b0,line52_a}+{3'b0,line53_a}+{3'b0,line54_a}+{3'b0,line55_a}; 	
	
end 

reg [28:0]sumline_b1;
reg [28:0]sumline_b2;
reg [28:0]sumline_b3;
reg [28:0]sumline_b4;
reg [28:0]sumline_b5;

always @(posedge i_clk )
begin
    sumline_b1 <= {3'b0,line11_b}+{3'b0,line12_b}+{3'b0,line13_b}+{3'b0,line14_b}+{3'b0,line15_b};
    sumline_b2 <= {3'b0,line21_b}+{3'b0,line22_b}+{3'b0,line23_b}+{3'b0,line24_b}+{3'b0,line25_b};
    sumline_b3 <= {3'b0,line31_b}+{3'b0,line32_b}+{3'b0,line33_b}+{3'b0,line34_b}+{3'b0,line35_b}; 
    sumline_b4 <= {3'b0,line41_b}+{3'b0,line42_b}+{3'b0,line43_b}+{3'b0,line44_b}+{3'b0,line45_b};
    sumline_b5 <= {3'b0,line51_b}+{3'b0,line52_b}+{3'b0,line53_b}+{3'b0,line54_b}+{3'b0,line55_b}; 
end 

//2行9列后+2行2列+第2个时钟

reg [17:0]sumline_a;

always @(posedge i_clk )
begin
    sumline_a <= {3'b0,sumline_a1}+{3'b0,sumline_a2}+{3'b0,sumline_a3}+{3'b0,sumline_a4}+{3'b0,sumline_a5};
end

reg [31:0]sumline_b;

always @(posedge i_clk )
begin
    sumline_b <= {3'b0,sumline_b1}+{3'b0,sumline_b2}+{3'b0,sumline_b3}+{3'b0,sumline_b4}+{3'b0,sumline_b5};
end
//2行9列后+2行2列+第3个时钟
    reg   [23:0] ave_a1;
	wire [11:0] ave_a;
	
	
	
	
	always @(posedge i_clk )
begin
    
	ave_a1 <= {1'b0,sumline_a[17:0],5'b0}+{3'b0,sumline_a[17:0],3'b0}+{6'b0,sumline_a[17:0]};//代替和除以25，32+8+1。41>>10
end 
assign ave_a=ave_a1[21:10];





                                                                                             
reg [37:0] ave_b;
wire [25:0] ave_b_dly;	
	always @(posedge i_clk )
begin
 
	ave_b <= {1'b0,sumline_b[31:0],5'b0}+{3'b0,sumline_b[31:0],3'b0}+{6'b0,sumline_b[31:0]};//代替和除以25，32+8+1。41>>10
	//ave_b_dly<=ave_b[35:10];
end 
assign ave_b_dly=ave_b[35:10];
//2行9列后+2行2列+第4-5个时钟,q=aI+b

//先对输入像素延迟2行9列后++2行2列+第3个时钟，4行14，660*4+14=2654,680*4+14=2734


wire[13:0] I_dly;
wire[25:0] q_1;
reg[25:0] q_dly;
data_delay 
    #(
      .DATA_WIDTH(14),
      .DLY_NUM(2653)
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

reg [25:0]o_q;
	
	always @(posedge i_clk )
begin
    q_dly<=q_1;
	o_q<=q_dly+ave_b_dly;
end 
assign o_Ifilter=ctrl?o_q[25:12]:i_I0;//输出除以4096


    
    
endmodule

