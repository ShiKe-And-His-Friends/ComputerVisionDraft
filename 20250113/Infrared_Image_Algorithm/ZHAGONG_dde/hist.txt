`timescale 1ns / 1ps
module hist 
#(
	parameter 			MAP_Max = 8'd165 ,
	parameter 			MAP_Mid = 8'd90  ,
	parameter 			MAP_Min = 8'd89  )
(
input clkin,
input clkin_4x,
input rst_n,

input HS_in,
input VS_in,
input [13:0]Lfdata_in,
input signed[13:0]Hfdata_in,

output reg hst_hs,
output reg hst_vs,
output reg [7:0]hst_dout
    );

reg Hst_fifo_rd;
wire [13:0]Hst_fifo_dout;
Hist_fifo Hist_fifo (
  .rst(!rst_n), // input rst
  .wr_clk(clkin), // input wr_clk
  .rd_clk(clkin_4x), // input rd_clk
  .din(Lfdata_in), // input [15 : 0] din
  .wr_en(HS_in&&VS_in), // input wr_en
  .rd_en(Hst_fifo_rd), // input rd_en
  .dout(Hst_fifo_dout), // output [15 : 0] dout
  .full(), // output full
  .empty(), // output empty
  .wr_rst_busy(),  // output wire wr_rst_busy
  .rd_rst_busy()  // output wire rd_rst_busy
);

reg HS_in_r,HS_in_rr;
reg VS_in_r,VS_in_rr;
reg[7:0]MAP_Max_r;
reg[7:0]MAP_Mid_r;
reg[7:0]MAP_Min_r; 

always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			MAP_Max_r <= 0;
			MAP_Mid_r <= 0;
			MAP_Min_r <= 0;
//			platform_r <= 0;
		end
	else if(VS_in_r & !VS_in_rr)
		begin
			MAP_Max_r <= MAP_Max;
			MAP_Mid_r <= MAP_Mid;
			MAP_Min_r <= MAP_Min;
//			platform_r <= platform;
		end		
	else
		begin
			MAP_Max_r <= MAP_Max_r;  
			MAP_Mid_r <= MAP_Mid_r;  
			MAP_Min_r <= MAP_Min_r;  
//			platform_r <= platform_r; 
		end
end
always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			HS_in_r <= 0;
			HS_in_rr <= 0;
			VS_in_r <= 0;
			VS_in_rr <= 0;
		end
	else
		begin
			HS_in_r <= HS_in;
			HS_in_rr <= HS_in_r;
			VS_in_r <= VS_in;
			VS_in_rr <= VS_in_r;
		end
end

reg Cnt_run;
reg [1:0]p_cnt;
reg [10:0]h_cnt;
reg [10:0]v_cnt;
always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			Cnt_run <= 0;
			p_cnt <= 0;
			h_cnt <= 0;
			v_cnt <= 0;
		end
	else
		begin
			if(!HS_in_rr&&HS_in_r)
				Cnt_run <= 1;
			if(!VS_in)
				begin
					p_cnt <= 0;
					h_cnt <= 0;
					v_cnt <= 0;
				end
			else
				begin
					if(p_cnt >= 3)
						begin
							p_cnt <= 0;
							if(h_cnt >= 645-1)
								begin
									h_cnt <= 0;
									v_cnt <= v_cnt + 1;
									Cnt_run <= 0;
								end
							else
								h_cnt <= h_cnt + 1;
						end
					else if(Cnt_run)
						p_cnt <= p_cnt + 1;
					else ;
				end
				
		end
end

reg [17:0]b_cnt;
reg [2:0]hst_state;
reg [2:0]hst_state_1;
reg [2:0]hst_state_2;
always @ (posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)	
		begin
			b_cnt <= 0;
			hst_state <= 0;
        end
    else
		begin		
			if(!VS_in)
                begin
                    if(b_cnt == 200000)
                        b_cnt <= 200000;
                    else
                        b_cnt <= b_cnt + 1;
                end
			else
				b_cnt <= 0;
		
            hst_state[0] <= b_cnt>=10     && b_cnt<10+16384;//65546;
            hst_state[1] <= b_cnt>=10+16384+54  && b_cnt<10+16384+54+16384;//131136;
            hst_state[2] <= b_cnt>=10+16384+54+16384+64 && b_cnt<10+16384+54+16384+64+16384;//196736;
            hst_state_1 <= hst_state;
            hst_state_2 <= hst_state_1;
        end
end

reg [31:0] data_sum;
reg [13:0] data_mean;
always @ (posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)	
		begin
			data_sum <= 0;
			data_mean <= 0;
		end
	else
		begin
			if(p_cnt == 2 && h_cnt >= 64 && h_cnt < 64+512)
				data_sum <= data_sum + Lfdata_in;
            else if(v_cnt == 511 & h_cnt == 600)
                data_mean <= data_sum[31:18];
            else if(v_cnt == 511 & h_cnt == 610)
				data_sum <= 0;
			else ;
		end
end

always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			Hst_fifo_rd <= 0;
		end
	else
		begin
			Hst_fifo_rd <= Cnt_run && p_cnt == 2 && h_cnt >= 2 && h_cnt < 642;
		end
end

reg [13:0]Ram_cnt;
reg [13:0]Ram_cnt_1;
reg [13:0]Ram_cnt_2;
reg [13:0]Ram_cnt_3;
reg Hst_ram_rd;
wire [7:0]Hst_ram_dout;
reg [13:0]Hst_ram_rd_addr;
reg Hst_ram_wr;
reg [7:0]Hst_ram_din;
reg [13:0]Hst_ram_wr_addr;
always @ (posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)	
		begin
			Ram_cnt <= 0;
			Hst_ram_wr <= 0;
			Hst_ram_rd <= 0;
			Hst_ram_din <= 0;
			Hst_ram_wr_addr <= 0;
			Hst_ram_rd_addr <= 0;
        end
    else
		begin	
			Ram_cnt_1 <= Ram_cnt;
			Ram_cnt_2 <= Ram_cnt_1;
			Ram_cnt_3 <= Ram_cnt_2;
				
            case(hst_state)
            3'b001 :    begin
                            Hst_ram_wr <= 0;
                            Hst_ram_rd <= 1;
                            Hst_ram_din <= 0;
                            if(Ram_cnt >= 16383)
                                Ram_cnt <= 0;
                            else
                                Ram_cnt <= Ram_cnt + 1;
                            Hst_ram_rd_addr <= Ram_cnt;
                        end
            3'b010 :    begin
                            Hst_ram_wr <= 0;
                            Hst_ram_rd <= 1;
                            Hst_ram_din <= 0;
                            if(Ram_cnt >= 16383)
                                Ram_cnt <= 0;
                            else
                                Ram_cnt <= Ram_cnt + 1;
                            if(Ram_cnt <= data_mean)
                                Hst_ram_rd_addr <= data_mean - Ram_cnt;
                            else
                                Hst_ram_rd_addr <= Ram_cnt;
                        end
            3'b100 :    begin
                            Hst_ram_wr <= 1;
                            Hst_ram_rd <= 0;
                            Hst_ram_din <= 0;
                            if(Ram_cnt >= 16383)
                                Ram_cnt <= 0;
                            else
                                Ram_cnt <= Ram_cnt + 1;
                            Hst_ram_wr_addr <= Ram_cnt;
                        end
            default :   begin
                            Ram_cnt <= 0;
                            Hst_ram_rd_addr <= Hst_fifo_dout;
                            Hst_ram_wr_addr <= Hst_ram_rd_addr;
                            Hst_ram_rd <= p_cnt == 0 && h_cnt >= 3 && h_cnt < 643;
                            Hst_ram_wr <= p_cnt == 2 && h_cnt >= 3 && h_cnt < 643;
                            if(Hst_ram_dout == 255)
                                Hst_ram_din <= 255;
                            else
                                Hst_ram_din <= Hst_ram_dout + 1;
                        end
            endcase
        end
end




Hist_ram Hist_ram (
  .clka(clkin_4x),    // input wire clka
  .wea(Hst_ram_wr),      // input wire [0 : 0] wea
  .addra(Hst_ram_wr_addr),  // input wire [15 : 0] addra
  .dina(Hst_ram_din),    // input wire [7 : 0] dina
  .clkb(clkin_4x),    // input wire clkb
  .enb(Hst_ram_rd),      // input wire enb
  .addrb(Hst_ram_rd_addr),  // input wire [15 : 0] addrb
  .doutb(Hst_ram_dout)  // output wire [7 : 0] doutb
);




reg [17:0]hst_cnt;
reg [17:0]hst_cnt_r;
reg [17:0]div_hsum;
reg [17:0]div_lsum;
reg [17:0]hst_hsum;
reg [17:0]hst_lsum;
reg [23:0]divisor;
reg [23:0]divisor_r;
reg [31:0]dividend;
always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			hst_hsum <= 0;
			hst_lsum <= 0;
			div_hsum <= 0;
			div_lsum <= 0;
			divisor <= 0;
			divisor_r <= 0;
			dividend <= 0;
			hst_cnt <= 0;
			hst_cnt_r <= 0;
		end
	else
		begin
            if(b_cnt == 16383+15)//65550)
            hst_cnt <= hst_cnt_r;

            dividend <= Ram_cnt_3 <= data_mean ? div_lsum << 13 : div_hsum << 13;
            divisor <= divisor_r;

            case(hst_state_2)
            3'b001 :    begin
                            if(Hst_ram_dout > 2)
                                hst_cnt_r <= hst_cnt_r + 1;
                            else ;
                            if(Ram_cnt_2 <= data_mean)
                                begin
                                    if(Hst_ram_dout >= 100)
                                        hst_lsum <= hst_lsum + 100;
                                    else
                                        hst_lsum <= hst_lsum + Hst_ram_dout;
                                end
                            else 
                                begin
                                    if(Hst_ram_dout >= 100)
                                        hst_hsum <= hst_hsum + 100;
                                   else
                                        hst_hsum <= hst_hsum + Hst_ram_dout;
                                end
                        end
            3'b010 :    begin
                            if(Ram_cnt_2 <= data_mean)
                                begin
                                    divisor_r <= {6'd0,hst_lsum};
                                   if(Hst_ram_dout >= 100)
                                        div_lsum <= div_lsum + 100;
                                   else
                                        div_lsum <= div_lsum + Hst_ram_dout;
                                end
                            else 
                                begin
                                    divisor_r <={6'd0, hst_hsum};
                                    if(Hst_ram_dout >= 100)
                                        div_hsum <= div_hsum + 100;
                                    else
                                        div_hsum <= div_hsum + Hst_ram_dout;
                                end
                        end
            3'b100 :    begin
                            hst_hsum <= 0;
                            hst_lsum <= 0;
                            div_hsum <= 0;
                            div_lsum <= 0;
                            divisor <= 0;
                            divisor_r <= 0;
                            dividend <= 0;
                            hst_cnt_r <= 0;
                        end
            default :   ;
            endcase
        end
end

wire [55:0]div_q;
Hist_divider Hist_divider (
  .aclk(clkin_4x),                                      // input wire aclk
  .s_axis_divisor_tvalid(1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(divisor),      // input wire [23 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(dividend),    // input wire [31 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(div_q)            // output wire [55 : 0] m_axis_dout_tdata
);





reg [31:0]div_q_r;
reg [31:0]div_q_rr;
always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
            div_q_r <= 0;
            div_q_rr <= 0;
        end
    else
        begin
            div_q_r <= div_q >> 24;
            div_q_rr <= div_q_r;
        end
end

wire [39:0]HST_mult_hdata;
wire [7:0]HST_mult_hdata_r;
assign HST_mult_hdata_r = HST_mult_hdata[39:13] >= MAP_Max_r ? MAP_Max_r : HST_mult_hdata[20:13];

mult_gen_2 HIST_mult_H (
  .CLK(clkin_4x),  // input wire CLK
  .A(div_q_rr),      // input wire [31 : 0] A
  .B(MAP_Max_r),      // input wire [7 : 0] B
  .P(HST_mult_hdata)      // output wire [39 : 0] P
);



wire [39:0]HST_mult_ldata;
wire [7:0]HST_mult_ldata_r;
assign HST_mult_ldata_r = HST_mult_ldata[39:13] >= MAP_Min_r ? MAP_Min_r : HST_mult_ldata[20:13];
mult_gen_2 HIST_mult_L (
  .CLK(clkin_4x),  // input wire CLK
  .A(div_q_rr),      // input wire [31 : 0] A
  .B(MAP_Min_r),      // input wire [7 : 0] B
  .P(HST_mult_ldata)      // output wire [39 : 0] P
);

reg hst_map_wr;
reg [7:0]hst_map_din;
reg [7:0]HST_mult_hdata_rr;
reg [7:0]HST_mult_ldata_rr;
reg [13:0]hst_map_addr;
reg [13:0]map_cnt;
reg [13:0]map_cnt_r;
always @(posedge clkin_4x or negedge rst_n)
begin
	if(!rst_n)
		begin
			hst_map_wr <= 0;
			hst_map_din <= 0;
			hst_map_addr <= 0;
			HST_mult_hdata_rr <= 0;
			HST_mult_ldata_rr <= 0;
			map_cnt <= 0;
			map_cnt_r <= 0;
		end
	else
		begin
			hst_map_wr <= b_cnt >= 16495 && b_cnt < 16495+16384;
            if(b_cnt >= 16494 && b_cnt < 16494+16383)
                begin
                    if(map_cnt == 16383)
                        map_cnt <= 0;
                    else
                        map_cnt <= map_cnt + 1;
                end
            else
                map_cnt <= 0;

            map_cnt_r <= map_cnt;

            if(map_cnt == data_mean)
                HST_mult_hdata_rr <= 0;
            else
                begin
                    if(HST_mult_hdata_r >= HST_mult_hdata_rr + 1)
                        HST_mult_hdata_rr <= HST_mult_hdata_rr + 1;
                    else    
                        HST_mult_hdata_rr <= HST_mult_hdata_r;
                end

            if(map_cnt == 0)
                HST_mult_ldata_rr <= 0;
            else
                begin
                    if(HST_mult_ldata_r > HST_mult_ldata_rr + 1)
                        HST_mult_ldata_rr <= HST_mult_ldata_rr + 1;
                    else    
                        HST_mult_ldata_rr <= HST_mult_ldata_r;
                end

			if(map_cnt_r <= data_mean)
                begin
                    hst_map_din <= MAP_Mid_r - HST_mult_ldata_rr;
                    hst_map_addr <= data_mean - map_cnt_r;
                end
			else 
                begin
                    hst_map_din <= MAP_Mid_r + HST_mult_hdata_rr;
                    hst_map_addr <= map_cnt_r;
                end

		end
end

wire [7:0]hst_dout_r;
wire signed[8:0]hst_dout_rr;
assign hst_dout_rr = hst_dout_r;


Hist_map_1 Hist_map_1 (
  .clka(clkin_4x),    // input wire clka
  .wea(hst_map_wr),      // input wire [0 : 0] wea
  .addra(hst_map_addr),  // input wire [15 : 0] addra
  .dina(hst_map_din),    // input wire [7 : 0] dina
  .clkb(clkin),    // input wire clkb
  .enb(HS_in&&VS_in),      // input wire enb
  .addrb(Lfdata_in),  // input wire [15 : 0] addrb
  .doutb(hst_dout_r)  // output wire [7 : 0] doutb
);
reg hst_hs_r;
reg hst_vs_r;
reg signed[13:0]Hfdata_in_r;
always @(posedge clkin or negedge rst_n)
begin
	if(!rst_n)
		begin
			hst_hs <= 0;
			hst_hs_r <= 0;
			hst_vs <= 0;
			hst_vs_r <= 0;
			hst_dout <= 0;
			Hfdata_in_r <= 0;
		end
	else
		begin
			hst_hs_r <= HS_in;
			hst_vs_r <= VS_in;
			hst_hs <= hst_hs_r;
			hst_vs <= hst_vs_r;
			Hfdata_in_r <= Hfdata_in;
            if(hst_dout_rr + Hfdata_in_r >= 255)
                hst_dout <= 255;
            else if(hst_dout_rr + Hfdata_in_r <= 0)
                hst_dout <= 0;
            else
                hst_dout <= hst_dout_rr + Hfdata_in_r;
		end
end

ila_1 ila_u(
.clk(clkin),
.probe0(Hst_fifo_dout),//14
.probe1(Hst_ram_dout),//8
.probe2(div_q_r),//32
.probe3(HST_mult_hdata),//40
.probe4(hst_dout_r)//8
);





endmodule
