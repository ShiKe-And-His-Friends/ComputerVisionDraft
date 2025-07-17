`timescale 1ns / 1ps
`include "top_define.vh"
//////////////////////////////////////////////////////////////////////////////////
// Company:  vital
// Engineer: zzy
// 
// Create Date: 2024/08/05 9:19:58
// Design Name: 
// Module Name: hist_gen
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


module hist_gen
#(
parameter DW  = 14,
parameter IMAGE_WIDTH = 640,
parameter IMAGE_HEIGHT = 512,
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
//    input [7:0]                i_img_data_h,
    
    input [HIST_SUM_DW - 1:0]  i_limit_up,   //统计出现次数上限
    input [HIST_SUM_DW - 1:0]  i_limit_down, //统计出现次数下限
    
    output                     o_field_limit,
    output                     o_line_limit,
    output [DW - 1:0]          o_data_limit,
    output                     o_bc_cal_en,
    
    output                     o_hist_wr_field_vld,
    output                     o_hist_wr_line_vld,
    output [HIST_RAM_DW - 1:0] o_hist_wr_data,
    output                     o_hist_rd_vld,
    output [HIST_RAM_DW - 1:0] o_hist_rd_data, 
    
    output reg [DW - 1:0]      o_hist_max,   //右侧灰度级抛点位
    output reg [DW - 1:0]      o_hist_min,  //左侧灰度级抛点位
    output reg [DW - 1:0]      o_data_avg   //面阵均值
    
    
);

//delay
localparam DLY_NUM              = 21;  //流水延时

//state define
localparam IDLE                 = 3'b000;
localparam HIST_INDEX           = 3'b001;
localparam WAIT_SEARCH           = 3'b011;
localparam HIST_SEARCH           = 3'b010;
localparam WAIT_CLEAR           = 3'b110;
localparam HIST_CLEAR           = 3'b111;

//
reg [DW - 1:0] img_data;
reg            field_vld;
reg            field_vld_d1;
reg            field_vld_d2;

reg            line_vld;
reg            line_vld_d1;
reg            line_vld_d2;

//
wire [HIST_RAM_DW:0]     index_sum;
wire hist_clear_en;
//port a
wire                     bram_port_a_wena;
wire [HIST_RAM_DW - 1:0] bram_port_a_data;
wire [HIST_RAM_AW - 1:0] bram_port_a_addr;
wire [HIST_RAM_DW - 1:0] bram_port_a_q;
//port b 

wire                     bram_port_b_wenb;
wire [HIST_RAM_DW - 1:0] bram_port_b_data;
wire [HIST_RAM_AW - 1:0] bram_port_b_addr;
wire [HIST_RAM_DW - 1:0] bram_port_b_q;



wire                  hist_search_en;
reg                   hist_search_en_r;
reg                   hist_clear_en_r;

//
reg                     hist_search_en_d1;
reg                     hist_search_en_d2;
reg [HIST_SUM_DW - 1:0] search_max_cnt;
reg [HIST_SUM_DW - 1:0] search_min_cnt;
reg [HIST_SUM_DW - 1:0] search_max_cnt_d;
reg [HIST_SUM_DW - 1:0] search_min_cnt_d;
//reg [HIST_RAM_AW - 1:0] search_addr_a_d1;
//reg [HIST_RAM_AW - 1:0] search_addr_a_d2;
//reg [HIST_RAM_AW - 1:0] search_addr_b_d1;
//reg [HIST_RAM_AW - 1:0] search_addr_b_d2;

assign o_bc_cal_en = hist_clear_en;

//input signal flows
always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        field_vld <= 1'b0;
        field_vld_d1 <= 1'b0;
        field_vld_d2 <= 1'b0;
        
        line_vld <= 1'b0;
        line_vld_d1 <= 1'b0;
        line_vld_d2 <= 1'b0;
        
        img_data <= {DW{1'b0}};
    end
    else begin
        //
        field_vld <= i_field_vld;
        field_vld_d1 <= field_vld;
        field_vld_d2 <= field_vld_d1;
        
        line_vld <= i_line_vld;
        line_vld_d1 <= line_vld;
        line_vld_d2 <= line_vld_d1;
        
        img_data <= i_img_data;
    end
end

assign o_field_limit = field_vld;
assign o_line_limit  = line_vld;
assign o_data_limit  = img_data;

//index sig
wire [HIST_RAM_AW - 1:0]    data_nxt_index;
wire                        rd_nxt_index;
reg  [HIST_RAM_AW - 1:0]    data_cur_index;
reg                         rd_cur_index;
reg  [HIST_RAM_AW - 1:0]    data_pre_index;
reg                         rd_pre_index;

assign data_nxt_index = img_data[HIST_RAM_AW - 1:0];
assign rd_nxt_index   = line_vld;

always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        data_cur_index <= {HIST_RAM_AW{1'b0}};
        rd_cur_index   <= 1'b0;
        
        data_pre_index <= {HIST_RAM_AW{1'b0}};
        rd_pre_index   <= 1'b0;
    end
    else begin
        rd_cur_index <= rd_nxt_index;
        rd_pre_index <= rd_cur_index;
        data_cur_index <= data_nxt_index;
        data_pre_index <= data_cur_index;
    end
end


wire rd_index_start;
wire rd_index_end;

wire data_update;
assign data_update = (data_cur_index != data_pre_index); 
reg [HIST_RAM_DW - 1:0] equal_cnt; 

assign rd_index_start = (~rd_pre_index) & rd_cur_index;
assign rd_index_end   = rd_cur_index & (~rd_nxt_index);

//reg [7:0] abs_img_data_h;
//data_h---abs
//always@(posedge i_clk or posedge i_rst)begin
//    if(i_rst)begin
//        abs_img_data_h <= 8'd0;
//    end
//    else begin
//        if(i_img_data_h[7] == 1)begin
//            abs_img_data_h <= ~i_img_data_h + 1;
//        end
//        else begin
//            abs_img_data_h <= i_img_data_h;
//        end
//    end
//end

//equal
always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        equal_cnt <= {HIST_RAM_DW{1'b1}};
    end
    else begin
        if(rd_cur_index == 1'b1)begin
            if(rd_index_start || (data_cur_index != data_pre_index))begin
                equal_cnt <= {HIST_RAM_DW{1'b0}} + 1;
            end
            else if(equal_cnt < (2**HIST_RAM_DW - 1))begin 
                equal_cnt <= equal_cnt + 1;
            end
            else begin
                equal_cnt <= {HIST_RAM_DW{1'b1}}; 
            end
        end
        else begin
            equal_cnt <= equal_cnt;
        end
    end
 end
 
 //state machine
 reg [2:0]                state_cur;
 reg [2:0]                state_nxt;
 reg                      dly_over_flag;
 reg [4:0]                dly_cnt; 
 reg                      addr_over_flag;
 reg [HIST_RAM_AW - 1:0]  addr_cnt; 
 
 //state_machine_first 
 always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        state_cur <= IDLE;
    end
    else begin
        state_cur <= state_nxt;
    end
end
 
//state_machine_second
always@(*)begin
    state_nxt <= IDLE;
    case(state_cur)
        IDLE:begin
                if(line_vld)begin 
                    state_nxt <= HIST_INDEX;
                end
                else begin
                    state_nxt <= IDLE;
                end
        end
        //
        HIST_INDEX:begin
                if(~line_vld)begin 
                    state_nxt <= WAIT_SEARCH;
                end
                else begin
                    state_nxt <= HIST_INDEX;
                end        
        end
        //
        WAIT_SEARCH:begin
                if(dly_over_flag == 1'b1)begin //
                    state_nxt <= HIST_SEARCH;
                end
                else begin
                    state_nxt <= WAIT_SEARCH;
                end            
        end
        //
        HIST_SEARCH:begin
                if(addr_over_flag == 1'b1)begin //
                    state_nxt <= WAIT_CLEAR;
                end
                else begin
                    state_nxt <= HIST_SEARCH;
                end            
        end
        
        WAIT_CLEAR:begin
                if(dly_over_flag == 1'b1)begin //
                    state_nxt <= HIST_CLEAR;
                end
                else begin
                    state_nxt <= WAIT_CLEAR;
                end            
        end
        
        HIST_CLEAR:begin
                if(addr_over_flag == 1'b1)begin //
                    state_nxt <= IDLE;
                end
                else begin
                    state_nxt <= HIST_CLEAR;
                end        
        end
        
        default:begin
           state_nxt <= IDLE; 
        end
    endcase
end



//
reg                     ram_wena;
//reg                     ram_wr_a;
reg [HIST_RAM_AW - 1:0] ram_addr_a;
//reg                     ram_rd_a;

reg                     ram_wenb;
//reg                     ram_wr_b;
reg [HIST_RAM_AW - 1:0] ram_addr_b;
//reg                     ram_rd_b;

//reg ram_a_en;
//reg ram_b_en;


//state_machine_third    
 always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        hist_search_en_r  <= 1'b0;
        hist_clear_en_r  <= 1'b0;
        ram_wena <= 1'b0;
        ram_addr_a <= {HIST_RAM_AW{1'b0}};
        ram_wenb <= 1'b0;
        ram_addr_b <= {HIST_RAM_AW{1'b1}};
    end
    else begin
        case(state_cur)
            IDLE:begin
                hist_clear_en_r <= 1'b0;
                ram_wena <= 1'b0;
                ram_addr_a <= {HIST_RAM_AW{1'b0}};
                ram_wenb <= 1'b0;
                ram_addr_b <= {HIST_RAM_AW{1'b1}};
            end
            
            HIST_INDEX:begin
                if(line_vld == 1'b1)begin
                    ram_addr_a <= data_nxt_index;
                    ram_wena   <= 1'b0;
                end
                else begin
                    ram_addr_a <= {HIST_RAM_AW{1'b0}};
                    ram_wena   <= 1'b0;
                end
                
                if(line_vld_d1 == 1'b1)begin
                    ram_wenb <= rd_index_end||(rd_cur_index &&(data_cur_index!=data_nxt_index)); //最后一个像素或遇到不同像素了
                    ram_addr_b<= data_cur_index;
                end
                else begin
                    ram_wenb <= 1'b0;
                    ram_addr_b <= {HIST_RAM_AW{1'b1}};
                end
            end
            
            WAIT_SEARCH:begin
                ram_wena <= 1'b0;
                ram_addr_a <= {HIST_RAM_AW{1'b0}};
                ram_wenb <= 1'b0;
                ram_addr_b <= {HIST_RAM_AW{1'b1}};
            end
            
            HIST_SEARCH:begin
                hist_search_en_r <= 1'b1;
                ram_addr_a <= addr_cnt;
                ram_wena <= 1'b0; 
                ram_wenb <= 1'b0; 
                ram_addr_b <= {HIST_RAM_AW{1'b1}} - addr_cnt;
                
            end
            
            WAIT_CLEAR:begin
                hist_search_en_r <= 1'b0;
                ram_wena <= 1'b0; 
                ram_addr_a <= {HIST_RAM_AW{1'b0}};
                ram_wenb <= 1'b0;
                ram_addr_a <= {HIST_RAM_AW{1'b1}};
            end
            
            HIST_CLEAR:begin
                hist_clear_en_r <= 1'b1;
                ram_wena <= 1'b0;
                ram_addr_a <= {HIST_RAM_AW{1'b0}};
//                ram_addr_a <= addr_cnt;
                ram_wenb <= 1'b1;
                ram_addr_b <= addr_cnt;
            end
            
            default:begin
                hist_clear_en_r <= 1'b0;
                ram_wena <= 1'b0;
                ram_addr_a <= {HIST_RAM_AW{1'b0}};
                ram_wenb <= 1'b0;
                ram_addr_b <= {HIST_RAM_AW{1'b1}};
            end
        endcase
    end
end


assign hist_search_en = (i_rst)?1'b0:hist_search_en_r;
assign hist_clear_en = (i_rst)?1'b0:hist_clear_en_r;
assign index_sum = (i_rst)?13'd0:({1'b0,bram_port_a_q} + {1'b0,equal_cnt}); 

assign bram_port_a_data = {HIST_RAM_DW{1'b0}};
assign bram_port_a_addr = ram_addr_a;
assign bram_port_a_wena = (i_rst)?1'b0:ram_wena;

assign bram_port_b_data = (state_cur==HIST_CLEAR||state_cur==IDLE)?{HIST_RAM_DW{1'b0}}:index_sum[HIST_RAM_DW]?{HIST_RAM_DW{1'b1}}:index_sum[HIST_RAM_DW-1:0];
                            
assign bram_port_b_addr = ram_addr_b;
assign bram_port_b_wenb = (i_rst)?1'b0:ram_wenb;

hist_gen_ram inst_hist_gen_ram (
  .clka(i_clk),    // input wire clka
  .ena(1'b1),      // input wire ena
  .rsta(i_rst),            // input wire rsta
  .wea(bram_port_a_wena),      // input wire [0 : 0] wea
  .addra(bram_port_a_addr),  // input wire [13 : 0] addra
  .dina(bram_port_a_data),    // input wire [11 : 0] dina
  .douta(bram_port_a_q),  // output wire [11 : 0] douta
  .clkb(i_clk),    // input wire clkb
  .enb(1'b1),      // input wire enb
  .rstb(i_rst),            // input wire rstb
  .web(bram_port_b_wenb),      // input wire [0 : 0] web
  .addrb(bram_port_b_addr),  // input wire [11 : 0] addrb
  .dinb(bram_port_b_data),    // input wire [13 : 0] dinb
  .doutb(bram_port_b_q)  // output wire [13 : 0] doutb
);

assign o_hist_wr_field_vld = field_vld_d1;
assign o_hist_wr_line_vld = line_vld_d2;
assign o_hist_wr_data     = bram_port_b_data;

assign o_hist_rd_data     = bram_port_a_q;
assign o_hist_rd_vld      = hist_search_en_d1;

always @(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        dly_cnt <= 5'd0;
        dly_over_flag <= 1'b0;
    end
    else begin
        if((state_cur == WAIT_CLEAR)||(state_cur == WAIT_SEARCH))begin
            dly_cnt <= dly_cnt + 1;
        end
        else begin
            dly_cnt <= 5'd0;
        end
        
        if(dly_cnt < DLY_NUM)begin
            dly_over_flag <= 1'b0;
        end
        else begin
            dly_over_flag <= 1'b1;
        end
    end
end

always @(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        addr_cnt <= {HIST_RAM_AW{1'b0}};
        //que?
    end
    else begin
        case(state_cur)
            HIST_SEARCH:begin
                addr_cnt <= addr_cnt + 1;
                if(addr_cnt < (2**HIST_RAM_AW - 2))begin
                    addr_over_flag <= 1'b0;
                end
                else begin
                    addr_over_flag <= 1'b1;
                end
            end
            
            //
            HIST_CLEAR:begin
                addr_cnt <= addr_cnt + 1;
                if(addr_cnt < (2**(HIST_RAM_AW) - 2))begin
                    addr_over_flag <= 1'b0;
                end
                else begin
                    addr_over_flag <= 1'b1;
                end
            end
            default:begin
                addr_cnt <= {HIST_RAM_AW{1'b0}};
                addr_over_flag <= 1'b0;
            end
        endcase
    end
end

reg [HIST_RAM_AW - 1:0] ram_addr_a_d1;
reg [HIST_RAM_AW - 1:0] ram_addr_a_d2;
reg [HIST_RAM_AW - 1:0] ram_addr_b_d1;
reg [HIST_RAM_AW - 1:0] ram_addr_b_d2;


always @(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        hist_search_en_d1 <= 1'b0;
        hist_search_en_d2 <= 1'b0;
        search_min_cnt <= {HIST_SUM_DW{1'b0}};
        search_max_cnt <= {HIST_SUM_DW{1'b0}};
        search_min_cnt_d <= {HIST_SUM_DW{1'b0}};
        search_max_cnt_d <= {HIST_SUM_DW{1'b0}};
        ram_addr_a_d1    <= {HIST_RAM_AW{1'b0}};
        ram_addr_a_d2    <= {HIST_RAM_AW{1'b0}};
        ram_addr_b_d1    <= {HIST_RAM_AW{1'b0}};
        ram_addr_b_d2    <= {HIST_RAM_AW{1'b0}};
    end
    else begin
        hist_search_en_d1<= hist_search_en;
        hist_search_en_d2<= hist_search_en_d1;
        if(hist_search_en_d1 == 1'b1)begin
            search_min_cnt <= search_min_cnt + bram_port_a_q;
            search_max_cnt <= search_max_cnt + bram_port_b_q;
        end
        else begin
            search_min_cnt <= {HIST_SUM_DW{1'b0}};
            search_max_cnt <= {HIST_SUM_DW{1'b0}};
        end
        search_min_cnt_d <= search_min_cnt;
        search_max_cnt_d <= search_max_cnt;
        
        ram_addr_a_d1 <= ram_addr_a;
        ram_addr_a_d2 <= ram_addr_a_d1;
        ram_addr_b_d1 <= ram_addr_b;
        ram_addr_b_d2 <= ram_addr_b_d1;
    end
end

reg [DW-1:0] hist_min;
reg [DW-1:0] hist_max;

always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        hist_min <= {DW{1'b0}};
        hist_max <= {DW{1'b0}}; 
    end
    else begin
        if(hist_search_en_d2 == 1'b1)begin
            if((search_min_cnt_d < i_limit_down)&&(search_min_cnt >= i_limit_down))begin
                hist_min <= {{DW - HIST_RAM_AW{1'b0}},ram_addr_a_d2};  
            end
            if((search_max_cnt_d < i_limit_up) && (search_max_cnt >= i_limit_up))begin
                hist_max <= {{DW - HIST_RAM_AW{1'b0}},ram_addr_b_d2}; 
            end
            
        end
    end
end

reg signed [HIST_SUM_DW + DW - 1:0] img_data_sum;
wire      [HIST_SUM_DW + DW - 11:0] img_data_q;
reg       [HIST_RAM_DW + DW - 11:0] data_aver_pre;


 always@(posedge i_clk or posedge i_rst)begin
    if(i_rst)begin
        img_data_sum <= {(HIST_SUM_DW){1'b0}};
        o_data_avg <=  {DW{1'b0}};
        o_hist_min <=  {DW{1'b0}};
        o_hist_max <=  {DW{1'b0}};
        data_aver_pre <= {HIST_RAM_DW + DW - 10{1'b0}}; 
    end
    else begin
        if(i_line_vld)begin
           img_data_sum <=  img_data_sum  + i_img_data;
        end
        else if((state_cur == WAIT_CLEAR) && dly_over_flag)begin 
            data_aver_pre <= o_data_avg; 
            o_data_avg <= (img_data_q[DW-1:0] + data_aver_pre)/2; 
            img_data_sum <= {HIST_SUM_DW + DW{1'b0}};  
            o_hist_min <= hist_min;  
            o_hist_max <= hist_max;  
        end
    end
end


wire [38:0] div_data_out;

//div
div_gen_0 inst_div_gen_0 (
  .aclk(i_clk),                                      // input wire aclk
  .s_axis_divisor_tvalid(1'b1),    // input wire s_axis_divisor_tvalid
  .s_axis_divisor_tdata(9'd320),      // input wire [15 : 0] s_axis_divisor_tdata
  .s_axis_dividend_tvalid(1'b1),  // input wire s_axis_dividend_tvalid
  .s_axis_dividend_tdata(img_data_sum[HIST_SUM_DW + DW - 1:10]),    // input wire [15 : 0] s_axis_dividend_tdata
  .m_axis_dout_tvalid(),          // output wire m_axis_dout_tvalid
  .m_axis_dout_tdata(div_data_out)            // output wire [31 : 0] m_axis_dout_tdata
);

assign img_data_q = div_data_out[38:16];


//wire [127:0] probe0; 

//ila_0 your_instance_name (
//	.clk(i_clk), // input wire clk
//	.probe0(probe0) // input wire [127:0] probe0
//);

//assign probe0 = 
//{

//bram_port_b_wenb,
//bram_port_b_data,
//bram_port_b_addr,
//bram_port_b_q,


//bram_port_a_wena,
//bram_port_a_data,
//bram_port_a_addr,
//bram_port_a_q    






//};

endmodule
