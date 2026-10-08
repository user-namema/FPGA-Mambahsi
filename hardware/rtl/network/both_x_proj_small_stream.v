`timescale 1ns/1ps
`default_nettype none

// Block1/2 x_proj engines trade excess service margin for fewer reduction
// resources: Spa is 32 inputs x two cycles; Spe keeps four output rows but
// each row is 8 inputs x two cycles.

module spa_x_proj_stream_small #(
    parameter integer BLOCK_ID=1,
    parameter integer PIXEL_COUNT=(BLOCK_ID==1)?64:16,
    parameter signed [15:0] DT_MULTIPLIER=16'sd24229,
    parameter signed [6:0] DT_SHIFT=7'sd20,
    parameter signed [15:0] B_MULTIPLIER=16'sd23482,
    parameter signed [6:0] B_SHIFT=7'sd21,
    parameter signed [15:0] C_MULTIPLIER=16'sd29259,
    parameter signed [6:0] C_SHIFT=7'sd21
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire [7:0] in_pixel_addr,
    input wire [5:0] in_channel_base,input wire [63:0] in_data,
    input wire cfg_weight_we,input wire [11:0] cfg_weight_addr,input wire signed [7:0] cfg_weight_data,
    input wire signed [31:0] cfg_dt_multiplier,input wire signed [6:0] cfg_dt_shift,
    input wire signed [31:0] cfg_b_multiplier,input wire signed [6:0] cfg_b_shift,
    input wire signed [31:0] cfg_c_multiplier,input wire signed [6:0] cfg_c_shift,
    output reg dt_valid,output reg [7:0] dt_pixel_addr,output reg [17:0] dt_data,
    output reg b_valid,output reg [7:0] b_pixel_addr,output reg [127:0] b_data,
    output reg c_valid,output reg [7:0] c_pixel_addr,output reg [127:0] c_data,
    output reg done
);
    // Two output rows share sixteen packed DSPs. Four 16-channel chunks keep
    // the original row throughput while halving the physical multiplier bank.
    localparam integer PAR=16;
    localparam integer CHUNKS=64/PAR;
    localparam integer CONTINUOUS_PREFETCH=(BLOCK_ID==1);
    // Block1 is the only scaled Spa stage whose two-row engine needs
    // 17*4=68 clocks/record, i.e. 4352 clocks/tile.  Its parameter ROM
    // already has four read ports, so use ports 2/3 and one additional
    // packed pair only in Block1.  Block2 remains the two-row area version.
    localparam integer ROW_LANES=(BLOCK_ID==1)?4:2;
    localparam integer PREFETCH_ROW=(BLOCK_ID==1)?28:30;
    localparam integer PIXEL_INDEX_BITS=$clog2(PIXEL_COUNT);
    wire unused_cfg=cfg_weight_we^cfg_weight_addr[0]^cfg_weight_data[0]
        ^cfg_dt_multiplier[0]^cfg_dt_shift[0]^cfg_b_multiplier[0]^cfg_b_shift[0]
        ^cfg_c_multiplier[0]^cfg_c_shift[0];

    reg [447:0] collect_input;
    wire wr_en=in_valid&&(in_channel_base==6'd56);
    wire [511:0] wr_data={in_data,collect_input};
    reg rd_en;reg [7:0] rd_addr;wire [511:0] rd_data;
    spa_x_input_bram_256x512 u_input_bram(
        .clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_pixel_addr),.dina(wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    reg [31:0] ready_count;
    always @(posedge clk) begin
        if(!rst_n) begin collect_input<=0;ready_count<=0;end
        else if(frame_start) begin ready_count<=0;end
        else if(in_valid) begin
            if(in_channel_base!=6'd56) collect_input[in_channel_base*8 +:64]<=in_data;
            if(in_channel_base==6'd56) ready_count<=ready_count+1'b1;
        end
    end

    localparam [2:0] ST_WAIT=0,ST_READ=1,ST_CAPTURE=2,ST_ISSUE=3;
    reg [2:0] state;reg [31:0] time_index;reg [5:0] row;reg [4:0] chunk;
    wire[7:0]local_pixel=
        {{(8-PIXEL_INDEX_BITS){1'b0}},
          time_index[PIXEL_INDEX_BITS-1:0]};
    reg [511:0] activation;
    reg next_prefetch_started;
    wire [31:0] next_record_sequence = time_index + 32'd1;
    wire next_record_ready=(ready_count > next_record_sequence);
    wire request=(state==ST_ISSUE);
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=ST_WAIT;time_index<=0;row<=0;chunk<=0;activation<=0;rd_en<=0;rd_addr<=0;
            next_prefetch_started<=0;
        end else if(frame_start) begin
            state<=ST_WAIT;time_index<=0;row<=0;chunk<=0;rd_en<=0;rd_addr<=0;
            next_prefetch_started<=0;
        end else begin
            rd_en<=0;
            case(state)
                ST_WAIT: begin
                    row<=0;chunk<=0;
                    next_prefetch_started<=0;
                    if(time_index<ready_count) begin rd_en<=1;rd_addr<=local_pixel;state<=ST_READ;end
                end
                ST_READ: state<=ST_CAPTURE;
                ST_CAPTURE: begin activation<=rd_data;next_prefetch_started<=0;state<=ST_ISSUE;end
                ST_ISSUE: begin
                    // The BRAM address is issued two useful dot-product
                    // cycles before the record boundary.  Its one-cycle
                    // synchronous result is therefore stable when the final
                    // chunk retires, so the next record starts immediately.
                    if(CONTINUOUS_PREFETCH&&(row==PREFETCH_ROW)
                        &&(chunk==CHUNKS-1)
                        &&next_record_ready) begin
                        rd_en<=1;
                        rd_addr<=(local_pixel==PIXEL_COUNT-1)?8'd0:local_pixel+1'b1;
                        next_prefetch_started<=1;
                    end
                    if(chunk==CHUNKS-1) begin
                        chunk<=0;
                        if(row==6'd32) begin
                            row<=0;time_index<=time_index+1'b1;
                            if(next_prefetch_started) begin
                                activation<=rd_data;
                                next_prefetch_started<=0;
                                state<=ST_ISSUE;
                            end else state<=ST_WAIT;
                        end
                        else row<=row+ROW_LANES;
                    end else chunk<=chunk+1'b1;
                end
                default: state<=ST_WAIT;
            endcase
        end
    end

`ifndef SYNTHESIS
    integer sim_continuous_record_transitions;
    integer sim_fallback_record_transitions;
    always @(posedge clk) begin
        if(!rst_n||frame_start) begin
            sim_continuous_record_transitions<=0;
            sim_fallback_record_transitions<=0;
        end else if((state==ST_ISSUE)&&(row==6'd32)
                    &&(chunk==CHUNKS-1)
                    ) begin
            if(next_prefetch_started)
                sim_continuous_record_transitions
                    <=sim_continuous_record_transitions+1;
            else
                sim_fallback_record_transitions
                    <=sim_fallback_record_transitions+1;
        end
    end
`endif

    wire [511:0] rom_weight_low,rom_weight_high,rom_weight_unused2,rom_weight_unused3;
    mamba_spa_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID)) u_weight_rom(
        .clk(clk),.en(request),.addr0(row),.addr1(row+1'b1),
        .addr2((row==6'd32)?6'd0:row+2'd2),
        .addr3((row==6'd32)?6'd0:row+2'd3),.data0(rom_weight_low),
        .data1(rom_weight_high),.data2(rom_weight_unused2),.data3(rom_weight_unused3));
    reg rom_valid;reg [127:0] rom_act;reg [18:0] rom_meta;
    wire[127:0]rom_weight_low_chunk=rom_weight_low[rom_meta[4:0]*128 +:128];
    wire[127:0]rom_weight_high_chunk=rom_weight_high[rom_meta[4:0]*128 +:128];
    wire[127:0]rom_weight_row2_chunk=rom_weight_unused2[rom_meta[4:0]*128 +:128];
    wire[127:0]rom_weight_row3_chunk=rom_weight_unused3[rom_meta[4:0]*128 +:128];
    always @(posedge clk) begin
        if(!rst_n) begin rom_valid<=0;rom_act<=0;rom_meta<=0;end
        else if(frame_start) begin rom_valid<=0;rom_meta<=0;end
        else begin
            rom_valid<=request;
            if(request) begin
                rom_act<=activation[chunk*128 +:128];
                rom_meta<={local_pixel,row,chunk};
            end
        end
    end

    wire dot_valid;wire signed[19:0]dot_low,dot_high;wire[18:0]dot_meta;
    wire dot_valid_23;wire signed[19:0]dot_row2,dot_row3;
    wire[18:0]dot_meta_23;
    mamba_dot16_pair_pipeline#(19) u_dot(
        .clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(rom_valid),
        .activation(rom_act),.weight_low(rom_weight_low_chunk),
        .weight_high(rom_weight_high_chunk),.in_meta(rom_meta),
        .out_valid(dot_valid),.out_sum_low(dot_low),.out_sum_high(dot_high),
        .out_meta(dot_meta));
    generate
        if(BLOCK_ID==1) begin:G_BLOCK1_ROWS23
            mamba_dot16_pair_pipeline#(19) u_dot_rows23(
                .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
                .in_valid(rom_valid),.activation(rom_act),
                .weight_low(rom_weight_row2_chunk),
                .weight_high(rom_weight_row3_chunk),.in_meta(rom_meta),
                .out_valid(dot_valid_23),.out_sum_low(dot_row2),
                .out_sum_high(dot_row3),.out_meta(dot_meta_23));
        end else begin:G_NO_ROWS23
            assign dot_valid_23=1'b0;
            assign dot_row2=20'sd0;
            assign dot_row3=20'sd0;
            assign dot_meta_23=19'd0;
        end
    endgenerate
    wire first_chunk=(dot_meta[4:0]==0);
    wire last_chunk=(dot_meta[4:0]==CHUNKS-1);
    (* use_dsp = "no" *) reg signed [47:0] accumulator[0:3];
    reg accumulator_result_valid;
    reg [13:0] accumulator_result_meta;
    integer acc_lane;
    always @(posedge clk) begin
        if(!rst_n) begin
            accumulator[0]<=0;accumulator[1]<=0;
            accumulator[2]<=0;accumulator[3]<=0;
            accumulator_result_valid<=0;accumulator_result_meta<=0;
        end else if(frame_start) begin
            accumulator_result_valid<=0;accumulator_result_meta<=0;
        end
        else if(dot_valid) begin
            accumulator_result_valid<=last_chunk;
            for(acc_lane=0;acc_lane<ROW_LANES;acc_lane=acc_lane+1)begin
                if(first_chunk) accumulator[acc_lane]<=acc_lane==0
                    ?{{28{dot_low[19]}},dot_low}:acc_lane==1
                    ?{{28{dot_high[19]}},dot_high}:acc_lane==2
                    ?{{28{dot_row2[19]}},dot_row2}
                    :{{28{dot_row3[19]}},dot_row3};
                else accumulator[acc_lane]<=accumulator[acc_lane]+(acc_lane==0
                    ?{{28{dot_low[19]}},dot_low}:acc_lane==1
                    ?{{28{dot_high[19]}},dot_high}:acc_lane==2
                    ?{{28{dot_row2[19]}},dot_row2}
                    :{{28{dot_row3[19]}},dot_row3});
            end
            if(last_chunk) accumulator_result_meta<=dot_meta[18:5];
        end else begin
            accumulator_result_valid<=0;
        end
    end

    wire [5:0] complete_row=accumulator_result_meta[5:0];
    wire signed [36:0] rq_product[0:3];genvar rq_lane;
    generate for(rq_lane=0;rq_lane<4;rq_lane=rq_lane+1)begin:G_PAIR_RQ
        if(rq_lane<ROW_LANES)begin:G_VALID_RQ
            wire[6:0]rq_row=complete_row+rq_lane;
            wire signed[15:0]rq_multiplier=(rq_row<2)?DT_MULTIPLIER:
                (rq_row<18)?B_MULTIPLIER:C_MULTIPLIER;
            requant_mult_21x16 u_rq(.CLK(clk),.A(accumulator[rq_lane][20:0]),
                .B(rq_multiplier),.P(rq_product[rq_lane]));
        end else begin:G_UNUSED_RQ
            assign rq_product[rq_lane]=37'sd0;
        end
    end endgenerate
    reg rv0,rv1,rv2;reg [13:0] rm0,rm1,rm2;
    always @(posedge clk) begin
        if(!rst_n||frame_start) begin rv0<=0;rv1<=0;rv2<=0;rm0<=0;rm1<=0;rm2<=0;end
        else begin
            rv0<=accumulator_result_valid;rv1<=rv0;rv2<=rv1;
            if(accumulator_result_valid) rm0<=accumulator_result_meta;
            if(rv0) rm1<=rm0;if(rv1) rm2<=rm1;
        end
    end
    function [8:0] q9;input signed [36:0] p;input signed [6:0] s;reg signed [63:0]v,m,r;begin
        v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>255)q9=9'h0ff;else if(r< -256)q9=9'h100;else q9=r[8:0];end endfunction
    function [7:0] q8;input signed [36:0] p;input signed [6:0] s;reg signed [63:0]v,m,r;begin
        v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    wire [7:0] result_pixel=rm2[13:6];wire [5:0] result_row_base=rm2[5:0];
    integer result_lane;integer result_row;
    always @(posedge clk) begin
        if(!rst_n) begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;dt_data<=0;b_data<=0;c_data<=0;
            dt_pixel_addr<=0;b_pixel_addr<=0;c_pixel_addr<=0;
        end else if(frame_start) begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;
            dt_pixel_addr<=0;b_pixel_addr<=0;c_pixel_addr<=0;
        end else begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;
            if(rv2) for(result_lane=0;result_lane<ROW_LANES;result_lane=result_lane+1) begin
                result_row=result_row_base+result_lane;
                case(result_row)
                    0,1: begin
                        dt_data[result_row*9 +:9]<=q9(rq_product[result_lane],DT_SHIFT);
                        if(result_row==1) begin dt_valid<=1;dt_pixel_addr<=result_pixel;end
                    end
                    2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17: begin
                        b_data[(result_row-2)*8 +:8]<=q8(rq_product[result_lane],B_SHIFT);
                        if(result_row==17) begin b_valid<=1;b_pixel_addr<=result_pixel;end
                    end
                    18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33: begin
                        c_data[(result_row-18)*8 +:8]<=q8(rq_product[result_lane],C_SHIFT);
                        if(result_row==33) begin
                            c_valid<=1;c_pixel_addr<=result_pixel;
                            if(result_pixel==PIXEL_COUNT-1) done<=1;
                        end
                    end
                endcase
            end
        end
    end
endmodule

module spe_x_proj_stream_small #(
    parameter integer BLOCK_ID=1,
    parameter integer PIXEL_COUNT=(BLOCK_ID==1)?64:16,
    parameter signed [15:0] DT_MULTIPLIER=16'sd17999,
    parameter signed [6:0] DT_SHIFT=7'sd16,
    parameter signed [15:0] B_MULTIPLIER=16'sd25796,
    parameter signed [6:0] B_SHIFT=7'sd20,
    parameter signed [15:0] C_MULTIPLIER=16'sd26002,
    parameter signed [6:0] C_SHIFT=7'sd20
)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire [7:0] in_pixel_addr,input wire [1:0] in_token,
    input wire [3:0] in_channel_base,input wire [63:0] in_data,
    input wire cfg_weight_we,input wire [9:0] cfg_weight_addr,input wire signed [7:0] cfg_weight_data,
    input wire signed [31:0] cfg_dt_multiplier,input wire signed [6:0] cfg_dt_shift,
    input wire signed [31:0] cfg_b_multiplier,input wire signed [6:0] cfg_b_shift,
    input wire signed [31:0] cfg_c_multiplier,input wire signed [6:0] cfg_c_shift,
    output reg dt_valid,output reg [7:0] dt_pixel_addr,output reg [1:0] dt_token,output reg [8:0] dt_data,
    output reg b_valid,output reg [7:0] b_pixel_addr,output reg [1:0] b_token,output reg [127:0] b_data,
    output reg c_valid,output reg [7:0] c_pixel_addr,output reg [1:0] c_token,output reg [127:0] c_data,
    output reg done
);
    // Block1 has enough DSP headroom to finish all 16 inputs in one issue;
    // Block2 retains the area-scaled two-chunk 8-input implementation.
    localparam integer PAR=(BLOCK_ID==1)?16:8;
    localparam integer CHUNKS=16/PAR;
    localparam integer RECORD_COUNT=PIXEL_COUNT*4;
    localparam integer RECORD_INDEX_BITS=$clog2(RECORD_COUNT);
    // A BMG synchronous read must be launched at least two clock edges before
    // the boundary that consumes rd_data.  Block1 has one chunk per row, so
    // row28 -> row32 provided only one edge and reused the previous record.
    // Block2 has two chunks and may keep the later row28 request.
    localparam integer PREFETCH_ROW=(BLOCK_ID==1)?24:28;
    wire unused_cfg=cfg_weight_we^cfg_weight_addr[0]^cfg_weight_data[0]
        ^cfg_dt_multiplier[0]^cfg_dt_shift[0]^cfg_b_multiplier[0]^cfg_b_shift[0]
        ^cfg_c_multiplier[0]^cfg_c_shift[0];
    reg [63:0] collect_low;wire [9:0] in_time={in_pixel_addr,in_token};
    wire wr_en=in_valid&&(in_channel_base==4'd8);wire [127:0] wr_data={in_data,collect_low};
    reg rd_en;reg [9:0] rd_addr;wire [127:0] rd_data;
    spe_x_input_bram_1024x128 u_input_bram(
        .clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_time),.dina(wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    reg [31:0] ready_count;
    always @(posedge clk) begin
        if(!rst_n) begin collect_low<=0;ready_count<=0;end
        else if(frame_start) begin ready_count<=0;end
        else if(in_valid) begin
            if(in_channel_base==0) collect_low<=in_data;
            if(in_channel_base==8) ready_count<=ready_count+1'b1;
        end
    end
    localparam [2:0] ST_WAIT=0,ST_READ=1,ST_CAPTURE=2,ST_ISSUE=3;
    reg [2:0] state;reg [31:0] time_index;reg [5:0] row;reg [2:0] chunk;reg [127:0] activation;
    wire[9:0]local_record=
        {{(10-RECORD_INDEX_BITS){1'b0}},
          time_index[RECORD_INDEX_BITS-1:0]};
    reg next_prefetch_started;
    wire [31:0] next_record_sequence = time_index + 32'd1;
    wire next_record_ready=(ready_count > next_record_sequence);
    wire request=(state==ST_ISSUE);
    always @(posedge clk) begin
        if(!rst_n) begin
            state<=ST_WAIT;time_index<=0;row<=0;chunk<=0;activation<=0;rd_en<=0;rd_addr<=0;
            next_prefetch_started<=0;
        end else if(frame_start) begin
            state<=ST_WAIT;time_index<=0;row<=0;chunk<=0;rd_en<=0;rd_addr<=0;
            next_prefetch_started<=0;
        end else begin
            rd_en<=0;
            case(state)
                ST_WAIT: begin
                    row<=0;chunk<=0;next_prefetch_started<=0;
                    if(time_index<ready_count) begin rd_en<=1;rd_addr<=local_record;state<=ST_READ;end
                end
                ST_READ: state<=ST_CAPTURE;
                ST_CAPTURE: begin activation<=rd_data;next_prefetch_started<=0;state<=ST_ISSUE;end
                ST_ISSUE: begin
                    // Fetch the following token while the last two output-row
                    // groups are still in flight.  This removes the W/R/C
                    // bubble with one flag and one comparator; unlike copying
                    // the Spa engine it adds no wide accumulator bank.
                    if((row==PREFETCH_ROW)&&(chunk==CHUNKS-1)
                        &&next_record_ready)begin
                        rd_en<=1;rd_addr<=(local_record==RECORD_COUNT-1)?10'd0:local_record+1'b1;
                        next_prefetch_started<=1;
                    end
                    if(chunk==CHUNKS-1) begin
                        chunk<=0;
                        if(row==6'd32) begin
                            row<=0;time_index<=time_index+1'b1;
                            if(next_prefetch_started)begin
                                activation<=rd_data;
                                next_prefetch_started<=0;
                                state<=ST_ISSUE;
                            end else state<=ST_WAIT;
                        end
                        else row<=row+3'd4;
                    end else chunk<=chunk+1'b1;
                end
                default: state<=ST_WAIT;
            endcase
        end
    end
`ifndef SYNTHESIS
    integer sim_continuous_record_transitions;
    integer sim_fallback_record_transitions;
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin
            sim_continuous_record_transitions<=0;
            sim_fallback_record_transitions<=0;
        end else if((state==ST_ISSUE)&&(row==6'd32)
            &&(chunk==CHUNKS-1))begin
            if(next_prefetch_started)
                sim_continuous_record_transitions
                    <=sim_continuous_record_transitions+1;
            else sim_fallback_record_transitions
                    <=sim_fallback_record_transitions+1;
        end
    end
`endif
    wire [127:0]w0,w1,w2,w3;
    mamba_spe_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID)) u_weight_rom(
        .clk(clk),.en(request),.addr0(row),.addr1(row+1'b1),
        .addr2(row+2'd2),.addr3(row+2'd3),
        .data0(w0),.data1(w1),.data2(w2),.data3(w3));
    reg rom_valid;reg [127:0] rom_act;reg [19:0] rom_meta;
    wire[511:0]rom_weights_full={w3,w2,w1,w0};
    generate
        if (BLOCK_ID == 1) begin : G_B1_ROM_PAYLOAD_FREE_RUNNING
            // Block1 previously used request as a CE on all 128 activation
            // bits (and their metadata).  The dot pipeline already uses
            // rom_valid as its sole qualification, so advance payload every
            // clock and reset only the valid bit.  This removes the FSM -> CE
            // route without changing the ROM/dot latency.
            always @(posedge clk) begin
                if(!rst_n || frame_start) rom_valid<=1'b0;
                else rom_valid<=request;
                rom_act<=activation;
                rom_meta<={1'b0,local_record,row,chunk};
            end
        end else begin : G_B2_ROM_PAYLOAD_GATED
            always @(posedge clk) begin
                if(!rst_n) begin rom_valid<=0;rom_act<=0;rom_meta<=0;end
                else if(frame_start) begin rom_valid<=0;rom_meta<=0;end
                else begin
                    rom_valid<=request;
                    if(request) begin
                        rom_act<=activation;
                        rom_meta<={1'b0,local_record,row,chunk};
                    end
                end
            end
        end
    endgenerate
    wire dot_valid;wire[19:0]dot_meta;wire signed[20:0]dot_part[0:3];
    wire dot_valid_pair[0:1];wire[19:0]dot_meta_pair[0:1];genvar dl;
    generate
        if(BLOCK_ID==1)begin:GEN_DOT16_FAST
            for(dl=0;dl<2;dl=dl+1)begin:G
                wire signed[19:0]dot16_low,dot16_high;
                mamba_dot16_pair_pipeline#(20)u_dot(.clk(clk),.rst_n(rst_n),
                    .frame_start(frame_start),.in_valid(rom_valid),.activation(rom_act),
                    .weight_low(rom_weights_full[(dl*2)*128 +:128]),
                    .weight_high(rom_weights_full[(dl*2+1)*128 +:128]),
                    .in_meta(rom_meta),.out_valid(dot_valid_pair[dl]),
                    .out_sum_low(dot16_low),.out_sum_high(dot16_high),
                    .out_meta(dot_meta_pair[dl]));
                assign dot_part[dl*2]={{1{dot16_low[19]}},dot16_low};
                assign dot_part[dl*2+1]={{1{dot16_high[19]}},dot16_high};
            end
        end else begin:GEN_DOT8_SCALED
            for(dl=0;dl<2;dl=dl+1)begin:G
                wire signed[18:0]dot8_low,dot8_high;
                mamba_dot8_pair_pipeline#(20)u_dot(.clk(clk),.rst_n(rst_n),
                    .frame_start(frame_start),.in_valid(rom_valid),
                    .activation(rom_act[rom_meta[2:0]*64 +:64]),
                    .weight_low(rom_weights_full[(dl*2)*128+rom_meta[2:0]*64 +:64]),
                    .weight_high(rom_weights_full[(dl*2+1)*128+rom_meta[2:0]*64 +:64]),
                    .in_meta(rom_meta),.out_valid(dot_valid_pair[dl]),
                    .out_sum_low(dot8_low),.out_sum_high(dot8_high),
                    .out_meta(dot_meta_pair[dl]));
                assign dot_part[dl*2]={{2{dot8_low[18]}},dot8_low};
                assign dot_part[dl*2+1]={{2{dot8_high[18]}},dot8_high};
            end
        end
    endgenerate
    assign dot_valid=dot_valid_pair[0];assign dot_meta=dot_meta_pair[0];
    wire last_chunk=(dot_meta[2:0]==CHUNKS-1);
    wire first_chunk=(dot_meta[2:0]==0);
    (* use_dsp = "no" *) reg signed[47:0]accumulator[0:3];
    reg accumulator_result_valid;reg[16:0]accumulator_result_meta;integer al;
    always@(posedge clk)begin
        if(!rst_n)begin accumulator_result_valid<=0;accumulator_result_meta<=0;
            for(al=0;al<4;al=al+1)accumulator[al]<=0;end
        else if(frame_start)begin accumulator_result_valid<=0;accumulator_result_meta<=0;end
        else if(dot_valid)begin accumulator_result_valid<=last_chunk;
            if(last_chunk)accumulator_result_meta<=dot_meta[19:3];
            for(al=0;al<4;al=al+1)begin
                if(first_chunk)accumulator[al]<={{27{dot_part[al][20]}},dot_part[al]};
                else accumulator[al]<=accumulator[al]+{{27{dot_part[al][20]}},dot_part[al]};
            end
        end else accumulator_result_valid<=0;
    end
    wire[5:0]complete_row=accumulator_result_meta[5:0];wire signed[36:0]rq_product[0:3];genvar ql;
    generate for(ql=0;ql<4;ql=ql+1)begin:GEN_RQ_HP
        wire[6:0]sr=complete_row+ql;
        wire signed[15:0]sm=(sr==0)?DT_MULTIPLIER:(sr<17)?B_MULTIPLIER:C_MULTIPLIER;
        requant_mult_21x16 u_rq(.CLK(clk),.A(accumulator[ql][20:0]),.B(sm),.P(rq_product[ql]));
    end endgenerate
    reg rv0,rv1,rv2;reg [16:0] rm0,rm1,rm2;
    always @(posedge clk) begin
        if(!rst_n||frame_start) begin rv0<=0;rv1<=0;rv2<=0;rm0<=0;rm1<=0;rm2<=0;end
        else begin
            rv0<=accumulator_result_valid;rv1<=rv0;rv2<=rv1;
            if(accumulator_result_valid)rm0<=accumulator_result_meta;if(rv0)rm1<=rm0;if(rv1)rm2<=rm1;
        end
    end
    function [8:0] q9;input signed [36:0] p;input signed [6:0] s;reg signed [63:0]v,m,r;begin
        v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>255)q9=9'h0ff;else if(r< -256)q9=9'h100;else q9=r[8:0];end endfunction
    function [7:0] q8;input signed [36:0] p;input signed [6:0] s;reg signed [63:0]v,m,r;begin
        v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    wire [10:0] result_time=rm2[16:6];wire [5:0] result_row_base=rm2[5:0];
    wire [7:0] result_pixel=result_time[9:2];wire [1:0] result_token=result_time[1:0];
    integer ol;integer result_row;
    always @(posedge clk) begin
        if(!rst_n) begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;dt_data<=0;b_data<=0;c_data<=0;
            dt_pixel_addr<=0;dt_token<=0;b_pixel_addr<=0;b_token<=0;c_pixel_addr<=0;c_token<=0;
        end else if(frame_start) begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;
            dt_pixel_addr<=0;dt_token<=0;b_pixel_addr<=0;b_token<=0;c_pixel_addr<=0;c_token<=0;
        end else begin
            dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;
            if(rv2) for(ol=0;ol<4;ol=ol+1) begin
                result_row=result_row_base+ol;
                if(result_row==0)begin dt_data<=q9(rq_product[ol],DT_SHIFT);dt_pixel_addr<=result_pixel;dt_token<=result_token;dt_valid<=1;end
                else if(result_row<17)begin b_data[(result_row-1)*8 +:8]<=q8(rq_product[ol],B_SHIFT);
                    if(result_row==16)begin b_valid<=1;b_pixel_addr<=result_pixel;b_token<=result_token;end end
                else if(result_row<33)begin c_data[(result_row-17)*8 +:8]<=q8(rq_product[ol],C_SHIFT);
                    if(result_row==32)begin c_valid<=1;c_pixel_addr<=result_pixel;c_token<=result_token;
                        if(result_time==(PIXEL_COUNT*4)-1)done<=1;end end
            end
        end
    end
endmodule

// Public wrappers preserve the pre-existing module names and ports.
module spa_x_proj_stream #(
    parameter integer BLOCK_ID=0,parameter integer PIXEL_COUNT=256,
    parameter signed [15:0]DT_MULTIPLIER=16'sd31024,parameter signed [6:0]DT_SHIFT=7'sd19,
    parameter signed [15:0]B_MULTIPLIER=16'sd17810,parameter signed [6:0]B_SHIFT=7'sd21,
    parameter signed [15:0]C_MULTIPLIER=16'sd16685,parameter signed [6:0]C_SHIFT=7'sd21
)(input wire clk,input wire rst_n,input wire frame_start,input wire downstream_record_ready,input wire in_valid,input wire[7:0]in_pixel_addr,
input wire[5:0]in_channel_base,input wire[63:0]in_data,input wire cfg_weight_we,input wire[11:0]cfg_weight_addr,
input wire signed[7:0]cfg_weight_data,input wire signed[31:0]cfg_dt_multiplier,input wire signed[6:0]cfg_dt_shift,
input wire signed[31:0]cfg_b_multiplier,input wire signed[6:0]cfg_b_shift,input wire signed[31:0]cfg_c_multiplier,input wire signed[6:0]cfg_c_shift,
output wire dt_valid,output wire[7:0]dt_pixel_addr,output wire[17:0]dt_data,output wire b_valid,output wire[7:0]b_pixel_addr,
output wire[127:0]b_data,output wire c_valid,output wire[7:0]c_pixel_addr,output wire[127:0]c_data,
output wire u_valid,output wire[7:0]u_pixel_addr,output wire[5:0]u_channel_base,output wire[63:0]u_data,
output wire record_credit_ready,output wire input_record_consumed,output wire output_record_launch,output wire done);
    generate if(BLOCK_ID==0)begin:G0
    block0_spa_x_proj_steam #(.BLOCK_ID(BLOCK_ID),
        .PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(DT_MULTIPLIER),
        .DT_SHIFT(DT_SHIFT),.B_MULTIPLIER(B_MULTIPLIER),
        .B_SHIFT(B_SHIFT),.C_MULTIPLIER(C_MULTIPLIER),
        .C_SHIFT(C_SHIFT))u(
        .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
        .downstream_record_ready(downstream_record_ready),
        .in_valid(in_valid),.in_pixel_addr(in_pixel_addr),
        .in_channel_base(in_channel_base),.in_data(in_data),
        .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),
        .cfg_weight_data(cfg_weight_data),
        .cfg_dt_multiplier(cfg_dt_multiplier),.cfg_dt_shift(cfg_dt_shift),
        .cfg_b_multiplier(cfg_b_multiplier),.cfg_b_shift(cfg_b_shift),
        .cfg_c_multiplier(cfg_c_multiplier),.cfg_c_shift(cfg_c_shift),
        .dt_valid(dt_valid),.dt_pixel_addr(dt_pixel_addr),.dt_data(dt_data),
        .b_valid(b_valid),.b_pixel_addr(b_pixel_addr),.b_data(b_data),
        .c_valid(c_valid),.c_pixel_addr(c_pixel_addr),.c_data(c_data),
        .u_valid(u_valid),.u_pixel_addr(u_pixel_addr),
        .u_channel_base(u_channel_base),.u_data(u_data),
        .record_credit_ready(record_credit_ready),
        .input_record_consumed(input_record_consumed),
        .output_record_launch(output_record_launch),.done(done));
    end else begin:GS
        // Block1/2 keep record-level streaming, but share a two-row/four-chunk
        // accumulator instead of materialising two complete 34-row result
        // banks.  U is fed directly from the Conv stream in the block wrapper.
        spa_x_proj_stream_small #(.BLOCK_ID(BLOCK_ID),
            .PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(DT_MULTIPLIER),
            .DT_SHIFT(DT_SHIFT),.B_MULTIPLIER(B_MULTIPLIER),
            .B_SHIFT(B_SHIFT),.C_MULTIPLIER(C_MULTIPLIER),
            .C_SHIFT(C_SHIFT))u(
            .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
            .in_valid(in_valid),.in_pixel_addr(in_pixel_addr),
            .in_channel_base(in_channel_base),.in_data(in_data),
            .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),
            .cfg_weight_data(cfg_weight_data),
            .cfg_dt_multiplier(cfg_dt_multiplier),.cfg_dt_shift(cfg_dt_shift),
            .cfg_b_multiplier(cfg_b_multiplier),.cfg_b_shift(cfg_b_shift),
            .cfg_c_multiplier(cfg_c_multiplier),.cfg_c_shift(cfg_c_shift),
            .dt_valid(dt_valid),.dt_pixel_addr(dt_pixel_addr),.dt_data(dt_data),
            .b_valid(b_valid),.b_pixel_addr(b_pixel_addr),.b_data(b_data),
            .c_valid(c_valid),.c_pixel_addr(c_pixel_addr),.c_data(c_data),
            .done(done));
        assign u_valid=1'b0;
        assign u_pixel_addr=8'd0;
        assign u_channel_base=6'd0;
        assign u_data=64'd0;
        assign record_credit_ready=1'b1;
        assign input_record_consumed=1'b0;
        assign output_record_launch=1'b0;
        wire unused_downstream_record_ready=downstream_record_ready;
    end endgenerate
endmodule

module spe_x_proj_stream #(
    parameter integer BLOCK_ID=0,parameter integer PIXEL_COUNT=256,
    parameter signed[15:0]DT_MULTIPLIER=16'sd17095,parameter signed[6:0]DT_SHIFT=7'sd16,
    parameter signed[15:0]B_MULTIPLIER=16'sd16866,parameter signed[6:0]B_SHIFT=7'sd19,
    parameter signed[15:0]C_MULTIPLIER=16'sd31499,parameter signed[6:0]C_SHIFT=7'sd20
)(input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,input wire[7:0]in_pixel_addr,input wire[1:0]in_token,
input wire[3:0]in_channel_base,input wire[63:0]in_data,input wire cfg_weight_we,input wire[9:0]cfg_weight_addr,input wire signed[7:0]cfg_weight_data,
input wire signed[31:0]cfg_dt_multiplier,input wire signed[6:0]cfg_dt_shift,input wire signed[31:0]cfg_b_multiplier,input wire signed[6:0]cfg_b_shift,
input wire signed[31:0]cfg_c_multiplier,input wire signed[6:0]cfg_c_shift,output wire dt_valid,output wire[7:0]dt_pixel_addr,output wire[1:0]dt_token,
output wire[8:0]dt_data,output wire b_valid,output wire[7:0]b_pixel_addr,output wire[1:0]b_token,output wire[127:0]b_data,
output wire c_valid,output wire[7:0]c_pixel_addr,output wire[1:0]c_token,output wire[127:0]c_data,output wire done);
    generate if(BLOCK_ID==0)begin:G0
        spe_x_proj_stream_block0 #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(DT_MULTIPLIER),.DT_SHIFT(DT_SHIFT),.B_MULTIPLIER(B_MULTIPLIER),.B_SHIFT(B_SHIFT),.C_MULTIPLIER(C_MULTIPLIER),.C_SHIFT(C_SHIFT))u(
            .clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(in_valid),.in_pixel_addr(in_pixel_addr),.in_token(in_token),.in_channel_base(in_channel_base),.in_data(in_data),
            .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),.cfg_weight_data(cfg_weight_data),.cfg_dt_multiplier(cfg_dt_multiplier),.cfg_dt_shift(cfg_dt_shift),
            .cfg_b_multiplier(cfg_b_multiplier),.cfg_b_shift(cfg_b_shift),.cfg_c_multiplier(cfg_c_multiplier),.cfg_c_shift(cfg_c_shift),
            .dt_valid(dt_valid),.dt_pixel_addr(dt_pixel_addr),.dt_token(dt_token),.dt_data(dt_data),.b_valid(b_valid),.b_pixel_addr(b_pixel_addr),.b_token(b_token),.b_data(b_data),
            .c_valid(c_valid),.c_pixel_addr(c_pixel_addr),.c_token(c_token),.c_data(c_data),.done(done));
    end else begin:GS
        spe_x_proj_stream_small #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(DT_MULTIPLIER),.DT_SHIFT(DT_SHIFT),.B_MULTIPLIER(B_MULTIPLIER),.B_SHIFT(B_SHIFT),.C_MULTIPLIER(C_MULTIPLIER),.C_SHIFT(C_SHIFT))u(
            .clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(in_valid),.in_pixel_addr(in_pixel_addr),.in_token(in_token),.in_channel_base(in_channel_base),.in_data(in_data),
            .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),.cfg_weight_data(cfg_weight_data),.cfg_dt_multiplier(cfg_dt_multiplier),.cfg_dt_shift(cfg_dt_shift),
            .cfg_b_multiplier(cfg_b_multiplier),.cfg_b_shift(cfg_b_shift),.cfg_c_multiplier(cfg_c_multiplier),.cfg_c_shift(cfg_c_shift),
            .dt_valid(dt_valid),.dt_pixel_addr(dt_pixel_addr),.dt_token(dt_token),.dt_data(dt_data),.b_valid(b_valid),.b_pixel_addr(b_pixel_addr),.b_token(b_token),.b_data(b_data),
            .c_valid(c_valid),.c_pixel_addr(c_pixel_addr),.c_token(c_token),.c_data(c_data),.done(done));
    end endgenerate
endmodule

`default_nettype wire
