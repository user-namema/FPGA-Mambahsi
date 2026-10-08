`timescale 1ns/1ps
`default_nettype none

module spa_out_proj_rom_stream_small #(
    parameter integer BLOCK_ID=1,parameter integer PIXEL_COUNT=(BLOCK_ID==1)?64:16,
    parameter signed[15:0]FIXED_MULTIPLIER=16'sd18380,parameter signed[6:0]FIXED_SHIFT=7'sd19
)(input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,input wire[9:0]in_record_addr,
input wire[5:0]in_channel_base,input wire[31:0]in_data,input wire cfg_weight_we,input wire[10:0]cfg_weight_addr,
input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,input wire[4:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,
input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,output reg out_valid,output reg[7:0]out_pixel_addr,
output reg[4:0]out_channel_base,output reg[31:0]out_raw_data,output reg[31:0]out_data,output reg done);
    // Two output rows share sixteen packed DSPs. Four 16-input chunks keep
    // the same row service rate while halving the multiplier bank.
    localparam integer PAR=16;localparam integer CHUNKS=4;
    localparam integer CONTINUOUS_PREFETCH=(BLOCK_ID==1);
    localparam integer PIXEL_INDEX_BITS=$clog2(PIXEL_COUNT);
    wire unused_cfg=cfg_weight_we^cfg_weight_addr[0]^cfg_weight_data[0]^cfg_bias_we^cfg_bias_addr[0]^cfg_bias_data[0]^cfg_multiplier[0]^cfg_shift[0];
    reg[479:0]collect;wire wr_en=in_valid&&(in_channel_base==60);wire[511:0]wr_data={in_data,collect};
    reg rd_en;reg[7:0]rd_addr;wire[511:0]rd_data;reg[31:0]ready_count;
    spa_x_input_bram_256x512 u_ram(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_record_addr[7:0]),.dina(wr_data),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    always@(posedge clk)begin if(!rst_n)begin collect<=0;ready_count<=0;end
    else begin
        if(frame_start)ready_count<=0;
        else if(in_valid&&in_channel_base==60)ready_count<=ready_count+1'b1;
        if(in_valid&&in_channel_base!=60)collect[in_channel_base*8+:32]<=in_data;
    end end
    localparam[2:0]W=0,R=1,C=2,I=3;reg[2:0]state;reg[31:0]record_q;reg[4:0]row;reg[4:0]chunk;reg[511:0]activation;
    wire[7:0]local_pixel=
        {{(8-PIXEL_INDEX_BITS){1'b0}},record_q[PIXEL_INDEX_BITS-1:0]};
    reg next_prefetch_started;
    // Block1 Spa SSM and this out_proj both have an exact 64-cycle/record
    // service time.  Waiting until ready_count changes and then taking the
    // W/R/C BRAM path inserts five dead clocks per record (64*5=320 clocks per
    // tile).  Capture the producer's final 32-bit beat together with the
    // already collected 480 bits, and carry that complete next record across
    // the boundary.  BRAM remains the lossless fallback when the producer is
    // either early by more than one record or late.
    reg [511:0] next_activation;
    reg next_activation_valid;
    wire [31:0] next_record_sequence = record_q + 32'd1;
    wire next_record_ready=(ready_count > next_record_sequence);
    wire current_commit_now=wr_en&&(ready_count==record_q);
    wire next_commit_now=wr_en&&(ready_count == next_record_sequence);
    wire request=state==I;
    always@(posedge clk)begin if(!rst_n)begin state<=W;record_q<=0;row<=0;chunk<=0;activation<=0;rd_en<=0;rd_addr<=0;next_prefetch_started<=0;next_activation<=0;next_activation_valid<=0;end
    else if(frame_start)begin state<=W;record_q<=0;row<=0;chunk<=0;rd_en<=0;rd_addr<=0;next_prefetch_started<=0;next_activation_valid<=0;end else begin
        rd_en<=0;
        // A record which completes while its predecessor is being evaluated
        // is already available as wr_data on this edge.  Saving it locally
        // removes a same-address BRAM read/write dependency and is bit-exact.
        if((state==I)&&next_commit_now)begin
            next_activation<=wr_data;
            next_activation_valid<=1'b1;
        end
        case(state)
        W:begin row<=0;chunk<=0;next_prefetch_started<=0;
            if(current_commit_now)begin
                activation<=wr_data;
                next_activation_valid<=1'b0;
                state<=I;
            end else if(record_q<ready_count)begin rd_en<=1;rd_addr<=local_pixel;state<=R;end end
        R:state<=C;C:begin activation<=rd_data;next_prefetch_started<=0;state<=I;end
        I:begin
            if(CONTINUOUS_PREFETCH&&(row==28)&&(chunk==CHUNKS-1)
                &&next_record_ready)begin
                rd_en<=1;rd_addr<=(local_pixel==PIXEL_COUNT-1)?8'd0:local_pixel+1'b1;
                next_prefetch_started<=1;
            end
            if(chunk==CHUNKS-1)begin chunk<=0;if(row==30)begin
                row<=0;record_q<=record_q+1'b1;
                if(next_commit_now)begin
                    activation<=wr_data;
                    next_activation_valid<=1'b0;
                    next_prefetch_started<=1'b0;
                    state<=I;
                end else if(next_activation_valid)begin
                    activation<=next_activation;
                    next_activation_valid<=1'b0;
                    next_prefetch_started<=1'b0;
                    state<=I;
                end else if(next_prefetch_started)begin activation<=rd_data;next_prefetch_started<=0;state<=I;end
                else state<=W;
            end else row<=row+2'd2;end else chunk<=chunk+1'b1;
        end
        default:state<=W;endcase end end

`ifndef SYNTHESIS
    integer sim_continuous_record_transitions;
    integer sim_direct_record_transitions;
    integer sim_fallback_record_transitions;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin
            sim_continuous_record_transitions<=0;
            sim_direct_record_transitions<=0;
            sim_fallback_record_transitions<=0;
        end else if((state==I)&&(row==30)&&(chunk==CHUNKS-1)
                    )begin
            if(next_commit_now||next_activation_valid)begin
                sim_continuous_record_transitions<=sim_continuous_record_transitions+1;
                sim_direct_record_transitions<=sim_direct_record_transitions+1;
            end else if(next_prefetch_started)
                sim_continuous_record_transitions<=sim_continuous_record_transitions+1;
            else
                sim_fallback_record_transitions<=sim_fallback_record_transitions+1;
        end
    end
`endif
    wire[543:0]params_low,params_high,params_unused2,params_unused3;
    mamba_spa_out_param_rom_4r#(.BLOCK_ID(BLOCK_ID))u_rom(.clk(clk),.en(request),
        .addr0(row),.addr1(row+1'b1),.addr2(5'd0),.addr3(5'd0),
        .data0(params_low),.data1(params_high),.data2(params_unused2),.data3(params_unused3));
    reg rv;reg[127:0]ra;reg[17:0]rm_base;
    wire[127:0]rw_low=params_low[rm_base[4:0]*128+:128];
    wire[127:0]rw_high=params_high[rm_base[4:0]*128+:128];
    wire[81:0]rm={params_high[543:512],params_low[543:512],rm_base};
    always@(posedge clk)begin if(!rst_n)begin rv<=0;end
    else begin if(frame_start)rv<=0;else rv<=request;if(request)begin
        ra<=activation[chunk*128+:128];rm_base<={local_pixel,row,chunk};end end end
    wire dv;wire signed[19:0]part_low,part_high;wire[81:0]dm;
    mamba_dot16_pair_pipeline#(82)u_dot(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),
        .in_valid(rv),.activation(ra),.weight_low(rw_low),.weight_high(rw_high),
        .in_meta(rm),.out_valid(dv),.out_sum_low(part_low),
        .out_sum_high(part_high),.out_meta(dm));
    wire first=dm[4:0]==0,last=dm[4:0]==CHUNKS-1;
    (* use_dsp = "no" *) reg signed[47:0]acc[0:1];
    reg acc_result_valid;reg[76:0]acc_result_meta;
    (* use_dsp = "no" *) reg signed[47:0]sum_bias_q[0:1];
    reg bias_result_valid;reg[12:0]bias_result_meta;integer pair_lane;
    always@(posedge clk)begin
        if(!rst_n)begin acc[0]<=0;acc[1]<=0;acc_result_valid<=0;acc_result_meta<=0;
            sum_bias_q[0]<=0;sum_bias_q[1]<=0;bias_result_valid<=0;bias_result_meta<=0;end
        else begin
            if(frame_start)begin acc_result_valid<=0;bias_result_valid<=0;end
            else begin acc_result_valid<=0;bias_result_valid<=acc_result_valid;end
            if(dv)begin
                for(pair_lane=0;pair_lane<2;pair_lane=pair_lane+1)begin
                    if(first)acc[pair_lane]<=pair_lane==0?{{28{part_low[19]}},part_low}:{{28{part_high[19]}},part_high};
                    else acc[pair_lane]<=acc[pair_lane]+(pair_lane==0?{{28{part_low[19]}},part_low}:{{28{part_high[19]}},part_high});
                end
                if(last)begin if(!frame_start)acc_result_valid<=1;acc_result_meta<=dm[81:5];end
            end
            if(acc_result_valid)begin
                sum_bias_q[0]<=acc[0]+{{27{acc_result_meta[33]}},acc_result_meta[33:13]};
                sum_bias_q[1]<=acc[1]+{{27{acc_result_meta[65]}},acc_result_meta[65:45]};
                bias_result_meta<=acc_result_meta[12:0];
            end
        end
    end
    wire signed[36:0]rq[0:1];genvar rq_lane;
    generate for(rq_lane=0;rq_lane<2;rq_lane=rq_lane+1)begin:G_RQ
        requant_mult_21x16 u_rq(.CLK(clk),.A(sum_bias_q[rq_lane][20:0]),.B(FIXED_MULTIPLIER),.P(rq[rq_lane]));
    end endgenerate
    reg qv0,qv1,qv2;reg[12:0]qm0,qm1,qm2;always@(posedge clk)begin if(!rst_n||frame_start)begin qv0<=0;qv1<=0;qv2<=0;end else begin qv0<=bias_result_valid;qv1<=qv0;qv2<=qv1;end qm0<=bias_result_meta;qm1<=qm0;qm2<=qm1;end
    function[7:0]q8;input signed[36:0]v;reg signed[63:0]a,r;begin a=v<0?-$signed({{27{v[36]}},v}):$signed({{27{v[36]}},v});r=(a+(64'sd1<<<(FIXED_SHIFT-1)))>>>FIXED_SHIFT;if(v<0)r=-r;if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    reg[15:0]pack_raw,pack_relu;wire[7:0]raw_low=q8(rq[0]),raw_high=q8(rq[1]);
    wire[7:0]relu_low=raw_low[7]?0:raw_low,relu_high=raw_high[7]?0:raw_high;
    always@(posedge clk)begin if(!rst_n)begin out_valid<=0;done<=0;end
    else begin
        if(frame_start)begin out_valid<=0;done<=0;out_pixel_addr<=0;out_channel_base<=0;end
        else begin out_valid<=0;done<=0;end
        if(qv2)begin
        if(qm2[1:0]==2)begin if(!frame_start)begin out_valid<=1;out_pixel_addr<=qm2[12:5];out_channel_base<={qm2[4:2],2'b00};end
            out_raw_data<={raw_high,raw_low,pack_raw};out_data<={relu_high,relu_low,pack_relu};
            if(!frame_start&&qm2[12:5]==PIXEL_COUNT-1&&qm2[4:0]==30)done<=1;end
        else begin pack_raw<={raw_high,raw_low};pack_relu<={relu_high,relu_low};end end end end
endmodule

module spe_out_proj_rom_stream_small #(
    parameter integer BLOCK_ID=1,parameter integer PIXEL_COUNT=(BLOCK_ID==1)?64:16,
    parameter signed[15:0]FIXED_MULTIPLIER=16'sd24384,parameter signed[6:0]FIXED_SHIFT=7'sd17
)(input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,input wire[9:0]in_record_addr,input wire[5:0]in_channel_base,
input wire[31:0]in_data,input wire cfg_weight_we,input wire[6:0]cfg_weight_addr,input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,
input wire[2:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,
output reg out_valid,output reg[7:0]out_pixel_addr,output reg[1:0]out_token,output reg[63:0]out_raw_data,output reg[63:0]out_data,output reg done);
    // Two output rows share the packed bank. Block1 uses 8 features/cycle and
    // Block2 uses 4, preserving their former service rates with half the DSPs.
    localparam integer PAR=(BLOCK_ID==1)?8:4;
    localparam integer CHUNKS=16/PAR;
    localparam integer RECORD_COUNT=PIXEL_COUNT*4;
    localparam integer RECORD_INDEX_BITS=$clog2(RECORD_COUNT);
    wire unused_cfg=cfg_weight_we^cfg_weight_addr[0]^cfg_weight_data[0]^cfg_bias_we^cfg_bias_addr[0]^cfg_bias_data[0]^cfg_multiplier[0]^cfg_shift[0];
    reg[95:0]collect;wire wr_en=in_valid&&(in_channel_base==12);wire[127:0]wr_data={in_data,collect};reg rd_en;reg[9:0]rd_addr;wire[127:0]rd_data;reg[31:0]ready_count;
    spe_x_input_bram_1024x128 u_ram(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_record_addr),.dina(wr_data),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    always@(posedge clk)begin if(!rst_n)begin collect<=0;ready_count<=0;end
    else begin
        if(frame_start)ready_count<=0;
        else if(in_valid&&in_channel_base==12)ready_count<=ready_count+1'b1;
        if(in_valid&&in_channel_base!=12)collect[in_channel_base*8+:32]<=in_data;
    end end
    localparam[2:0]W=0,R=1,C=2,I=3;reg[2:0]state;reg[31:0]record_q;reg[2:0]row;reg[3:0]chunk;reg[127:0]activation;wire request=state==I;
    wire[9:0]local_record=
        {{(10-RECORD_INDEX_BITS){1'b0}},record_q[RECORD_INDEX_BITS-1:0]};
    reg next_prefetch_started;
    wire [31:0] next_record_sequence = record_q + 32'd1;
    wire next_record_ready=(ready_count > next_record_sequence);
    always@(posedge clk)begin if(!rst_n)begin state<=W;record_q<=0;row<=0;chunk<=0;activation<=0;rd_en<=0;rd_addr<=0;next_prefetch_started<=0;end
    else if(frame_start)begin state<=W;record_q<=0;row<=0;chunk<=0;rd_en<=0;rd_addr<=0;next_prefetch_started<=0;end else begin rd_en<=0;case(state)
        W:begin row<=0;chunk<=0;next_prefetch_started<=0;if(record_q<ready_count)begin rd_en<=1;rd_addr<=local_record;state<=R;end end
        R:state<=C;
        C:begin activation<=rd_data;next_prefetch_started<=0;state<=I;end
        I:begin
            if((row==4)&&(chunk==CHUNKS-1)
                &&next_record_ready)begin
                rd_en<=1;rd_addr<=(local_record==RECORD_COUNT-1)?10'd0:local_record+1'b1;
                next_prefetch_started<=1;
            end
            if(chunk==CHUNKS-1)begin
                chunk<=0;
                if(row==6)begin
                    row<=0;record_q<=record_q+1'b1;
                    if(next_prefetch_started)begin
                        activation<=rd_data;next_prefetch_started<=0;state<=I;
                    end else state<=W;
                end else row<=row+2'd2;
            end else chunk<=chunk+1'b1;
        end
        default:state<=W;
    endcase end end
`ifndef SYNTHESIS
    integer sim_continuous_record_transitions;
    integer sim_fallback_record_transitions;
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin
            sim_continuous_record_transitions<=0;
            sim_fallback_record_transitions<=0;
        end else if((state==I)&&(row==6)&&(chunk==CHUNKS-1)
            )begin
            if(next_prefetch_started)
                sim_continuous_record_transitions
                    <=sim_continuous_record_transitions+1;
            else sim_fallback_record_transitions
                    <=sim_fallback_record_transitions+1;
        end
    end
`endif
    wire[1279:0]params;mamba_spe_out_param_rom#(.BLOCK_ID(BLOCK_ID))u_rom(.clk(clk),.en(1'b1),.data(params));
    wire[127:0]row_weight_low=params[row*160+:128];
    wire[127:0]row_weight_high=params[(row+1'b1)*160+:128];
    wire[31:0]row_bias_low=params[row*160+128+:32];
    wire[31:0]row_bias_high=params[(row+1'b1)*160+128+:32];
    reg rv;reg[127:0]ra,rw_low,rw_high;reg[81:0]rm;
    always@(posedge clk)begin if(!rst_n)begin rv<=0;end
    else begin if(frame_start)rv<=0;else rv<=request;if(request)begin
        ra<=activation;rw_low<=row_weight_low;rw_high<=row_weight_high;
        rm<={row_bias_high,row_bias_low,1'b0,local_record,row,chunk};end end end
    wire dv;wire signed[20:0]part_low,part_high;wire[81:0]dm;
    generate
        if(BLOCK_ID==1)begin:G_DOT16_FAST
            wire signed[18:0]ds_low,ds_high;
            mamba_dot8_pair_pipeline#(82)u_dot(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),
                .in_valid(rv),.activation(ra[rm[3:0]*64+:64]),
                .weight_low(rw_low[rm[3:0]*64+:64]),.weight_high(rw_high[rm[3:0]*64+:64]),
                .in_meta(rm),.out_valid(dv),.out_sum_low(ds_low),.out_sum_high(ds_high),.out_meta(dm));
            assign part_low={{2{ds_low[18]}},ds_low};assign part_high={{2{ds_high[18]}},ds_high};
        end else begin:G_DOT8_SCALED
            wire signed[17:0]ds_low,ds_high;
            mamba_dot4_pair_pipeline#(82)u_dot(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),
                .in_valid(rv),.activation(ra[rm[3:0]*32+:32]),
                .weight_low(rw_low[rm[3:0]*32+:32]),.weight_high(rw_high[rm[3:0]*32+:32]),
                .in_meta(rm),.out_valid(dv),.out_sum_low(ds_low),.out_sum_high(ds_high),.out_meta(dm));
            assign part_low={{3{ds_low[17]}},ds_low};assign part_high={{3{ds_high[17]}},ds_high};
        end
    endgenerate
    wire first=dm[3:0]==0,last=dm[3:0]==CHUNKS-1;
    (* use_dsp = "no" *) reg signed[47:0]acc[0:1];
    reg acc_result_valid;reg[77:0]acc_result_meta;
    (* use_dsp = "no" *) reg signed[47:0]sum_bias_q[0:1];
    reg bias_result_valid;reg[13:0]bias_result_meta;integer spe_pair_lane;
    always@(posedge clk)begin if(!rst_n)begin acc[0]<=0;acc[1]<=0;acc_result_valid<=0;acc_result_meta<=0;sum_bias_q[0]<=0;sum_bias_q[1]<=0;bias_result_valid<=0;bias_result_meta<=0;end
    else begin
            if(frame_start)begin acc_result_valid<=0;bias_result_valid<=0;end
            else begin acc_result_valid<=0;bias_result_valid<=acc_result_valid;end
            if(dv)begin for(spe_pair_lane=0;spe_pair_lane<2;spe_pair_lane=spe_pair_lane+1)begin
                if(first)acc[spe_pair_lane]<=spe_pair_lane==0?{{27{part_low[20]}},part_low}:{{27{part_high[20]}},part_high};
                else acc[spe_pair_lane]<=acc[spe_pair_lane]+(spe_pair_lane==0?{{27{part_low[20]}},part_low}:{{27{part_high[20]}},part_high});end
                if(last)begin if(!frame_start)acc_result_valid<=1;acc_result_meta<=dm[81:4];end end
            if(acc_result_valid)begin sum_bias_q[0]<=acc[0]+{{27{acc_result_meta[34]}},acc_result_meta[34:14]};
                sum_bias_q[1]<=acc[1]+{{27{acc_result_meta[66]}},acc_result_meta[66:46]};
                bias_result_meta<=acc_result_meta[13:0];end end end
    wire signed[36:0]rq[0:1];genvar spe_rq_lane;
    generate for(spe_rq_lane=0;spe_rq_lane<2;spe_rq_lane=spe_rq_lane+1)begin:G_PAIR_RQ
        requant_mult_21x16 u_rq(.CLK(clk),.A(sum_bias_q[spe_rq_lane][20:0]),.B(FIXED_MULTIPLIER),.P(rq[spe_rq_lane]));
    end endgenerate
    reg qv0,qv1,qv2;reg[13:0]qm0,qm1,qm2;
    always@(posedge clk)begin if(!rst_n||frame_start)begin qv0<=0;qv1<=0;qv2<=0;end else begin qv0<=bias_result_valid;qv1<=qv0;qv2<=qv1;end qm0<=bias_result_meta;qm1<=qm0;qm2<=qm1;end
    function[7:0]q8;input signed[36:0]v;reg signed[63:0]a,r;begin a=v<0?-$signed({{27{v[36]}},v}):$signed({{27{v[36]}},v});r=(a+(64'sd1<<<(FIXED_SHIFT-1)))>>>FIXED_SHIFT;if(v<0)r=-r;if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    reg[47:0]pack_raw,pack_relu;wire[7:0]raw_low=q8(rq[0]),raw_high=q8(rq[1]);
    wire[7:0]relu_low=raw_low[7]?0:raw_low,relu_high=raw_high[7]?0:raw_high;
    wire[10:0]result_record=qm2[13:3];wire[2:0]result_row=qm2[2:0];
    always@(posedge clk)begin if(!rst_n)begin out_valid<=0;done<=0;end
    else begin
        if(frame_start)begin out_valid<=0;done<=0;out_pixel_addr<=0;out_token<=0;end
        else begin out_valid<=0;done<=0;end
        if(qv2)begin
        if(result_row==6)begin if(!frame_start)begin out_valid<=1;out_pixel_addr<=result_record[9:2];out_token<=result_record[1:0];end
            out_raw_data<={raw_high,raw_low,pack_raw};out_data<={relu_high,relu_low,pack_relu};
            if(!frame_start&&result_record==(PIXEL_COUNT*4)-1)done<=1;end
        else begin pack_raw[result_row*8+:16]<={raw_high,raw_low};pack_relu[result_row*8+:16]<={relu_high,relu_low};end end end end
endmodule

module spa_out_proj_rom_stream #(
    parameter integer BLOCK_ID=0,parameter integer PIXEL_COUNT=256,parameter signed[15:0]FIXED_MULTIPLIER=16'sd30313,parameter signed[6:0]FIXED_SHIFT=7'sd19
)(input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,input wire[9:0]in_record_addr,input wire[5:0]in_channel_base,input wire[31:0]in_data,
input wire cfg_weight_we,input wire[10:0]cfg_weight_addr,input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,input wire[4:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,
input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,output wire out_valid,output wire[7:0]out_pixel_addr,output wire[4:0]out_channel_base,output wire[31:0]out_raw_data,output wire[31:0]out_data,output wire done);
generate if(BLOCK_ID==0)begin:G0
block0_spa_out_proj_steam#(.BLOCK_ID(BLOCK_ID),
    .PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(FIXED_MULTIPLIER),
    .FIXED_SHIFT(FIXED_SHIFT))u(
    .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
    .in_valid(in_valid),.in_record_addr(in_record_addr),
    .in_channel_base(in_channel_base),.in_data(in_data),
    .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),
    .cfg_weight_data(cfg_weight_data),.cfg_bias_we(cfg_bias_we),
    .cfg_bias_addr(cfg_bias_addr),.cfg_bias_data(cfg_bias_data),
    .cfg_multiplier(cfg_multiplier),.cfg_shift(cfg_shift),
    .out_valid(out_valid),.out_pixel_addr(out_pixel_addr),
    .out_channel_base(out_channel_base),.out_raw_data(out_raw_data),
    .out_data(out_data),.done(done));
end else begin:GS
spa_out_proj_rom_stream_small#(.BLOCK_ID(BLOCK_ID),
    .PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(FIXED_MULTIPLIER),
    .FIXED_SHIFT(FIXED_SHIFT))u(
    .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
    .in_valid(in_valid),.in_record_addr(in_record_addr),
    .in_channel_base(in_channel_base),.in_data(in_data),
    .cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),
    .cfg_weight_data(cfg_weight_data),.cfg_bias_we(cfg_bias_we),
    .cfg_bias_addr(cfg_bias_addr),.cfg_bias_data(cfg_bias_data),
    .cfg_multiplier(cfg_multiplier),.cfg_shift(cfg_shift),
    .out_valid(out_valid),.out_pixel_addr(out_pixel_addr),
    .out_channel_base(out_channel_base),.out_raw_data(out_raw_data),
    .out_data(out_data),.done(done));
end endgenerate
endmodule

module spe_out_proj_rom_stream #(
    parameter integer BLOCK_ID=0,parameter integer PIXEL_COUNT=256,parameter signed[15:0]FIXED_MULTIPLIER=16'sd32455,parameter signed[6:0]FIXED_SHIFT=7'sd18
)(input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,input wire[9:0]in_record_addr,input wire[5:0]in_channel_base,input wire[31:0]in_data,
input wire cfg_weight_we,input wire[6:0]cfg_weight_addr,input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,input wire[2:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,
input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,output wire out_valid,output wire[7:0]out_pixel_addr,output wire[1:0]out_token,output wire[63:0]out_raw_data,output wire[63:0]out_data,output wire done);
generate if(BLOCK_ID==0)begin:G0 spe_out_proj_rom_stream_block0#(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(FIXED_MULTIPLIER),.FIXED_SHIFT(FIXED_SHIFT))u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(in_valid),.in_record_addr(in_record_addr),.in_channel_base(in_channel_base),.in_data(in_data),.cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),.cfg_weight_data(cfg_weight_data),.cfg_bias_we(cfg_bias_we),.cfg_bias_addr(cfg_bias_addr),.cfg_bias_data(cfg_bias_data),.cfg_multiplier(cfg_multiplier),.cfg_shift(cfg_shift),.out_valid(out_valid),.out_pixel_addr(out_pixel_addr),.out_token(out_token),.out_raw_data(out_raw_data),.out_data(out_data),.done(done));
end else begin:GS spe_out_proj_rom_stream_small#(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(FIXED_MULTIPLIER),.FIXED_SHIFT(FIXED_SHIFT))u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(in_valid),.in_record_addr(in_record_addr),.in_channel_base(in_channel_base),.in_data(in_data),.cfg_weight_we(cfg_weight_we),.cfg_weight_addr(cfg_weight_addr),.cfg_weight_data(cfg_weight_data),.cfg_bias_we(cfg_bias_we),.cfg_bias_addr(cfg_bias_addr),.cfg_bias_data(cfg_bias_data),.cfg_multiplier(cfg_multiplier),.cfg_shift(cfg_shift),.out_valid(out_valid),.out_pixel_addr(out_pixel_addr),.out_token(out_token),.out_raw_data(out_raw_data),.out_data(out_data),.done(done));end endgenerate
endmodule

`default_nettype wire
