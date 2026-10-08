`timescale 1ns / 1ps
`default_nettype none

module x_proj_dot64x1_pipeline (
    input wire clk, input wire rst_n, input wire frame_start,
    input wire issue_valid,
    input wire [511:0] activation_vector,
    input wire [511:0] weight_vector,
    input wire [19:0] issue_metadata,
    output reg sum_valid,
    output reg signed [20:0] sum_value,
    output reg [19:0] sum_metadata
);
    wire signed [15:0] product [0:63];
    genvar feature;
    generate
        for (feature = 0; feature < 64; feature = feature + 1) begin : GEN_MULT
            mamba_mult_8x8 u_mult (
                .CLK(clk),
                .A(activation_vector[feature*8 +: 8]),
                .B(weight_vector[feature*8 +: 8]),
                .P(product[feature]));
        end
    endgenerate

    reg mv0, mv1, mv2;
    reg [19:0] mm0, mm1, mm2;
    reg signed [16:0] l1 [0:31];
    reg signed [17:0] l2 [0:15];
    reg signed [18:0] l3 [0:7];
    (* use_dsp = "no" *) reg signed [19:0] l4 [0:3];
    (* use_dsp = "no" *) reg signed [20:0] l5 [0:1];
    (* use_dsp = "no" *) reg signed [21:0] l6;
    reg v1, v2, v3, v4, v5, v6;
    reg [19:0] m1, m2, m3, m4, m5, m6;
    integer i;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            mv0 <= 1'b0; mv1 <= 1'b0; mv2 <= 1'b0;
            v1 <= 1'b0; v2 <= 1'b0; v3 <= 1'b0;
            v4 <= 1'b0; v5 <= 1'b0; v6 <= 1'b0;
            sum_valid <= 1'b0;
        end else begin
            mv0 <= issue_valid; mv1 <= mv0; mv2 <= mv1;
            v1 <= mv2; v2 <= v1; v3 <= v2;
            v4 <= v3; v5 <= v4; v6 <= v5;
            sum_valid <= v6;
        end

        // All reduction and metadata registers are payload.  Advance them on
        // every clock and let mv*/v* alone describe validity; this removes the
        // 2176/1152-fanout clock-enable networks previously driven by mv2/v1.
        mm0 <= issue_metadata; mm1 <= mm0; mm2 <= mm1;
        m1 <= mm2; m2 <= m1; m3 <= m2;
        m4 <= m3; m5 <= m4; m6 <= m5;
        for (i=0; i<32; i=i+1)
            l1[i] <= $signed({product[i*2][15],product[i*2]})
                   + $signed({product[i*2+1][15],product[i*2+1]});
        for (i=0; i<16; i=i+1)
            l2[i] <= $signed({l1[i*2][16],l1[i*2]})
                   + $signed({l1[i*2+1][16],l1[i*2+1]});
        for (i=0; i<8; i=i+1)
            l3[i] <= $signed({l2[i*2][17],l2[i*2]})
                   + $signed({l2[i*2+1][17],l2[i*2+1]});
        for (i=0; i<4; i=i+1)
            l4[i] <= $signed({l3[i*2][18],l3[i*2]})
                   + $signed({l3[i*2+1][18],l3[i*2+1]});
        for (i=0; i<2; i=i+1)
            l5[i] <= $signed({l4[i*2][19],l4[i*2]})
                   + $signed({l4[i*2+1][19],l4[i*2+1]});
        l6 <= $signed({l5[0][20],l5[0]})
            + $signed({l5[1][20],l5[1]});
        sum_value <= l6[20:0];
        sum_metadata <= m6;
    end
endmodule

module x_proj_dot16x4_pipeline (
    input wire clk, input wire rst_n, input wire frame_start,
    input wire issue_valid,
    input wire [127:0] activation_vector,
    input wire [511:0] weight_vectors,
    input wire [19:0] issue_metadata,
    output reg sum_valid,
    output wire [83:0] sum_values,
    output reg [19:0] sum_metadata
);
    wire signed [15:0] product [0:63];
    reg signed [16:0] l1 [0:31];
    reg signed [17:0] l2 [0:15];
    (* use_dsp = "no" *) reg signed [18:0] l3 [0:7];
    (* use_dsp = "no" *) reg signed [19:0] l4 [0:3];
    reg signed [20:0] result [0:3];
    reg mv0,mv1,mv2,v1,v2,v3,v4;
    reg [19:0] mm0,mm1,mm2,m1,m2,m3,m4;
    integer i;
    genvar lane;
    genvar feature;
    generate
        for (lane=0; lane<4; lane=lane+1) begin : GEN_RESULT
            assign sum_values[lane*21 +: 21] = result[lane];
        end
        for (lane=0; lane<2; lane=lane+1) begin : GEN_LANE_PAIR
            for (feature=0; feature<16; feature=feature+1) begin : GEN_FEATURE
                mamba_packed_signed_mult_2x8_3cyc u_mult (
                    .clk(clk),.activation(activation_vector[feature*8 +: 8]),
                    .weight_low(weight_vectors[((lane*2)*16+feature)*8 +: 8]),
                    .weight_high(weight_vectors[((lane*2+1)*16+feature)*8 +: 8]),
                    .product_low(product[(lane*2)*16+feature]),
                    .product_high(product[(lane*2+1)*16+feature]));
            end
        end
    endgenerate
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            mv0<=0; mv1<=0; mv2<=0; v1<=0; v2<=0; v3<=0; v4<=0;
            sum_valid<=0;
        end else begin
            mv0<=issue_valid; mv1<=mv0; mv2<=mv1;
            v1<=mv2; v2<=v1; v3<=v2; v4<=v3; sum_valid<=v4;
        end
        mm0<=issue_metadata; mm1<=mm0; mm2<=mm1;
        m1<=mm2; m2<=m1; m3<=m2; m4<=m3;
        for(i=0;i<32;i=i+1)
            l1[i] <= $signed({product[i*2][15],product[i*2]})
                   + $signed({product[i*2+1][15],product[i*2+1]});
        for(i=0;i<16;i=i+1)
            l2[i] <= $signed({l1[i*2][16],l1[i*2]})
                   + $signed({l1[i*2+1][16],l1[i*2+1]});
        for(i=0;i<8;i=i+1)
            l3[i] <= $signed({l2[i*2][17],l2[i*2]})
                   + $signed({l2[i*2+1][17],l2[i*2+1]});
        for(i=0;i<4;i=i+1)
            l4[i] <= $signed({l3[i*2][18],l3[i*2]})
                   + $signed({l3[i*2+1][18],l3[i*2+1]});
        for(i=0;i<4;i=i+1) result[i] <= {l4[i][19],l4[i]};
        sum_metadata<=m4;
    end
endmodule

module spa_x_proj_stream_block0 #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] DT_MULTIPLIER=16'sd31024,
    parameter signed [6:0] DT_SHIFT=7'sd19,
    parameter signed [15:0] B_MULTIPLIER=16'sd17810,
    parameter signed [6:0] B_SHIFT=7'sd21,
    parameter signed [15:0] C_MULTIPLIER=16'sd16685,
    parameter signed [6:0] C_SHIFT=7'sd21
) (
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire [7:0] in_pixel_addr,
    input wire [5:0] in_channel_base,input wire [63:0] in_data,
    input wire cfg_weight_we,input wire [11:0] cfg_weight_addr,
    input wire signed [7:0] cfg_weight_data,
    input wire signed [31:0] cfg_dt_multiplier,input wire signed [6:0] cfg_dt_shift,
    input wire signed [31:0] cfg_b_multiplier,input wire signed [6:0] cfg_b_shift,
    input wire signed [31:0] cfg_c_multiplier,input wire signed [6:0] cfg_c_shift,
    output reg dt_valid,output reg [7:0] dt_pixel_addr,output reg [17:0] dt_data,
    output reg b_valid,output reg [7:0] b_pixel_addr,output reg [127:0] b_data,
    output reg c_valid,output reg [7:0] c_pixel_addr,output reg [127:0] c_data,
    output reg done
);
    reg [447:0] collect_input;
    wire wr_en=in_valid&&(in_channel_base==6'd56);
    wire [511:0] wr_data={in_data,collect_input};
    reg rd_en; reg [7:0] rd_addr; wire [511:0] rd_data;
    spa_x_input_bram_256x512 u_input_bram(
        .clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_pixel_addr),.dina(wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    reg [8:0] ready_count;
    always @(posedge clk) begin
        if(!rst_n) begin collect_input<=0;ready_count<=0;end
        else if(frame_start) begin ready_count<=0;end
        else if(in_valid) begin
            if(in_channel_base!=6'd56) collect_input[in_channel_base*8 +:64]<=in_data;
            if(in_channel_base==6'd56) ready_count<=ready_count+1'b1;
        end
    end
    localparam [1:0] W=0,R=1,I=2;
    reg [1:0] state; reg [8:0] time_index; reg [5:0] row_base;
    wire request=(state==I);
    always @(posedge clk) begin
        if(!rst_n||frame_start) begin state<=W;time_index<=0;row_base<=0;rd_en<=0;rd_addr<=0;end
        else begin
            rd_en<=0;
            case(state)
                W: if(time_index<ready_count) begin rd_en<=1;rd_addr<=time_index[7:0];state<=R;end
                R: begin row_base<=0;state<=I;end
                I: if(row_base==6'd32) begin row_base<=0;time_index<=time_index+1'b1;state<=W;end
                   else row_base<=row_base+3'd4;
                default: state<=W;
            endcase
        end
    end
    wire [511:0] w0,w1,w2,w3;
    mamba_spa_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID)) u_weight_rom(
        .clk(clk),.en(request),.addr0(row_base),.addr1(row_base+1'b1),
        .addr2(row_base+2'd2),.addr3(row_base+2'd3),
        .data0(w0),.data1(w1),.data2(w2),.data3(w3));
    reg rom_valid;reg [511:0]rom_act;reg[19:0]rom_meta;
    always @(posedge clk) begin
        if(!rst_n||frame_start)rom_valid<=0;
        else rom_valid<=request;
        // Payload is free-running; rom_valid is its only validity contract.
        rom_act<=rd_data;
        rom_meta<={5'd0,time_index,row_base};
    end
    wire dv01,dv23;wire signed[20:0]ds0,ds1,ds2,ds3;
    wire[19:0]dm01,dm23;
    mamba_dot64_pair_pipeline#(20) u_dot01(.clk(clk),.rst_n(rst_n),
        .frame_start(frame_start),.in_valid(rom_valid),.activation(rom_act),
        .weight_low(w0),.weight_high(w1),.in_meta(rom_meta),.out_valid(dv01),
        .out_sum_low(ds0),.out_sum_high(ds1),.out_meta(dm01));
    mamba_dot64_pair_pipeline#(20) u_dot23(.clk(clk),.rst_n(rst_n),
        .frame_start(frame_start),.in_valid(rom_valid),.activation(rom_act),
        .weight_low(w2),.weight_high(w3),.in_meta(rom_meta),.out_valid(dv23),
        .out_sum_low(ds2),.out_sum_high(ds3),.out_meta(dm23));
    wire dot_valid=dv01;wire[19:0]dot_meta=dm01;
    wire signed[20:0]dot_sums[0:3];
    assign dot_sums[0]=ds0;assign dot_sums[1]=ds1;
    assign dot_sums[2]=ds2;assign dot_sums[3]=ds3;
    wire signed[36:0]rq_p[0:3];genvar ql;
    generate for(ql=0;ql<4;ql=ql+1)begin:GEN_RQ_HP
        wire[6:0]sr=dot_meta[5:0]+ql;
        wire signed[15:0]sm=(sr<2)?DT_MULTIPLIER:(sr<18)?B_MULTIPLIER:C_MULTIPLIER;
        requant_mult_21x16 u_rq(.CLK(clk),.A(dot_sums[ql]),.B(sm),.P(rq_p[ql]));
    end endgenerate
    reg rv0,rv1,rv2;reg[19:0]rm0,rm1,rm2;
    always @(posedge clk)begin
        if(!rst_n||frame_start)begin rv0<=0;rv1<=0;rv2<=0;end
        else begin rv0<=dot_valid;rv1<=rv0;rv2<=rv1;end
        rm0<=dot_meta;rm1<=rm0;rm2<=rm1;
    end
    function[8:0]q9;input signed[36:0]p;input signed[6:0]s;reg signed[63:0]v,m,r;begin v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>255)q9=9'h0ff;else if(r< -256)q9=9'h100;else q9=r[8:0];end endfunction
    function[7:0]q8;input signed[36:0]p;input signed[6:0]s;reg signed[63:0]v,m,r;begin v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    wire[5:0]rrb=rm2[5:0];wire[7:0]rp=rm2[13:6];integer ol;integer orow;
    always @(posedge clk)begin
        if(!rst_n)begin dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;dt_data<=0;b_data<=0;c_data<=0;end
        else if(frame_start)begin dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;end
        else begin dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;if(rv2)for(ol=0;ol<4;ol=ol+1)begin
            orow=rrb+ol;
            if(orow<2)begin dt_data[orow*9 +:9]<=q9(rq_p[ol],DT_SHIFT);if(orow==1)begin dt_valid<=1;dt_pixel_addr<=rp;end end
            else if(orow<18)begin b_data[(orow-2)*8 +:8]<=q8(rq_p[ol],B_SHIFT);if(orow==17)begin b_valid<=1;b_pixel_addr<=rp;end end
            else if(orow<34)begin c_data[(orow-18)*8 +:8]<=q8(rq_p[ol],C_SHIFT);if(orow==33)begin c_valid<=1;c_pixel_addr<=rp;if(rp==PIXEL_COUNT-1)done<=1;end end
        end end
    end
endmodule

module spe_x_proj_stream_block0 #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] DT_MULTIPLIER=16'sd17095,
    parameter signed [6:0] DT_SHIFT=7'sd16,
    parameter signed [15:0] B_MULTIPLIER=16'sd16866,
    parameter signed [6:0] B_SHIFT=7'sd19,
    parameter signed [15:0] C_MULTIPLIER=16'sd31499,
    parameter signed [6:0] C_SHIFT=7'sd20
) (
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[7:0]in_pixel_addr,input wire[1:0]in_token,input wire[3:0]in_channel_base,input wire[63:0]in_data,
    input wire cfg_weight_we,input wire[9:0]cfg_weight_addr,input wire signed[7:0]cfg_weight_data,
    input wire signed[31:0]cfg_dt_multiplier,input wire signed[6:0]cfg_dt_shift,
    input wire signed[31:0]cfg_b_multiplier,input wire signed[6:0]cfg_b_shift,
    input wire signed[31:0]cfg_c_multiplier,input wire signed[6:0]cfg_c_shift,
    output reg dt_valid,output reg[7:0]dt_pixel_addr,output reg[1:0]dt_token,output reg[8:0]dt_data,
    output reg b_valid,output reg[7:0]b_pixel_addr,output reg[1:0]b_token,output reg[127:0]b_data,
    output reg c_valid,output reg[7:0]c_pixel_addr,output reg[1:0]c_token,output reg[127:0]c_data,output reg done
);
    reg[63:0]collect_low;wire[9:0]in_time={in_pixel_addr,in_token};wire wr_en=in_valid&&(in_channel_base==4'd8);
    wire[127:0]wr_data={in_data,collect_low};reg rd_en;reg[9:0]rd_addr;wire[127:0]rd_data;
    spe_x_input_bram_1024x128 u_input_bram(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(in_time),.dina(wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(rd_data));
    // Absolute producer/consumer sequences make the 1024-record BRAM a
    // two-tile-safe circular store.  Only the low address bits reach BRAM;
    // the upper bits prevent a local EOF from looking like an empty queue.
    // Absolute record counters must survive the 64-tile / 65536 boundary.
    reg[31:0]ready_count;
    always @(posedge clk)begin if(!rst_n)begin collect_low<=0;ready_count<=0;end
    else if(frame_start)begin ready_count<=0;end else if(in_valid)begin
        if(in_channel_base==0)collect_low<=in_data;if(in_channel_base==8)ready_count<=ready_count+1'b1;end end
    localparam[1:0]W=0,R=1,I=2;reg[1:0]state;reg[31:0]time_index;reg[5:0]row_base;wire request=state==I;
    // The input BRAM has one registered read cycle.  A request made while the
    // last row group (32) is being issued is therefore too late: the following
    // row-0 capture would still see the previous record.  Start the read while
    // row group 16 is issued and remember whether the next record was actually
    // prefetched.  This keeps 0/16/32 -> 0/16/32 continuous without mixing two
    // adjacent activation records.
    reg next_prefetched;
    wire [31:0] next_record_sequence = time_index + 32'd1;
    wire next_record_ready=(ready_count > next_record_sequence);
    always @(posedge clk)begin if(!rst_n||frame_start)begin state<=W;time_index<=0;row_base<=0;rd_en<=0;rd_addr<=0;next_prefetched<=0;end else begin rd_en<=0;case(state)
        W:if(time_index<ready_count)begin rd_en<=1;rd_addr<=time_index[9:0];state<=R;next_prefetched<=0;end
        R:begin row_base<=0;state<=I;end
        I:begin
            if((row_base==16)&&next_record_ready)begin
                rd_en<=1;
                rd_addr<=time_index[9:0]+1'b1;
                next_prefetched<=1;
            end
            if(row_base==32)begin
                row_base<=0;
                time_index<=time_index+1'b1;
                if(next_prefetched)begin state<=I;next_prefetched<=0;end
                else begin state<=W;next_prefetched<=0;end
            end else row_base<=row_base+5'd16;
        end
        default:state<=W;endcase end end
    wire[127:0]w0,w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15;
    mamba_spe_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID))u_weight_rom0(.clk(clk),.en(request),
        .addr0(row_base),.addr1(row_base+1'b1),.addr2(row_base+2'd2),.addr3(row_base+2'd3),
        .data0(w0),.data1(w1),.data2(w2),.data3(w3));
    mamba_spe_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID))u_weight_rom1(.clk(clk),.en(request),
        .addr0(row_base+3'd4),.addr1(row_base+3'd5),.addr2(row_base+3'd6),.addr3(row_base+3'd7),
        .data0(w4),.data1(w5),.data2(w6),.data3(w7));
    mamba_spe_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID))u_weight_rom2(.clk(clk),.en(request),
        .addr0(row_base+4'd8),.addr1(row_base+4'd9),.addr2(row_base+4'd10),.addr3(row_base+4'd11),
        .data0(w8),.data1(w9),.data2(w10),.data3(w11));
    mamba_spe_x_weight_rom_4r #(.BLOCK_ID(BLOCK_ID))u_weight_rom3(.clk(clk),.en(request),
        .addr0(row_base+4'd12),.addr1(row_base+4'd13),.addr2(row_base+4'd14),.addr3(row_base+4'd15),
        .data0(w12),.data1(w13),.data2(w14),.data3(w15));
    reg rom_valid;reg[127:0]rom_act;reg[19:0]rom_meta;
    always @(posedge clk)begin
        if(!rst_n||frame_start)rom_valid<=0;else rom_valid<=request;
        // Do not infer request-driven CEs on activation/metadata payload.
        rom_act<=rd_data;rom_meta<={3'd0,time_index[10:0],row_base};
    end
    wire dv0,dv1,dv2,dv3;wire[83:0]ds0,ds1,ds2,ds3;wire[19:0]dm0,dm1,dm2,dm3;
    x_proj_dot16x4_pipeline u_dot0(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.issue_valid(rom_valid),
        .activation_vector(rom_act),.weight_vectors({w3,w2,w1,w0}),.issue_metadata(rom_meta),
        .sum_valid(dv0),.sum_values(ds0),.sum_metadata(dm0));
    x_proj_dot16x4_pipeline u_dot1(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.issue_valid(rom_valid),
        .activation_vector(rom_act),.weight_vectors({w7,w6,w5,w4}),.issue_metadata(rom_meta),
        .sum_valid(dv1),.sum_values(ds1),.sum_metadata(dm1));
    x_proj_dot16x4_pipeline u_dot2(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.issue_valid(rom_valid),
        .activation_vector(rom_act),.weight_vectors({w11,w10,w9,w8}),.issue_metadata(rom_meta),
        .sum_valid(dv2),.sum_values(ds2),.sum_metadata(dm2));
    x_proj_dot16x4_pipeline u_dot3(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.issue_valid(rom_valid),
        .activation_vector(rom_act),.weight_vectors({w15,w14,w13,w12}),.issue_metadata(rom_meta),
        .sum_valid(dv3),.sum_values(ds3),.sum_metadata(dm3));
    wire dot_valid=dv0;wire[19:0]dot_meta=dm0;wire[335:0]dot_sums={ds3,ds2,ds1,ds0};
    wire[5:0]drb=dot_meta[5:0];wire signed[36:0]rqp[0:15];genvar ql;
    generate for(ql=0;ql<16;ql=ql+1)begin:GEN_RQ
        wire[6:0]sr=drb+ql;wire signed[15:0]sm=(sr==0)?DT_MULTIPLIER:(sr<17)?B_MULTIPLIER:C_MULTIPLIER;
        requant_mult_21x16 u_rq(.CLK(clk),.A(dot_sums[ql*21 +:21]),.B(sm),.P(rqp[ql]));end endgenerate
    reg rv0,rv1,rv2;reg[19:0]rm0,rm1,rm2;
    always @(posedge clk)begin
        if(!rst_n||frame_start)begin rv0<=0;rv1<=0;rv2<=0;end
        else begin rv0<=dot_valid;rv1<=rv0;rv2<=rv1;end
        rm0<=dot_meta;rm1<=rm0;rm2<=rm1;
    end
    function[8:0]q9;input signed[36:0]p;input signed[6:0]s;reg signed[63:0]v,m,r;begin v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>255)q9=9'h0ff;else if(r< -256)q9=9'h100;else q9=r[8:0];end endfunction
    function[7:0]q8;input signed[36:0]p;input signed[6:0]s;reg signed[63:0]v,m,r;begin v={{27{p[36]}},p};m=v<0?-v:v;
        if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    wire[10:0]rt=rm2[16:6];wire[5:0]rrb=rm2[5:0];wire[7:0]rp=rt[9:2];wire[1:0]rto=rt[1:0];integer ol;integer orow;
    always @(posedge clk)begin if(!rst_n)begin dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;dt_data<=0;b_data<=0;c_data<=0;end
    else if(frame_start)begin dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;end else begin
        dt_valid<=0;b_valid<=0;c_valid<=0;done<=0;if(rv2)for(ol=0;ol<16;ol=ol+1)begin orow=rrb+ol;
            if(orow==0)begin dt_data<=q9(rqp[ol],DT_SHIFT);dt_pixel_addr<=rp;dt_token<=rto;dt_valid<=1;end
            else if(orow<17)begin b_data[(orow-1)*8 +:8]<=q8(rqp[ol],B_SHIFT);if(orow==16)begin b_valid<=1;b_pixel_addr<=rp;b_token<=rto;end end
            else if(orow<33)begin c_data[(orow-17)*8 +:8]<=q8(rqp[ol],C_SHIFT);if(orow==32)begin c_valid<=1;c_pixel_addr<=rp;c_token<=rto;if(rt==(PIXEL_COUNT*4)-1)done<=1;end end
        end end end
endmodule

`default_nettype wire
