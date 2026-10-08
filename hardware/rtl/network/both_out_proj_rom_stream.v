`timescale 1ns / 1ps
`default_nettype none

// Four parallel 64-element dots.  The bias travels with the ROM word and is
// added in a separate registered stage after the six-level binary tree.
module out_proj_dot64x4_bias_pipeline (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 issue_valid,
    input  wire [511:0]         activation_vector,
    input  wire [2047:0]        weight_vectors,
    input  wire [127:0]         bias_values,
    input  wire [19:0]          issue_metadata,
    output reg                  sum_valid,
    output reg signed [20:0]    sum_0,
    output reg signed [20:0]    sum_1,
    output reg signed [20:0]    sum_2,
    output reg signed [20:0]    sum_3,
    output reg [19:0]           sum_metadata
);
    wire signed [15:0] products [0:255];
    genvar output_lane;
    genvar input_lane;
    generate
        for (output_lane=0; output_lane<2; output_lane=output_lane+1) begin: G_O_PAIR
            for (input_lane=0; input_lane<64; input_lane=input_lane+1) begin: G_I
                mamba_packed_signed_mult_2x8_3cyc u_mult(
                    .clk(clk),.activation(activation_vector[input_lane*8 +: 8]),
                    .weight_low(weight_vectors[((output_lane*2)*64+input_lane)*8 +: 8]),
                    .weight_high(weight_vectors[((output_lane*2+1)*64+input_lane)*8 +: 8]),
                    .product_low(products[(output_lane*2)*64+input_lane]),
                    .product_high(products[(output_lane*2+1)*64+input_lane]));
            end
        end
    endgenerate

    reg v0,v1,v2,v3,v4,v5,v6,v7,v8;
    reg [19:0] m0,m1,m2,m3,m4,m5,m6,m7,m8;
    reg [127:0] b0,b1,b2,b3,b4,b5,b6,b7,b8;
    reg signed [16:0] l1 [0:127];
    reg signed [17:0] l2 [0:63];
    reg signed [18:0] l3 [0:31];
    reg signed [19:0] l4 [0:15];
    reg signed [20:0] l5 [0:7];
    reg signed [20:0] l6 [0:3];
    integer k;

    always @(posedge clk) begin
        if(!rst_n) begin
            v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;v8<=0;
            sum_valid<=0;
        end else begin
            if(frame_start) begin
                v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;v8<=0;
                sum_valid<=0;
            end else begin
                v0<=issue_valid; v1<=v0; v2<=v1; v3<=v2; v4<=v3;
                v5<=v4; v6<=v5; v7<=v6; v8<=v7; sum_valid<=v8;
            end
            // Metadata, bias and adder payloads are valid-qualified only.
            // Keeping them outside frame_start removes a distributed CE.
            if(issue_valid) begin m0<=issue_metadata; b0<=bias_values; end
            if(v0) begin m1<=m0;b1<=b0;end
            if(v1) begin m2<=m1;b2<=b1;end
            if(v2) begin
                m3<=m2;b3<=b2;
                for(k=0;k<128;k=k+1)
                    l1[k] <= $signed({products[2*k][15],products[2*k]})
                           + $signed({products[2*k+1][15],products[2*k+1]});
            end
            if(v3) begin
                m4<=m3;b4<=b3;
                for(k=0;k<64;k=k+1)
                    l2[k] <= $signed({l1[2*k][16],l1[2*k]})
                           + $signed({l1[2*k+1][16],l1[2*k+1]});
            end
            if(v4) begin
                m5<=m4;b5<=b4;
                for(k=0;k<32;k=k+1)
                    l3[k] <= $signed({l2[2*k][17],l2[2*k]})
                           + $signed({l2[2*k+1][17],l2[2*k+1]});
            end
            if(v5) begin
                m6<=m5;b6<=b5;
                for(k=0;k<16;k=k+1)
                    l4[k] <= $signed({l3[2*k][18],l3[2*k]})
                           + $signed({l3[2*k+1][18],l3[2*k+1]});
            end
            if(v6) begin
                m7<=m6;b7<=b6;
                for(k=0;k<8;k=k+1)
                    l5[k] <= $signed({l4[2*k][19],l4[2*k]})
                           + $signed({l4[2*k+1][19],l4[2*k+1]});
            end
            if(v7) begin
                m8<=m7;b8<=b7;
                for(k=0;k<4;k=k+1)
                    l6[k] <= $signed(l5[2*k]) + $signed(l5[2*k+1]);
            end
            // This extra register is intentional: the four wide dot sums and
            // four biases never form one long combinational adder level.
            if(v8) begin
                sum_0 <= $signed(l6[0]) + $signed(b8[0 +: 21]);
                sum_1 <= $signed(l6[1]) + $signed(b8[32 +: 21]);
                sum_2 <= $signed(l6[2]) + $signed(b8[64 +: 21]);
                sum_3 <= $signed(l6[3]) + $signed(b8[96 +: 21]);
                sum_metadata <= m8;
            end
        end
    end
endmodule

module spa_out_proj_rom_stream_block0 #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] FIXED_MULTIPLIER=16'sd30313,
    parameter signed [6:0]  FIXED_SHIFT=7'sd19
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[9:0]in_record_addr,
    input wire[5:0]in_channel_base,input wire[31:0]in_data,
    input wire cfg_weight_we,input wire[10:0]cfg_weight_addr,
    input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,
    input wire[4:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,
    input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,
    output reg out_valid,output reg[7:0]out_pixel_addr,
    output reg[4:0]out_channel_base,output reg[31:0]out_raw_data,
    output reg[31:0]out_data,output reg done
);
    // Legacy configuration ports are retained for top-level compatibility.
    wire unused_cfg = cfg_weight_we ^ cfg_bias_we ^ cfg_weight_addr[0]
                    ^ cfg_weight_data[0] ^ cfg_bias_addr[0]
                    ^ cfg_bias_data[0] ^ cfg_multiplier[0] ^ cfg_shift[0];
    reg signed[7:0] activation_mem[0:63];
    reg [511:0] record_activation;
    wire[511:0] activation_wire;
    genvar g;
    generate for(g=0;g<64;g=g+1) begin:G_ACT
        assign activation_wire[g*8 +: 8]=activation_mem[g];
    end endgenerate
    integer i;
    always @(posedge clk) begin
        if(!rst_n)
            record_activation <= 512'd0;
        else if(in_valid) begin
            for(i=0;i<4;i=i+1)
                activation_mem[in_channel_base+i] <= in_data[i*8 +: 8];
            if(in_channel_base==6'd60)
                // Freeze the complete record before the following record can
                // overwrite activation_mem[0:3].  The current final beat must
                // be bypassed because its RAM writes complete after this edge.
                record_activation <= {in_data,activation_wire[479:0]};
        end
    end

    reg active;
    reg[9:0] record_q;
    reg[2:0] group_q;
    always @(posedge clk) begin
        if(!rst_n || frame_start) begin active<=0;record_q<=0;group_q<=0;end
        else begin
            if(active) begin
                if(group_q==7) begin active<=0;group_q<=0;end
                else group_q<=group_q+1'b1;
            end
            if(in_valid && in_channel_base==6'd60) begin
                active<=1;record_q<=in_record_addr;group_q<=0;
            end
        end
    end

    wire[4:0] row0={group_q,2'b00};
    wire[4:0] row1=row0+1'b1,row2=row0+2'd2,row3=row0+2'd3;
    wire[543:0] p0,p1,p2,p3;
    mamba_spa_out_param_rom_4r #(.BLOCK_ID(BLOCK_ID)) u_rom(
        // ROM data is consumed only when rom_valid is asserted.  Keeping the
        // native read ports enabled removes active from the replicated BRAM
        // EN network without changing the one-cycle ROM timing contract.
        .clk(clk),.en(1'b1),.addr0(row0),.addr1(row1),.addr2(row2),.addr3(row3),
        .data0(p0),.data1(p1),.data2(p2),.data3(p3));
    reg rom_valid;
    reg[511:0] act_d;
    reg[19:0] meta_d;
    always @(posedge clk) begin
        if(!rst_n) begin rom_valid<=0;end
        else begin
            if(frame_start) rom_valid<=0;
            else rom_valid<=active;
            if(active) begin act_d<=record_activation;meta_d<={7'd0,record_q,group_q};end
        end
    end
    wire dot_valid;
    wire signed[20:0] s0,s1,s2,s3;
    wire[19:0] dot_meta;
    out_proj_dot64x4_bias_pipeline u_dot(
        .clk(clk),.rst_n(rst_n),.frame_start(frame_start),.issue_valid(rom_valid),
        .activation_vector(act_d),.weight_vectors({p3[511:0],p2[511:0],p1[511:0],p0[511:0]}),
        .bias_values({p3[543:512],p2[543:512],p1[543:512],p0[543:512]}),
        .issue_metadata(meta_d),.sum_valid(dot_valid),.sum_0(s0),.sum_1(s1),
        .sum_2(s2),.sum_3(s3),.sum_metadata(dot_meta));
    wire signed[20:0] sums[0:3];
    assign sums[0]=s0;assign sums[1]=s1;assign sums[2]=s2;assign sums[3]=s3;
    wire signed[36:0] rq[0:3];
    generate for(g=0;g<4;g=g+1) begin:G_RQ
        requant_mult_21x16 u(.CLK(clk),.A(sums[g]),.B(FIXED_MULTIPLIER),.P(rq[g]));
    end endgenerate
    reg rv0,rv1,rv2; reg[19:0] rm0,rm1,rm2;
    function[7:0] q8;input signed[36:0]v;reg signed[63:0]a,r;begin
        a=v<0 ? -$signed({{27{v[36]}},v}) : $signed({{27{v[36]}},v});
        r=FIXED_SHIFT>0 ? (a+(64'sd1<<<(FIXED_SHIFT-1)))>>>FIXED_SHIFT : a;
        if(v<0)r=-r;if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];
    end endfunction
    integer o; reg[7:0] raw;
    always @(posedge clk) begin
        if(!rst_n) begin rv0<=0;rv1<=0;rv2<=0;
            out_valid<=0;done<=0;end
        else begin
            if(frame_start) begin
                rv0<=0;rv1<=0;rv2<=0;out_valid<=0;
                out_pixel_addr<=0;out_channel_base<=0;done<=0;
            end else begin
                rv0<=dot_valid;rv1<=rv0;rv2<=rv1;out_valid<=rv2;done<=0;
            end
            if(dot_valid)rm0<=dot_meta;if(rv0)rm1<=rm0;if(rv1)rm2<=rm1;
            if(rv2) begin
                if(!frame_start) begin out_pixel_addr<=rm2[10:3];out_channel_base<={rm2[2:0],2'b00}; end
                for(o=0;o<4;o=o+1) begin raw=q8(rq[o]);out_raw_data[o*8 +: 8]<=raw;out_data[o*8 +: 8]<=raw[7]?8'd0:raw;end
                if(!frame_start && (rm2[12:3]==PIXEL_COUNT-1) && rm2[2:0]==7)done<=1;
            end
        end
    end
endmodule

module spe_out_proj_rom_stream_block0 #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] FIXED_MULTIPLIER=16'sd32455,
    parameter signed [6:0]  FIXED_SHIFT=7'sd18
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[9:0]in_record_addr,input wire[5:0]in_channel_base,
    input wire[31:0]in_data,input wire cfg_weight_we,input wire[6:0]cfg_weight_addr,
    input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,input wire[2:0]cfg_bias_addr,
    input wire signed[31:0]cfg_bias_data,input wire signed[15:0]cfg_multiplier,
    input wire signed[6:0]cfg_shift,output reg out_valid,output reg[7:0]out_pixel_addr,
    output reg[1:0]out_token,output reg[63:0]out_raw_data,output reg[63:0]out_data,
    output reg done
);
    wire unused_cfg=cfg_weight_we^cfg_bias_we^cfg_weight_addr[0]^cfg_weight_data[0]
                   ^cfg_bias_addr[0]^cfg_bias_data[0]^cfg_multiplier[0]^cfg_shift[0];
    reg signed[7:0] act[0:15];integer i;
    always @(posedge clk) if(in_valid)for(i=0;i<4;i=i+1)act[in_channel_base+i]<=in_data[i*8 +: 8];
    wire[127:0] act_wire;genvar g;
    generate for(g=0;g<16;g=g+1)begin:G_A assign act_wire[g*8 +:8]=act[g];end endgenerate
    wire[1279:0] params;
    wire issue=in_valid && in_channel_base==6'd12;
    mamba_spe_out_param_rom #(.BLOCK_ID(BLOCK_ID))u_rom(.clk(clk),.en(issue),.data(params));
    reg rom_valid;reg[127:0]act_d;reg[9:0]rec_d;
    always @(posedge clk)begin
        if(!rst_n)begin rom_valid<=0;end
        else begin
        if(frame_start)begin rom_valid<=0;rec_d<=0;end
        else rom_valid<=issue;
        if(issue)begin
            // Include the final channel-12 beat captured on this same edge.
            act_d<={in_data,act_wire[95:0]};rec_d<=in_record_addr;
        end end
    end
    wire[1023:0] weights;
    wire[255:0] biases;
    generate for(g=0;g<8;g=g+1)begin:G_UNPACK
        assign weights[g*128 +:128]=params[g*160 +:128];
        assign biases[g*32 +:32]=params[g*160+128 +:32];
    end endgenerate
    wire sv;wire[167:0]sumv;wire[19:0]sm;
    out_proj_dot16x8_pipeline u_dot(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),
        .issue_valid(rom_valid),.activation_vector(act_d),.weight_vectors(weights),
        .issue_metadata({10'd0,rec_d}),.sum_valid(sv),.sum_values(sumv),.sum_metadata(sm));
    reg av;reg[9:0]ar;reg signed[20:0]acc[0:7];
    always @(posedge clk)begin
        if(!rst_n||frame_start)begin av<=0;ar<=0;end else begin av<=sv;
            if(sv)begin ar<=sm[9:0];for(i=0;i<8;i=i+1)acc[i]<=$signed(sumv[i*21 +:21])+$signed(biases[i*32 +:21]);end
        end
    end
    wire signed[36:0]rq[0:7];
    generate for(g=0;g<8;g=g+1)begin:G_RQ requant_mult_21x16 u(.CLK(clk),.A(acc[g]),.B(FIXED_MULTIPLIER),.P(rq[g]));end endgenerate
    reg rv0,rv1,rv2;reg[9:0]rr0,rr1,rr2;
    function[7:0]q8;input signed[36:0]v;reg signed[63:0]a,r;begin
        a=v<0?-$signed({{27{v[36]}},v}):$signed({{27{v[36]}},v});
        r=(a+(64'sd1<<<(FIXED_SHIFT-1)))>>>FIXED_SHIFT;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    integer o;reg[7:0]raw;
    always @(posedge clk)begin
        if(!rst_n)begin rv0<=0;rv1<=0;rv2<=0;
            out_valid<=0;done<=0;end
        else begin
            if(frame_start)begin rv0<=0;rv1<=0;rv2<=0;
                out_valid<=0;out_pixel_addr<=0;out_token<=0;done<=0;end
            else begin rv0<=av;rv1<=rv0;rv2<=rv1;out_valid<=rv2;done<=0;end
            if(av)rr0<=ar;if(rv0)rr1<=rr0;if(rv1)rr2<=rr1;
            if(rv2)begin if(!frame_start)begin out_pixel_addr<=rr2[9:2];out_token<=rr2[1:0];end
                for(o=0;o<8;o=o+1)begin raw=q8(rq[o]);out_raw_data[o*8 +:8]<=raw;out_data[o*8 +:8]<=raw[7]?8'd0:raw;end
                if(!frame_start && rr2==(PIXEL_COUNT*4)-1)done<=1;end
        end
    end
endmodule

`default_nettype wire
