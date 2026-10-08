`timescale 1ns/1ps
`default_nettype none

// One signed INT8 activation is shared by two signed INT8 output-channel
// weights.  The same DSP Macro definition used by patch embedding performs
// (A+D)*B in three cycles.  A 17-bit lane spacing is sufficient for a signed
// 8x8 product.  A negative low product borrows one from the packed high lane;
// add that bit back after extraction to recover the exact second product.
module mamba_packed_signed_mult_2x8_3cyc(
    input  wire                    clk,
    input  wire signed [7:0]       activation,
    input  wire signed [7:0]       weight_low,
    input  wire signed [7:0]       weight_high,
    output wire signed [15:0]      product_low,
    output wire signed [15:0]      product_high
);
    wire signed [26:0] dsp_a
        = $signed({{2{weight_high[7]}},weight_high,17'b0});
    wire signed [26:0] dsp_d
        = $signed({{19{weight_low[7]}},weight_low});
    wire signed [8:0] dsp_b
        = $signed({activation[7],activation});
    wire signed [35:0] dsp_p;
    wire signed [16:0] high_corrected;

    patch_packed_dsp_3cyc u_dsp(
        .CLK(clk),.A(dsp_a),.B(dsp_b),.D(dsp_d),.P(dsp_p));

    assign product_low=dsp_p[15:0];
    assign high_corrected
        =$signed({dsp_p[32],dsp_p[32:17]})
        +$signed({16'd0,dsp_p[16]});
    assign product_high=high_corrected[15:0];
endmodule

// Rank/dt_proj variant: one signed INT9 activation and two signed INT8
// weights.  The product is signed 17-bit, therefore the packed lane spacing
// is 18 bits.
module mamba_packed_signed_mult_2x9_3cyc(
    input  wire                    clk,
    input  wire signed [8:0]       activation,
    input  wire signed [7:0]       weight_low,
    input  wire signed [7:0]       weight_high,
    output wire signed [16:0]      product_low,
    output wire signed [16:0]      product_high
);
    wire signed [26:0] dsp_a
        =$signed({weight_high[7],weight_high,18'b0});
    wire signed [26:0] dsp_d
        =$signed({{19{weight_low[7]}},weight_low});
    wire signed [35:0] dsp_p;
    wire signed [17:0] high_corrected;

    patch_packed_dsp_3cyc u_dsp(
        .CLK(clk),.A(dsp_a),.B(activation),.D(dsp_d),.P(dsp_p));

    assign product_low=dsp_p[16:0];
    assign high_corrected
        =$signed({dsp_p[35],dsp_p[34:18]})
        +$signed({17'd0,dsp_p[17]});
    assign product_high=high_corrected[16:0];
endmodule

module mamba_dot64_pair_pipeline #(parameter integer META_WIDTH=24)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[511:0]activation,input wire[511:0]weight_low,
    input wire[511:0]weight_high,input wire[META_WIDTH-1:0]in_meta,
    output wire out_valid,output wire signed[20:0]out_sum_low,
    output wire signed[20:0]out_sum_high,
    output wire[META_WIDTH-1:0]out_meta);
    wire signed[15:0]pl[0:63],ph[0:63];genvar i;
    generate for(i=0;i<64;i=i+1)begin:G
        mamba_packed_signed_mult_2x8_3cyc u(.clk(clk),
            .activation(activation[i*8+:8]),.weight_low(weight_low[i*8+:8]),
            .weight_high(weight_high[i*8+:8]),.product_low(pl[i]),.product_high(ph[i]));
    end endgenerate
    reg v0,v1,v2,v3,v4,v5,v6,v7,v8;
    reg[META_WIDTH-1:0]m0,m1,m2,m3,m4,m5,m6,m7,m8;
    reg signed[16:0]l1l[0:31],l1h[0:31];reg signed[17:0]l2l[0:15],l2h[0:15];
    reg signed[18:0]l3l[0:7],l3h[0:7];reg signed[19:0]l4l[0:3],l4h[0:3];
    reg signed[20:0]l5l[0:1],l5h[0:1];reg signed[21:0]sl,sh;integer k;
    always@(posedge clk)begin
        if(!rst_n)begin
            v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;v8<=0;
        end else begin
            if(frame_start)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;v8<=0;end
            else begin v0<=in_valid;v1<=v0;v2<=v1;v3<=v2;v4<=v3;v5<=v4;v6<=v5;v7<=v6;v8<=v7;end
            // Payload/metadata are deliberately free-running.  v0..v8 are
            // the sole validity contract and no longer drive wide CE trees.
            m0<=in_meta;m1<=m0;m2<=m1;m3<=m2;m4<=m3;m5<=m4;m6<=m5;m7<=m6;m8<=m7;
            for(k=0;k<32;k=k+1)begin
                l1l[k]<=$signed({pl[2*k][15],pl[2*k]})+$signed({pl[2*k+1][15],pl[2*k+1]});
                l1h[k]<=$signed({ph[2*k][15],ph[2*k]})+$signed({ph[2*k+1][15],ph[2*k+1]});end
            for(k=0;k<16;k=k+1)begin
                l2l[k]<=$signed({l1l[2*k][16],l1l[2*k]})+$signed({l1l[2*k+1][16],l1l[2*k+1]});
                l2h[k]<=$signed({l1h[2*k][16],l1h[2*k]})+$signed({l1h[2*k+1][16],l1h[2*k+1]});end
            for(k=0;k<8;k=k+1)begin
                l3l[k]<=$signed({l2l[2*k][17],l2l[2*k]})+$signed({l2l[2*k+1][17],l2l[2*k+1]});
                l3h[k]<=$signed({l2h[2*k][17],l2h[2*k]})+$signed({l2h[2*k+1][17],l2h[2*k+1]});end
            for(k=0;k<4;k=k+1)begin
                l4l[k]<=$signed({l3l[2*k][18],l3l[2*k]})+$signed({l3l[2*k+1][18],l3l[2*k+1]});
                l4h[k]<=$signed({l3h[2*k][18],l3h[2*k]})+$signed({l3h[2*k+1][18],l3h[2*k+1]});end
            for(k=0;k<2;k=k+1)begin
                l5l[k]<=$signed({l4l[2*k][19],l4l[2*k]})+$signed({l4l[2*k+1][19],l4l[2*k+1]});
                l5h[k]<=$signed({l4h[2*k][19],l4h[2*k]})+$signed({l4h[2*k+1][19],l4h[2*k+1]});end
            sl<=$signed({l5l[0][20],l5l[0]})+$signed({l5l[1][20],l5l[1]});
            sh<=$signed({l5h[0][20],l5h[0]})+$signed({l5h[1][20],l5h[1]});
        end
    end
    assign out_valid=v8;assign out_sum_low=sl[20:0];assign out_sum_high=sh[20:0];assign out_meta=m8;
endmodule

module mamba_dot32_pair_pipeline #(parameter integer META_WIDTH=24)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[255:0]activation,input wire[255:0]weight_low,
    input wire[255:0]weight_high,input wire[META_WIDTH-1:0]in_meta,
    output wire out_valid,output wire signed[20:0]out_sum_low,
    output wire signed[20:0]out_sum_high,output wire[META_WIDTH-1:0]out_meta);
    wire signed[15:0]pl[0:31],ph[0:31];genvar i;
    generate for(i=0;i<32;i=i+1)begin:G
        mamba_packed_signed_mult_2x8_3cyc u(.clk(clk),.activation(activation[i*8+:8]),
            .weight_low(weight_low[i*8+:8]),.weight_high(weight_high[i*8+:8]),
            .product_low(pl[i]),.product_high(ph[i]));end endgenerate
    reg v0,v1,v2,v3,v4,v5,v6,v7;reg[META_WIDTH-1:0]m0,m1,m2,m3,m4,m5,m6,m7;
    reg signed[16:0]l1l[0:15],l1h[0:15];reg signed[17:0]l2l[0:7],l2h[0:7];
    reg signed[18:0]l3l[0:3],l3h[0:3];reg signed[19:0]l4l[0:1],l4h[0:1];
    reg signed[20:0]sl,sh;integer k;
    always@(posedge clk)begin if(!rst_n)begin
        v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;
    end else begin if(frame_start)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;v7<=0;end
        else begin v0<=in_valid;v1<=v0;v2<=v1;v3<=v2;v4<=v3;v5<=v4;v6<=v5;v7<=v6;end
        m0<=in_meta;m1<=m0;m2<=m1;m3<=m2;m4<=m3;m5<=m4;m6<=m5;m7<=m6;
        for(k=0;k<16;k=k+1)begin l1l[k]<=$signed({pl[2*k][15],pl[2*k]})+$signed({pl[2*k+1][15],pl[2*k+1]});l1h[k]<=$signed({ph[2*k][15],ph[2*k]})+$signed({ph[2*k+1][15],ph[2*k+1]});end
        for(k=0;k<8;k=k+1)begin l2l[k]<=$signed({l1l[2*k][16],l1l[2*k]})+$signed({l1l[2*k+1][16],l1l[2*k+1]});l2h[k]<=$signed({l1h[2*k][16],l1h[2*k]})+$signed({l1h[2*k+1][16],l1h[2*k+1]});end
        for(k=0;k<4;k=k+1)begin l3l[k]<=$signed({l2l[2*k][17],l2l[2*k]})+$signed({l2l[2*k+1][17],l2l[2*k+1]});l3h[k]<=$signed({l2h[2*k][17],l2h[2*k]})+$signed({l2h[2*k+1][17],l2h[2*k+1]});end
        for(k=0;k<2;k=k+1)begin l4l[k]<=$signed({l3l[2*k][18],l3l[2*k]})+$signed({l3l[2*k+1][18],l3l[2*k+1]});l4h[k]<=$signed({l3h[2*k][18],l3h[2*k]})+$signed({l3h[2*k+1][18],l3h[2*k+1]});end
        sl<=$signed({l4l[0][19],l4l[0]})+$signed({l4l[1][19],l4l[1]});sh<=$signed({l4h[0][19],l4h[0]})+$signed({l4h[1][19],l4h[1]});
    end end
    assign out_valid=v7;assign out_sum_low=sl;assign out_sum_high=sh;assign out_meta=m7;
endmodule

module mamba_dot16_pair_pipeline #(parameter integer META_WIDTH=24)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[127:0]activation,input wire[127:0]weight_low,input wire[127:0]weight_high,
    input wire[META_WIDTH-1:0]in_meta,output wire out_valid,
    output wire signed[19:0]out_sum_low,output wire signed[19:0]out_sum_high,
    output wire[META_WIDTH-1:0]out_meta);
    wire signed[15:0]pl[0:15],ph[0:15];genvar i;
    generate for(i=0;i<16;i=i+1)begin:G mamba_packed_signed_mult_2x8_3cyc u(
        .clk(clk),.activation(activation[i*8+:8]),.weight_low(weight_low[i*8+:8]),
        .weight_high(weight_high[i*8+:8]),.product_low(pl[i]),.product_high(ph[i]));end endgenerate
    reg v0,v1,v2,v3,v4,v5,v6;reg[META_WIDTH-1:0]m0,m1,m2,m3,m4,m5,m6;
    reg signed[16:0]l1l[0:7],l1h[0:7];reg signed[17:0]l2l[0:3],l2h[0:3];
    reg signed[18:0]l3l[0:1],l3h[0:1];reg signed[19:0]sl,sh;integer k;
    always@(posedge clk)begin if(!rst_n)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;end else begin
        if(frame_start)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;v6<=0;end
        else begin v0<=in_valid;v1<=v0;v2<=v1;v3<=v2;v4<=v3;v5<=v4;v6<=v5;end
        m0<=in_meta;m1<=m0;m2<=m1;m3<=m2;m4<=m3;m5<=m4;m6<=m5;
        for(k=0;k<8;k=k+1)begin l1l[k]<=$signed({pl[2*k][15],pl[2*k]})+$signed({pl[2*k+1][15],pl[2*k+1]});l1h[k]<=$signed({ph[2*k][15],ph[2*k]})+$signed({ph[2*k+1][15],ph[2*k+1]});end
        for(k=0;k<4;k=k+1)begin l2l[k]<=$signed({l1l[2*k][16],l1l[2*k]})+$signed({l1l[2*k+1][16],l1l[2*k+1]});l2h[k]<=$signed({l1h[2*k][16],l1h[2*k]})+$signed({l1h[2*k+1][16],l1h[2*k+1]});end
        for(k=0;k<2;k=k+1)begin l3l[k]<=$signed({l2l[2*k][17],l2l[2*k]})+$signed({l2l[2*k+1][17],l2l[2*k+1]});l3h[k]<=$signed({l2h[2*k][17],l2h[2*k]})+$signed({l2h[2*k+1][17],l2h[2*k+1]});end
        sl<=$signed({l3l[0][18],l3l[0]})+$signed({l3l[1][18],l3l[1]});sh<=$signed({l3h[0][18],l3h[0]})+$signed({l3h[1][18],l3h[1]});end end
    assign out_valid=v6;assign out_sum_low=sl;assign out_sum_high=sh;assign out_meta=m6;
endmodule

module mamba_dot8_pair_pipeline #(parameter integer META_WIDTH=24)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[63:0]activation,input wire[63:0]weight_low,input wire[63:0]weight_high,
    input wire[META_WIDTH-1:0]in_meta,output wire out_valid,
    output wire signed[18:0]out_sum_low,output wire signed[18:0]out_sum_high,
    output wire[META_WIDTH-1:0]out_meta);
    wire signed[15:0]pl[0:7],ph[0:7];genvar i;
    generate for(i=0;i<8;i=i+1)begin:G mamba_packed_signed_mult_2x8_3cyc u(
        .clk(clk),.activation(activation[i*8+:8]),.weight_low(weight_low[i*8+:8]),
        .weight_high(weight_high[i*8+:8]),.product_low(pl[i]),.product_high(ph[i]));end endgenerate
    reg v0,v1,v2,v3,v4,v5;reg[META_WIDTH-1:0]m0,m1,m2,m3,m4,m5;
    reg signed[16:0]l1l[0:3],l1h[0:3];reg signed[17:0]l2l[0:1],l2h[0:1];
    reg signed[18:0]sl,sh;integer k;
    always@(posedge clk)begin if(!rst_n)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;end else begin
        if(frame_start)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;v5<=0;end
        else begin v0<=in_valid;v1<=v0;v2<=v1;v3<=v2;v4<=v3;v5<=v4;end
        m0<=in_meta;m1<=m0;m2<=m1;m3<=m2;m4<=m3;m5<=m4;
        for(k=0;k<4;k=k+1)begin l1l[k]<=$signed({pl[2*k][15],pl[2*k]})+$signed({pl[2*k+1][15],pl[2*k+1]});l1h[k]<=$signed({ph[2*k][15],ph[2*k]})+$signed({ph[2*k+1][15],ph[2*k+1]});end
        for(k=0;k<2;k=k+1)begin l2l[k]<=$signed({l1l[2*k][16],l1l[2*k]})+$signed({l1l[2*k+1][16],l1l[2*k+1]});l2h[k]<=$signed({l1h[2*k][16],l1h[2*k]})+$signed({l1h[2*k+1][16],l1h[2*k+1]});end
        sl<=$signed({l2l[0][17],l2l[0]})+$signed({l2l[1][17],l2l[1]});sh<=$signed({l2h[0][17],l2h[0]})+$signed({l2h[1][17],l2h[1]});end end
    assign out_valid=v5;assign out_sum_low=sl;assign out_sum_high=sh;assign out_meta=m5;
endmodule

module mamba_dot4_pair_pipeline #(parameter integer META_WIDTH=24)(
    input wire clk,input wire rst_n,input wire frame_start,input wire in_valid,
    input wire[31:0]activation,input wire[31:0]weight_low,input wire[31:0]weight_high,
    input wire[META_WIDTH-1:0]in_meta,output wire out_valid,
    output wire signed[17:0]out_sum_low,output wire signed[17:0]out_sum_high,
    output wire[META_WIDTH-1:0]out_meta);
    wire signed[15:0]pl[0:3],ph[0:3];genvar i;
    generate for(i=0;i<4;i=i+1)begin:G mamba_packed_signed_mult_2x8_3cyc u(
        .clk(clk),.activation(activation[i*8+:8]),.weight_low(weight_low[i*8+:8]),
        .weight_high(weight_high[i*8+:8]),.product_low(pl[i]),.product_high(ph[i]));end endgenerate
    reg v0,v1,v2,v3,v4;reg[META_WIDTH-1:0]m0,m1,m2,m3,m4;
    reg signed[16:0]l1l[0:1],l1h[0:1];reg signed[17:0]sl,sh;integer k;
    always@(posedge clk)begin if(!rst_n)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;end else begin
        if(frame_start)begin v0<=0;v1<=0;v2<=0;v3<=0;v4<=0;end
        else begin v0<=in_valid;v1<=v0;v2<=v1;v3<=v2;v4<=v3;end
        m0<=in_meta;m1<=m0;m2<=m1;m3<=m2;m4<=m3;
        for(k=0;k<2;k=k+1)begin l1l[k]<=$signed({pl[2*k][15],pl[2*k]})+$signed({pl[2*k+1][15],pl[2*k+1]});l1h[k]<=$signed({ph[2*k][15],ph[2*k]})+$signed({ph[2*k+1][15],ph[2*k+1]});end
        sl<=$signed({l1l[0][16],l1l[0]})+$signed({l1l[1][16],l1l[1]});sh<=$signed({l1h[0][16],l1h[0]})+$signed({l1h[1][16],l1h[1]});end end
    assign out_valid=v4;assign out_sum_low=sl;assign out_sum_high=sh;assign out_meta=m4;
endmodule

`default_nettype wire
