`timescale 1ns/1ps
`include "math_constants.vh"

// All modules accept one request per cycle. Reset invalidates payload only.
module nl_delay #(parameter integer WIDTH=1, DEPTH=1) (
    input wire clk, rst, input wire [WIDTH-1:0] din,
    output wire [WIDTH-1:0] dout
);
    reg [WIDTH-1:0] pipe [0:DEPTH-1];
    integer i;
    always @(posedge clk) begin
        pipe[0] <= din;
        for (i=1; i<DEPTH; i=i+1) pipe[i] <= pipe[i-1];
    end
    assign dout = pipe[DEPTH-1];
endmodule

module nl_poly #(
    parameter integer DEGREE=10,
    parameter [32*(DEGREE+1)-1:0] COEFF=`EXP_COEFF
) (
    input wire clk, rst, in_valid,
    input wire signed [31:0] x,
    output wire out_valid,
    output wire signed [31:0] y
);
    reg signed [31:0] xp [0:DEGREE-1];
    reg signed [31:0] yp [0:DEGREE-1];
    reg [DEGREE-1:0] vp;
    genvar g;
    generate for(g=0; g<DEGREE; g=g+1) begin: G_HORNER
        wire signed [31:0] xi;
        wire signed [31:0] yi;
        if(g==0) begin
            assign xi=x;
            assign yi=$signed(COEFF[DEGREE*32+:32]);
        end else begin
            assign xi=xp[g-1];
            assign yi=yp[g-1];
        end
        wire signed [63:0] product=xi*yi;
        always @(posedge clk) begin
            xp[g] <= xi;
            yp[g] <= (product >>> 30) + $signed(COEFF[(DEGREE-1-g)*32+:32]);
            if(rst) vp[g]<=1'b0;
            else if(g==0) vp[g]<=in_valid;
            else vp[g]<=vp[g-1];
        end
    end endgenerate
    assign y=yp[DEGREE-1];
    assign out_valid=vp[DEGREE-1];
endmodule

// exp(-x), x is unsigned Q24, output unsigned Q30.
// N0: L=13 registers; N1: L=4 registers. x>=32 is zero at Q24 precision.
module nl_exp_neg #(parameter integer METHOD=0) (
    input wire clk, rst, in_valid,
    input wire [47:0] x24,
    output wire out_valid,
    output wire [31:0] y30
);
    localparam [31:0] LOG2E_Q30=32'd1549082005;
    reg [79:0] product;
    reg cut1, cut2, v1, v2;
    reg [5:0] power2;
    reg [29:0] fraction;
    wire [79:0] shifted=product >> 24;
    always @(posedge clk) begin
        if(METHOD==0) product <= x24*LOG2E_Q30;
        else product <= x24*32'd1543503872; // 23/16 in Q30 (paper Eq.3)
        cut1 <= x24 >= 48'd536870912;
        cut2 <= cut1;
        power2 <= shifted[35:30];
        fraction <= shifted[29:0];
        if(rst) begin v1<=0; v2<=0; end
        else begin v1<=in_valid; v2<=v1; end
    end
    generate if(METHOD==0) begin: G_POLY
        wire pv;
        wire signed [31:0] py;
        wire [6:0] meta;
        nl_poly #(.DEGREE(10), .COEFF(`EXP_COEFF)) u_poly(
            .clk(clk),.rst(rst),.in_valid(v2),.x({2'b00,fraction}),.out_valid(pv),.y(py));
        nl_delay #(.WIDTH(7),.DEPTH(10)) u_meta(
            .clk(clk),.rst(rst),.din({cut2,power2}),.dout(meta));
        reg ov;
        reg [31:0] oy;
        always @(posedge clk) begin
            if(rst) ov<=0; else ov<=pv;
            oy <= (meta[6] || meta[5:0]>=32 || py<0) ? 0 : ($unsigned(py) >> meta[5:0]);
        end
        assign out_valid=ov;
        assign y30=oy;
    end else begin: G_PWL
        localparam [287:0] EP=`PWL_ENDPOINT;
        wire [2:0] seg=fraction[29:27];
        wire [31:0] e0=EP[seg*32+:32];
        wire [31:0] e1=EP[(seg+1)*32+:32];
        wire [58:0] slope_product=(e0-e1)*fraction[26:0];
        reg [31:0] interp;
        reg [5:0] p3;
        reg c3,v3,ov;
        reg [31:0] oy;
        always @(posedge clk) begin
            interp <= e0-(slope_product >> 27);
            p3<=power2;
            c3<=cut2;
            oy <= (c3 || p3>=32) ? 0 : (interp >> p3);
            if(rst) begin v3<=0; ov<=0; end
            else begin v3<=v2; ov<=v3; end
        end
        assign out_valid=ov;
        assign y30=oy;
    end endgenerate
endmodule

// softplus(sign*x24), Q24 output. N0 L=43, N1 L=5.
// N0 log(1+y)=2*z*(1+z^2/3+...+z^16/17), z=y/(2+y).
// Reciprocal is evaluated online via degree-18 geometric polynomial.
module nl_softplus #(parameter integer METHOD=0) (
    input wire clk,rst,in_valid,positive,
    input wire [47:0] x24,
    output wire out_valid,
    output wire [47:0] delta24
);
    localparam integer EL=(METHOD==0)?13:4;
    wire ev;
    wire [31:0] ey;
    wire [47:0] max_x;
    nl_exp_neg #(.METHOD(METHOD)) u_exp(
        .clk(clk),.rst(rst),.in_valid(in_valid),.x24(x24),.out_valid(ev),.y30(ey));
    nl_delay #(.WIDTH(48),.DEPTH(EL)) u_x(
        .clk(clk),.rst(rst),.din(positive?x24:48'd0),.dout(max_x));
    generate if(METHOD==1) begin: G_FAST
        reg ov;
        reg [47:0] delta;
        always @(posedge clk) begin
            delta<=max_x+(({16'd0,ey}+48'd32)>>6);
            if(rst) ov<=0; else ov<=ev;
        end
        assign out_valid=ov;
        assign delta24=delta;
    end else begin: G_LOG
        wire [63:0] r_product=(32'd1073741824-ey)*32'd357913941;
        reg [31:0] r;
        reg rv;
        reg [31:0] y1;
        reg [47:0] x1;
        always @(posedge clk) begin
            r<=r_product>>30;
            y1<=ey;
            x1<=max_x;
            if(rst) rv<=0; else rv<=ev;
        end
        wire iv;
        wire signed [31:0] reciprocal;
        wire [79:0] m1;
        nl_poly #(.DEGREE(18),.COEFF(`REC_COEFF)) u_recip(
            .clk(clk),.rst(rst),.in_valid(rv),.x(r),.out_valid(iv),.y(reciprocal));
        nl_delay #(.WIDTH(80),.DEPTH(18)) u_m1(
            .clk(clk),.rst(rst),.din({x1,y1}),.dout(m1));
        wire [63:0] zp=m1[31:0]*$unsigned(reciprocal);
        reg [31:0] z,z2;
        reg [47:0] x2,x3;
        reg [31:0] z3;
        reg zv,z2v;
        wire [63:0] zsquare=z*z;
        always @(posedge clk) begin
            z<=zp>>30;
            x2<=m1[79:32];
            z2<=zsquare>>30;
            z3<=z;
            x3<=x2;
            if(rst) begin zv<=0; z2v<=0; end
            else begin zv<=iv; z2v<=zv; end
        end
        wire lv;
        wire signed [31:0] lp;
        wire [79:0] m2;
        nl_poly #(.DEGREE(8),.COEFF(`LOG_COEFF)) u_log(
            .clk(clk),.rst(rst),.in_valid(z2v),.x(z2),.out_valid(lv),.y(lp));
        nl_delay #(.WIDTH(80),.DEPTH(8)) u_m2(
            .clk(clk),.rst(rst),.din({x3,z3}),.dout(m2));
        wire [63:0] logprod=m2[31:0]*$unsigned(lp);
        wire [63:0] log30=(logprod<<1)>>30;
        reg ov;
        reg [47:0] delta;
        always @(posedge clk) begin
            delta<=m2[79:32]+((log30+64'd32)>>6);
            if(rst) ov<=0; else ov<=lv;
        end
        assign out_valid=ov;
        assign delta24=delta;
    end endgenerate
endmodule
