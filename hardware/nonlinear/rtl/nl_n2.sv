`timescale 1ns/1ps

// Local adaptation of S2Mamba Eq9/10: y=x*log2(e), 2^frac(y) affine,
// followed by integer shift. Three pipeline registers to target FPGA timing.
module nl_n2_exp #(
    parameter [31:0] LAMBDA=32'd1073741824,
    parameter [31:0] ETA=32'd1073741824
) (
    input wire clk,
    input wire signed [63:0] x24,
    output reg [63:0] result30
);
    localparam signed [31:0] LOG2E=32'sd1549082005;
    wire signed [95:0] product = x24 * LOG2E;
    reg signed [63:0] y30;
    reg signed [15:0] exponent;
    reg [63:0] affine60;
    wire [61:0] fraction_product = y30[29:0] * LAMBDA;
    always @(posedge clk) begin
        y30 <= product >>> 24;
        affine60 <= {2'b0, fraction_product} + ({32'd0, ETA} << 30);
        if ((y30 >>> 30) < -63)
            exponent <= -64;
        else if ((y30 >>> 30) > 32)
            exponent <= 33;
        else
            exponent <= y30 >>> 30;
        if (exponent < -63)
            result30 <= 0;
        else if (exponent > 32)
            result30 <= 64'hffffffffffffffff;
        else if (exponent < 0)
            result30 <= (affine60 >> 30) >> (-exponent);
        else
            result30 <= (affine60 >> 30) << exponent;
    end
endmodule

// INT8 dt -> 16xUQ1.24 Abar + U19 K. No dt-indexed answer table.
// LOD/RU follows Eq20; exp/log coefficients frozen by two-stage offline fit.
// Numeric contract unchanged from N0/N1/N3. Local FPGA adaptation, not author RTL.
module nl_coeff_n2 #(
    parameter [31:0] SDT_Q30=0,
    parameter [47:0] SBSU_Q40=0,
    parameter [767:0] DECAY_Q24=0,
    parameter integer K_FRAC=24,
    parameter [31:0] EXP_L=0, EXP_B=0, STATE_L=0, STATE_B=0,
    parameter signed [31:0] LOG_L=0, LOG_B=0
) (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output reg [418:0] out_coeff
);
    localparam signed [63:0] LN2=64'sd744261118;
    wire [8:0] abs_dt = q_dt[7] ? -$signed({q_dt[7],q_dt}) : {1'b0,q_dt};
    wire [40:0] scaled_dt = abs_dt * SDT_Q30;
    wire [63:0] magnitude = ({23'd0,scaled_dt}+64'd32) >> 6;
    reg signed [63:0] x24;
    always @(posedge clk) x24 <= q_dt[7] ? -$signed(magnitude) : $signed(magnitude);
    wire [63:0] exp_soft;
    nl_n2_exp #(.LAMBDA(EXP_L),.ETA(EXP_B)) u_exp_soft(.clk(clk),.x24(x24),.result30(exp_soft));
    reg [63:0] alpha;
    reg [30:0] k30;
    reg [5:0] w;
    integer bit_index, lead;
    always @* begin
        lead=30;
        for(bit_index=30;bit_index<64;bit_index=bit_index+1)
            if(alpha[bit_index]) lead=bit_index;
    end
    always @(posedge clk) begin
        alpha <= exp_soft + 64'd1073741824;
        w <= lead-30;
        k30 <= alpha >> (lead-30);
    end
    wire signed [31:0] fraction_log = $signed({1'b0,k30})-32'sd1073741824;
    reg signed [63:0] log_product, w_product;
    wire signed [63:0] delta30 = (log_product >>> 30) + w_product + $signed(LOG_B);
    reg [47:0] delta24;
    always @(posedge clk) begin
        log_product <= fraction_log * LOG_L;
        w_product <= $signed({1'b0,w}) * LN2;
        delta24 <= (delta30 + 64'sd32 < 0) ? 48'd0 : ((delta30+64'sd32) >>> 6);
    end
    wire [63:0] state_exp [0:15];
    genvar s;
    generate for(s=0;s<16;s=s+1) begin: G_STATE
        wire [95:0] decay_product=delta24 * DECAY_Q24[s*48+:48];
        reg [47:0] mag24;
        always @(posedge clk) mag24 <= (decay_product+96'd8388608) >> 24;
        wire signed [63:0] negative_mag=-$signed({16'd0,mag24});
        nl_n2_exp #(.LAMBDA(STATE_L),.ETA(STATE_B)) u_exp_state(
            .clk(clk),.x24(negative_mag),.result30(state_exp[s]));
        wire [63:0] rounded_a=(state_exp[s]+64'd32) >> 6;
        always @(posedge clk)
            out_coeff[s*25+:25] <= rounded_a>64'd16777216 ? 25'd16777216 : rounded_a[24:0];
    end endgenerate
    wire [95:0] k_product=delta24*SBSU_Q40;
    wire [95:0] rounded_k=(k_product+(96'd1<<(63-K_FRAC))) >> (64-K_FRAC);
    reg [18:0] kq;
    wire [18:0] aligned_k;
    always @(posedge clk) kq <= rounded_k>96'd524287 ? 19'd524287 : rounded_k[18:0];
    nl_delay #(.WIDTH(19),.DEPTH(3)) u_k(.clk(clk),.rst(rst),.din(kq),.dout(aligned_k));
    always @(posedge clk) out_coeff[418:400] <= aligned_k;
    reg [12:0] valid_pipe;
    always @(posedge clk) begin
        if(rst) valid_pipe<=0;
        else valid_pipe<={valid_pipe[11:0],in_valid};
    end
    assign out_valid=valid_pipe[12];
    nl_delay #(.WIDTH(8),.DEPTH(13)) u_tag(
        .clk(clk),.rst(rst),.din(q_dt^8'h80),.dout(out_address));
endmodule
