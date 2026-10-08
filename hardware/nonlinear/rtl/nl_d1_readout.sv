`timescale 1ns/1ps
`default_nettype none

// COMMON readout microbenchmark, not an entire recurrent SSM.
// c_readout is an externally supplied signed C*state reduction, NOT computed
// by the adjacent coefficient generator. Aligning C here is a test adapter.
// In the board SSM the existing C-product pipeline provides this alignment.
module nl_d1_readout #(
    parameter integer BLOCK_ID = 0,
    parameter integer IS_SPE = 0,
    parameter integer D_LATENCY = 17
) (
    input  wire clk,
    input  wire rst,
    input  wire read_valid,
    input  wire [5:0] channel,
    input  wire [7:0] u_s8,
    input  wire signed [47:0] c_readout,
    output reg out_valid,
    output reg signed [47:0] total_readout
);
    wire signed [47:0] d_readout;
    mamba_d_path_lut #(
        .BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE), .LATENCY(D_LATENCY)
    ) u_d_lut (
        .clk(clk), .channel(channel), .u_s8(u_s8), .d_s48(d_readout)
    );

    reg [D_LATENCY-1:0] valid_pipe;
    reg signed [47:0] c_delay [0:D_LATENCY-1];
    integer i;
    always @(posedge clk) begin
        // Payload is free-running; only validity is reset.
        c_delay[0] <= c_readout;
        for (i = 1; i < D_LATENCY; i = i + 1)
            c_delay[i] <= c_delay[i-1];
        total_readout <= $signed(c_delay[D_LATENCY-1]) + $signed(d_readout);
        if (rst) begin
            valid_pipe <= '0;
            out_valid <= 1'b0;
        end else begin
            valid_pipe <= {valid_pipe[D_LATENCY-2:0], read_valid};
            out_valid <= valid_pipe[D_LATENCY-1];
        end
    end
endmodule
`default_nettype wire
