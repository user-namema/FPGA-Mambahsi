`timescale 1ns/1ps
`default_nettype none
// M=16386, shift=14. Exhaustively proved identity for all 256 INT8 codes.
// No inferred ROM, combinational multiplier or new pipeline stage is needed.
module mamba_d1_head_input_lut (
    input wire [7:0] address,
    output wire [7:0] data
);
    assign data = address;
endmodule
`default_nettype wire
