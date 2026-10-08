`timescale 1ns/1ps
// Delay operands BEFORE state RAM access, not after it. Recurrence spacing,
// state writeback and the original forwarding network remain unchanged.
module nf_align #(
    parameter integer WIDTH = 1,
    parameter integer DEPTH = 0
) (
    input wire clk, reset, in_valid,
    input wire [WIDTH-1:0] in_data,
    output wire out_valid, pending,
    output wire [WIDTH-1:0] out_data
);
    generate if (DEPTH == 0) begin : G_BYPASS
        assign out_data = in_data;
        assign out_valid = in_valid;
        assign pending = 1'b0;
    end else begin : G_PIPE
        reg [WIDTH-1:0] payload [0:DEPTH-1];
        reg [DEPTH-1:0] valid_q;
        integer i;
        always @(posedge clk) begin
            payload[0] <= in_data;
            for (i=1; i<DEPTH; i=i+1)
                payload[i] <= payload[i-1];
            if (reset)
                valid_q <= 0;
            else begin
                valid_q[0] <= in_valid;
                for (i=1; i<DEPTH; i=i+1)
                    valid_q[i] <= valid_q[i-1];
            end
        end
        assign out_data = payload[DEPTH-1];
        assign out_valid = valid_q[DEPTH-1];
        assign pending = |valid_q;
    end endgenerate
endmodule
