`timescale 1ns / 1ps
`default_nettype none

// Four Q24 SSM readouts -> four signed INT8 out_proj inputs.
// ssm_requant_mult_48x16 must be signed 48x16 -> signed 64, latency=3.
module ssm_requant_stream (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [9:0]           in_record_addr,
    input  wire [5:0]           in_channel_base,
    input  wire [191:0]         in_y_q24,
    input  wire signed [15:0]   cfg_multiplier,
    input  wire signed [6:0]    cfg_shift,
    output reg                  out_valid,
    output reg [9:0]            out_record_addr,
    output reg [5:0]            out_channel_base,
    output reg [31:0]           out_data
);
    wire signed [63:0] product [0:3];
    genvar lane;
    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : GEN_REQUANT
            ssm_requant_mult_48x16 u_multiplier (
                .CLK(clk),
                .A(in_y_q24[lane*48 +: 48]),
                .B(cfg_multiplier),
                .P(product[lane])
            );
        end
    endgenerate

    reg valid_d0, valid_d1, valid_d2;
    reg [9:0] record_d0, record_d1, record_d2;
    reg [5:0] channel_d0, channel_d1, channel_d2;

    function [7:0] round_shift_clip_int8;
        input signed [63:0] value;
        input signed [6:0] shift_value;
        reg signed [64:0] magnitude;
        reg signed [64:0] rounded;
        reg signed [64:0] signed_rounded;
        begin
            magnitude = (value < 0) ? -$signed({value[63], value})
                                    :  $signed({1'b0, value});
            if (shift_value > 0)
                rounded = (magnitude + (65'sd1 <<< (shift_value-1)))
                          >>> shift_value;
            else if (shift_value < 0)
                rounded = magnitude <<< (-shift_value);
            else
                rounded = magnitude;
            signed_rounded = (value < 0) ? -rounded : rounded;

            if (signed_rounded > 65'sd127)
                round_shift_clip_int8 = 8'h7f;
            else if (signed_rounded < -65'sd128)
                round_shift_clip_int8 = 8'h80;
            else
                round_shift_clip_int8 = signed_rounded[7:0];
        end
    endfunction

    integer output_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            valid_d0 <= 1'b0;
            valid_d1 <= 1'b0;
            valid_d2 <= 1'b0;
            out_valid <= 1'b0;
        end else begin
            valid_d0 <= in_valid;
            valid_d1 <= valid_d0;
            valid_d2 <= valid_d1;
            out_valid <= valid_d2;
        end

        // Metadata and arithmetic payload are free-running pipelines.  Their
        // values during invalid cycles are don't-care; only the four valid
        // bits above participate in frame reset.  This removes frame_start
        // and valid-derived CE networks from the wide post-SSM datapath.
        record_d0 <= in_record_addr;
        record_d1 <= record_d0;
        record_d2 <= record_d1;
        channel_d0 <= in_channel_base;
        channel_d1 <= channel_d0;
        channel_d2 <= channel_d1;
        out_record_addr <= record_d2;
        out_channel_base <= channel_d2;
        for (output_lane = 0; output_lane < 4;
             output_lane = output_lane + 1) begin
            out_data[output_lane*8 +: 8]
                <= round_shift_clip_int8(product[output_lane], cfg_shift);
        end
    end
endmodule

`default_nettype wire
