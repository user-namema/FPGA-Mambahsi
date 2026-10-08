`timescale 1ns / 1ps
`default_nettype none

// Streaming depthwise Conv1d for block 0.
// PyTorch kernel_size=4, padding=3 creates L+3 outputs, but the deployed model
// immediately crops to the first L.  Those retained samples are causal:
// y[t] = w0*x[t-3] + w1*x[t-2] + w2*x[t-1] + w3*x[t] + bias.
// Therefore one 8-channel word can be accepted and produced every cycle.

// One byte lane owns its complete causal history.  Keeping the valid/token
// decode at this hierarchy boundary prevents a branch-wide CE/reset network
// from reaching every history byte and activation-window register.  History
// payload is deliberately not reset: pixel/token zero supplies the exact
// causal boundary before any payload is consumed.
(* KEEP_HIERARCHY = "yes" *)
module mamba_spa_history_lane (
    input  wire       clk,
    input  wire       valid,
    input  wire       pixel_zero,
    input  wire [2:0] channel_bank,
    input  wire [7:0] sample,
    output reg  [31:0] window
);
    reg [7:0] history_1 [0:7];
    reg [7:0] history_2 [0:7];
    reg [7:0] history_3 [0:7];
    always @(posedge clk) begin
        if (valid) begin
            window[7:0]   <= pixel_zero ? 8'd0 : history_3[channel_bank];
            window[15:8]  <= pixel_zero ? 8'd0 : history_2[channel_bank];
            window[23:16] <= pixel_zero ? 8'd0 : history_1[channel_bank];
            window[31:24] <= sample;
            history_3[channel_bank] <= pixel_zero ? 8'd0
                                                   : history_2[channel_bank];
            history_2[channel_bank] <= pixel_zero ? 8'd0
                                                   : history_1[channel_bank];
            history_1[channel_bank] <= sample;
        end
    end
endmodule

(* KEEP_HIERARCHY = "yes" *)
module mamba_spe_history_lane (
    input  wire       clk,
    input  wire       valid,
    input  wire       token_zero,
    input  wire       channel_bank,
    input  wire [7:0] sample,
    output reg  [31:0] window
);
    reg [7:0] history_1 [0:1];
    reg [7:0] history_2 [0:1];
    reg [7:0] history_3 [0:1];
    always @(posedge clk) begin
        if (valid) begin
            window[7:0]   <= token_zero ? 8'd0 : history_3[channel_bank];
            window[15:8]  <= token_zero ? 8'd0 : history_2[channel_bank];
            window[23:16] <= token_zero ? 8'd0 : history_1[channel_bank];
            window[31:24] <= sample;
            history_3[channel_bank] <= token_zero ? 8'd0
                                                   : history_2[channel_bank];
            history_2[channel_bank] <= token_zero ? 8'd0
                                                   : history_1[channel_bank];
            history_1[channel_bank] <= sample;
        end
    end
endmodule

module spa_conv1d_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] REQUANT_MULTIPLIER=16'sd32020,
    parameter signed [6:0] REQUANT_SHIFT=7'sd18
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [7:0]           in_pixel_addr,
    input  wire [5:0]           in_channel_base,
    input  wire [63:0]          in_data,

    input  wire                 cfg_weight_we,
    input  wire [7:0]           cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [5:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [31:0]   cfg_requant_multiplier,
    input  wire signed [6:0]    cfg_requant_shift,

    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [5:0]            out_channel_base,
    output reg [63:0]           out_raw_data,
    output reg [63:0]           out_data,
    output reg                  done
);

    // Block0 is physically far from the parallel in_proj service.  Register
    // its complete ingress beat once at the Conv boundary so channel/address
    // decode is generated locally instead of fanning out from in_proj across
    // the device.  Other blocks retain their existing latency.
    reg                         b0_in_valid_q;
    reg [7:0]                   b0_in_pixel_addr_q;
    (* keep = "true", max_fanout = 16 *) reg [5:0] b0_in_channel_base_q;
    reg [63:0]                  b0_in_data_q;
    always @(posedge clk) begin
        if (!rst_n || frame_start)
            b0_in_valid_q <= 1'b0;
        else
            b0_in_valid_q <= in_valid;

        // Payload has no reset/CE dependency on frame_start.  Its contents
        // are consumed only when b0_in_valid_q is asserted.
        // Ingress payload free-runs.  b0_in_valid_q is the only qualifier,
        // removing the former 78-bit payload CE rooted at in_valid.
        b0_in_pixel_addr_q <= in_pixel_addr;
        b0_in_channel_base_q <= in_channel_base;
        b0_in_data_q <= in_data;
    end

    wire spa_issue_valid = (BLOCK_ID == 0) ? b0_in_valid_q : in_valid;
    wire [7:0] spa_issue_pixel_addr =
        (BLOCK_ID == 0) ? b0_in_pixel_addr_q : in_pixel_addr;
    wire [5:0] spa_issue_channel_base =
        (BLOCK_ID == 0) ? b0_in_channel_base_q : in_channel_base;
    wire [63:0] spa_issue_data =
        (BLOCK_ID == 0) ? b0_in_data_q : in_data;

    wire [511:0] param_word;
    mamba_spa_conv_param_rom #(.BLOCK_ID(BLOCK_ID)) u_param_rom (
        .clk(clk),.en(1'b1),
        .addr(spa_issue_channel_base[5:3]),.data(param_word));
    reg rom_valid;
    reg [7:0] rom_pixel;
    reg [5:0] rom_channel;
    wire signed [7:0] activation_window [0:31];
    // frame_start clears stream control only.  History is datapath payload;
    // pixel zero already supplies the exact causal boundary for every frame.
    always @(posedge clk) begin
        if (!rst_n) begin
            rom_valid <= 1'b0;
            rom_pixel <= 8'd0;
            rom_channel <= 6'd0;
        end else if (frame_start) begin
            rom_valid <= 1'b0;
            rom_pixel <= 8'd0;
            rom_channel <= 6'd0;
        end else if (spa_issue_valid) begin
            rom_valid <= 1'b1;
            rom_pixel <= spa_issue_pixel_addr;
            rom_channel <= spa_issue_channel_base;
        end else rom_valid <= 1'b0;
    end

    genvar history_lane;
    generate for(history_lane=0;history_lane<8;
                 history_lane=history_lane+1)begin:G_LOCAL_SPA_HISTORY_LANE
        wire[31:0] lane_window;
        mamba_spa_history_lane u_history_lane(
            .clk(clk),.valid(spa_issue_valid),
            .pixel_zero(spa_issue_pixel_addr==0),
            .channel_bank(spa_issue_channel_base[5:3]),
            .sample(spa_issue_data[history_lane*8 +: 8]),
            .window(lane_window));
        assign activation_window[history_lane*4+0]=lane_window[7:0];
        assign activation_window[history_lane*4+1]=lane_window[15:8];
        assign activation_window[history_lane*4+2]=lane_window[23:16];
        assign activation_window[history_lane*4+3]=lane_window[31:24];
    end endgenerate

    // 8 channels x 4 taps = 32 signed 8x8, latency-3 multipliers.
    wire signed [15:0] product [0:31];
    genvar lane;
    genvar tap;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : GEN_SPA_CONV_LANE
            for (tap = 0; tap < 4; tap = tap + 1) begin : GEN_SPA_CONV_TAP
                mamba_mult_8x8 u_mult (
                    .CLK(clk),
                    .A(activation_window[lane*4+tap]),
                    .B(param_word[(lane*4+tap)*8 +: 8]),
                    .P(product[lane*4 + tap])
                );
            end
        end
    endgenerate

    reg mult_valid_d0, mult_valid_d1, mult_valid_d2;
    reg [7:0] mult_pixel_d0, mult_pixel_d1, mult_pixel_d2;
    reg [5:0] mult_channel_d0, mult_channel_d1, mult_channel_d2;
    reg [255:0] bias_d0,bias_d1,bias_d2;
    always @(posedge clk) begin
        if (!rst_n) begin
            mult_valid_d0 <= 1'b0; mult_valid_d1 <= 1'b0; mult_valid_d2 <= 1'b0;
            mult_pixel_d0 <= 8'd0; mult_pixel_d1 <= 8'd0; mult_pixel_d2 <= 8'd0;
            mult_channel_d0 <= 6'd0; mult_channel_d1 <= 6'd0; mult_channel_d2 <= 6'd0;
            bias_d0<=0;bias_d1<=0;bias_d2<=0;
        end else begin
            if (frame_start) begin
                mult_valid_d0 <= 1'b0; mult_valid_d1 <= 1'b0; mult_valid_d2 <= 1'b0;
                mult_pixel_d0 <= 8'd0; mult_pixel_d1 <= 8'd0; mult_pixel_d2 <= 8'd0;
                mult_channel_d0 <= 6'd0; mult_channel_d1 <= 6'd0; mult_channel_d2 <= 6'd0;
            end else begin
                mult_valid_d0 <= rom_valid;
                mult_valid_d1 <= mult_valid_d0;
                mult_valid_d2 <= mult_valid_d1;
                if (rom_valid) begin mult_pixel_d0 <= rom_pixel; mult_channel_d0 <= rom_channel; end
                if (mult_valid_d0) begin mult_pixel_d1 <= mult_pixel_d0; mult_channel_d1 <= mult_channel_d0; end
                if (mult_valid_d1) begin mult_pixel_d2 <= mult_pixel_d1; mult_channel_d2 <= mult_channel_d1; end
            end
            // Bias is payload and is deliberately independent of frame_start.
            if (rom_valid) bias_d0 <= param_word[511:256];
            if (mult_valid_d0) bias_d1 <= bias_d0;
            if (mult_valid_d1) bias_d2 <= bias_d1;
        end
    end

    reg signed [16:0] sum_l1 [0:15];
    reg signed [17:0] sum_l2 [0:7];
    reg signed [31:0] accumulator [0:7];
    reg valid_l1, valid_l2, valid_acc;
    reg [7:0] pixel_l1, pixel_l2, pixel_acc;
    reg [5:0] channel_l1, channel_l2, channel_acc;
    reg [255:0] bias_l1,bias_l2;
    integer add_lane;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_l1 <= 1'b0; valid_l2 <= 1'b0; valid_acc <= 1'b0;
            pixel_l1 <= 8'd0; pixel_l2 <= 8'd0; pixel_acc <= 8'd0;
            channel_l1 <= 6'd0; channel_l2 <= 6'd0; channel_acc <= 6'd0;
            bias_l1<=0;bias_l2<=0;
        end else begin
            if (frame_start) begin
                valid_l1 <= 1'b0; valid_l2 <= 1'b0; valid_acc <= 1'b0;
                pixel_l1 <= 8'd0; pixel_l2 <= 8'd0; pixel_acc <= 8'd0;
                channel_l1 <= 6'd0; channel_l2 <= 6'd0; channel_acc <= 6'd0;
            end else begin
                valid_l1 <= mult_valid_d2;
                valid_l2 <= valid_l1;
                valid_acc <= valid_l2;
            end
            if (mult_valid_d2) begin
                if (!frame_start) begin pixel_l1 <= mult_pixel_d2; channel_l1 <= mult_channel_d2; end
                bias_l1<=bias_d2;
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1) begin
                    sum_l1[add_lane*2]
                        <= $signed({product[add_lane*4][15], product[add_lane*4]})
                         + $signed({product[add_lane*4+1][15], product[add_lane*4+1]});
                    sum_l1[add_lane*2+1]
                        <= $signed({product[add_lane*4+2][15], product[add_lane*4+2]})
                         + $signed({product[add_lane*4+3][15], product[add_lane*4+3]});
                end
            end
            if (valid_l1) begin
                if (!frame_start) begin pixel_l2 <= pixel_l1; channel_l2 <= channel_l1; end
                bias_l2<=bias_l1;
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1)
                    sum_l2[add_lane]
                        <= $signed({sum_l1[add_lane*2][16], sum_l1[add_lane*2]})
                         + $signed({sum_l1[add_lane*2+1][16], sum_l1[add_lane*2+1]});
            end
            if (valid_l2) begin
                if (!frame_start) begin pixel_acc <= pixel_l2; channel_acc <= channel_l2; end
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1)
                    accumulator[add_lane]
                        <= {{14{sum_l2[add_lane][17]}}, sum_l2[add_lane]}
                         + $signed(bias_l2[add_lane*32 +:32]);
            end
        end
    end

    wire signed [36:0] requant_product [0:7];
    genvar rq_lane;
    generate
        for (rq_lane = 0; rq_lane < 8; rq_lane = rq_lane + 1) begin : GEN_SPA_CONV_RQ
            requant_mult_21x16 u_requant (
                .CLK(clk), .A(accumulator[rq_lane][20:0]),
                .B(REQUANT_MULTIPLIER), .P(requant_product[rq_lane])
            );
        end
    endgenerate

    reg rq_valid_d0, rq_valid_d1, rq_valid_d2;
    reg [7:0] rq_pixel_d0, rq_pixel_d1, rq_pixel_d2;
    reg [5:0] rq_channel_d0, rq_channel_d1, rq_channel_d2;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_valid_d0 <= 1'b0; rq_valid_d1 <= 1'b0; rq_valid_d2 <= 1'b0;
            rq_pixel_d0 <= 8'd0; rq_pixel_d1 <= 8'd0; rq_pixel_d2 <= 8'd0;
            rq_channel_d0 <= 6'd0; rq_channel_d1 <= 6'd0; rq_channel_d2 <= 6'd0;
        end else begin
            rq_valid_d0 <= valid_acc; rq_valid_d1 <= rq_valid_d0; rq_valid_d2 <= rq_valid_d1;
            if (valid_acc) begin rq_pixel_d0 <= pixel_acc; rq_channel_d0 <= channel_acc; end
            if (rq_valid_d0) begin rq_pixel_d1 <= rq_pixel_d0; rq_channel_d1 <= rq_channel_d0; end
            if (rq_valid_d1) begin rq_pixel_d2 <= rq_pixel_d1; rq_channel_d2 <= rq_channel_d1; end
        end
    end

    function [7:0] round_shift_clip_int8;
        input signed [63:0] product_value;
        input signed [6:0] shift_value;
        reg signed [63:0] magnitude, half_lsb, rounded;
        begin
            if (shift_value > 0) begin
                half_lsb = 64'sd1 <<< (shift_value - 1'b1);
                magnitude = (product_value < 0) ? -product_value : product_value;
                rounded = (magnitude + half_lsb) >>> shift_value;
                if (product_value < 0) rounded = -rounded;
            end else if (shift_value < 0)
                rounded = product_value <<< (-shift_value);
            else
                rounded = product_value;
            if (rounded > 64'sd127) round_shift_clip_int8 = 8'h7f;
            else if (rounded < -64'sd128) round_shift_clip_int8 = 8'h80;
            else round_shift_clip_int8 = rounded[7:0];
        end
    endfunction

    integer output_lane;
    reg [7:0] raw_value;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0; out_pixel_addr <= 8'd0; out_channel_base <= 6'd0;
            out_raw_data <= 64'd0; out_data <= 64'd0; done <= 1'b0;
        end else begin
            if (frame_start) begin
                out_valid <= 1'b0; out_pixel_addr <= 8'd0; out_channel_base <= 6'd0;
                done <= 1'b0;
            end else begin
                out_valid <= rq_valid_d2;
                done <= 1'b0;
            end
            if (rq_valid_d2) begin
                if (!frame_start) begin
                    out_pixel_addr <= rq_pixel_d2;
                    out_channel_base <= rq_channel_d2;
                end
                for (output_lane = 0; output_lane < 8; output_lane = output_lane + 1) begin
                    raw_value = round_shift_clip_int8(requant_product[output_lane], REQUANT_SHIFT);
                    out_raw_data[output_lane*8 +: 8] <= raw_value;
                    out_data[output_lane*8 +: 8] <= raw_value[7] ? 8'd0 : raw_value;
                end
                if (!frame_start && (rq_pixel_d2 == PIXEL_COUNT-1)
                    && (rq_channel_d2 == 6'd56)) done <= 1'b1;
            end
        end
    end

endmodule


module spe_conv1d_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] REQUANT_MULTIPLIER=16'sd24383,
    parameter signed [6:0] REQUANT_SHIFT=7'sd18
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [7:0]           in_pixel_addr,
    input  wire [1:0]           in_token,
    input  wire [3:0]           in_channel_base,
    input  wire [63:0]          in_data,

    input  wire                 cfg_weight_we,
    input  wire [5:0]           cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [3:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [31:0]   cfg_requant_multiplier,
    input  wire signed [6:0]    cfg_requant_shift,

    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [1:0]            out_token,
    output reg [3:0]            out_channel_base,
    output reg [63:0]           out_raw_data,
    output reg [63:0]           out_data,
    output reg                  done
);

    wire [511:0] param_word;
    mamba_spe_conv_param_rom #(.BLOCK_ID(BLOCK_ID)) u_param_rom (
        .clk(clk),.en(1'b1),.addr(in_channel_base[3]),.data(param_word));
    reg rom_valid;
    reg [7:0] rom_pixel;
    reg [1:0] rom_token;
    reg [3:0] rom_channel;
    wire signed [7:0] activation_window [0:31];
    // Token zero is the branch-local causal boundary.  Do not distribute
    // frame_start into the 16x3 history payload registers.
    always @(posedge clk) begin
        if (!rst_n) begin
            rom_valid<=1'b0;rom_pixel<=0;rom_token<=0;rom_channel<=0;
        end else if (frame_start) begin
            rom_valid<=1'b0;rom_pixel<=0;rom_token<=0;rom_channel<=0;
        end else if (in_valid) begin
            rom_valid<=1'b1;rom_pixel<=in_pixel_addr;rom_token<=in_token;rom_channel<=in_channel_base;
        end else rom_valid<=1'b0;
    end

    genvar spe_history_lane;
    generate for(spe_history_lane=0;spe_history_lane<8;
                 spe_history_lane=spe_history_lane+1)begin:G_LOCAL_SPE_HISTORY_LANE
        wire[31:0] lane_window;
        mamba_spe_history_lane u_history_lane(
            .clk(clk),.valid(in_valid),.token_zero(in_token==0),
            .channel_bank(in_channel_base[3]),
            .sample(in_data[spe_history_lane*8 +: 8]),
            .window(lane_window));
        assign activation_window[spe_history_lane*4+0]=lane_window[7:0];
        assign activation_window[spe_history_lane*4+1]=lane_window[15:8];
        assign activation_window[spe_history_lane*4+2]=lane_window[23:16];
        assign activation_window[spe_history_lane*4+3]=lane_window[31:24];
    end endgenerate

    wire signed [15:0] product [0:31];
    genvar lane;
    genvar tap;
    generate
        for (lane = 0; lane < 8; lane = lane + 1) begin : GEN_SPE_CONV_LANE
            for (tap = 0; tap < 4; tap = tap + 1) begin : GEN_SPE_CONV_TAP
                mamba_mult_8x8 u_mult (
                    .CLK(clk), .A(activation_window[lane*4+tap]),
                    .B(param_word[(lane*4+tap)*8 +:8]),
                    .P(product[lane*4 + tap])
                );
            end
        end
    endgenerate

    reg mult_valid_d0, mult_valid_d1, mult_valid_d2;
    reg [7:0] mult_pixel_d0, mult_pixel_d1, mult_pixel_d2;
    reg [1:0] mult_token_d0, mult_token_d1, mult_token_d2;
    reg [3:0] mult_channel_d0, mult_channel_d1, mult_channel_d2;
    reg [255:0] spe_bias_d0,spe_bias_d1,spe_bias_d2;
    always @(posedge clk) begin
        if (!rst_n) begin
            mult_valid_d0 <= 1'b0; mult_valid_d1 <= 1'b0; mult_valid_d2 <= 1'b0;
            mult_pixel_d0 <= 8'd0; mult_pixel_d1 <= 8'd0; mult_pixel_d2 <= 8'd0;
            mult_token_d0 <= 2'd0; mult_token_d1 <= 2'd0; mult_token_d2 <= 2'd0;
            mult_channel_d0 <= 4'd0; mult_channel_d1 <= 4'd0; mult_channel_d2 <= 4'd0;
            spe_bias_d0<=0;spe_bias_d1<=0;spe_bias_d2<=0;
        end else begin
            if (frame_start) begin
                mult_valid_d0 <= 1'b0; mult_valid_d1 <= 1'b0; mult_valid_d2 <= 1'b0;
                mult_pixel_d0 <= 8'd0; mult_pixel_d1 <= 8'd0; mult_pixel_d2 <= 8'd0;
                mult_token_d0 <= 2'd0; mult_token_d1 <= 2'd0; mult_token_d2 <= 2'd0;
                mult_channel_d0 <= 4'd0; mult_channel_d1 <= 4'd0; mult_channel_d2 <= 4'd0;
            end else begin
                mult_valid_d0 <= rom_valid; mult_valid_d1 <= mult_valid_d0; mult_valid_d2 <= mult_valid_d1;
                if (rom_valid) begin mult_pixel_d0 <= rom_pixel; mult_token_d0 <= rom_token; mult_channel_d0 <= rom_channel; end
                if (mult_valid_d0) begin mult_pixel_d1 <= mult_pixel_d0; mult_token_d1 <= mult_token_d0; mult_channel_d1 <= mult_channel_d0; end
                if (mult_valid_d1) begin mult_pixel_d2 <= mult_pixel_d1; mult_token_d2 <= mult_token_d1; mult_channel_d2 <= mult_channel_d1; end
            end
            // Keep the 256-bit bias payload off the frame_start network.
            if (rom_valid) spe_bias_d0 <= param_word[511:256];
            if (mult_valid_d0) spe_bias_d1 <= spe_bias_d0;
            if (mult_valid_d1) spe_bias_d2 <= spe_bias_d1;
        end
    end

    reg signed [16:0] sum_l1 [0:15];
    reg signed [17:0] sum_l2 [0:7];
    reg signed [31:0] accumulator [0:7];
    reg valid_l1, valid_l2, valid_acc;
    reg [7:0] pixel_l1, pixel_l2, pixel_acc;
    reg [1:0] token_l1, token_l2, token_acc;
    reg [3:0] channel_l1, channel_l2, channel_acc;
    reg [255:0] spe_bias_l1,spe_bias_l2;
    integer add_lane;
    always @(posedge clk) begin
        if (!rst_n) begin
            valid_l1 <= 1'b0; valid_l2 <= 1'b0; valid_acc <= 1'b0;
            pixel_l1 <= 8'd0; pixel_l2 <= 8'd0; pixel_acc <= 8'd0;
            token_l1 <= 2'd0; token_l2 <= 2'd0; token_acc <= 2'd0;
            channel_l1 <= 4'd0; channel_l2 <= 4'd0; channel_acc <= 4'd0;
            spe_bias_l1<=0;spe_bias_l2<=0;
        end else begin
            if (frame_start) begin
                valid_l1 <= 1'b0; valid_l2 <= 1'b0; valid_acc <= 1'b0;
                pixel_l1 <= 8'd0; pixel_l2 <= 8'd0; pixel_acc <= 8'd0;
                token_l1 <= 2'd0; token_l2 <= 2'd0; token_acc <= 2'd0;
                channel_l1 <= 4'd0; channel_l2 <= 4'd0; channel_acc <= 4'd0;
            end else begin
                valid_l1 <= mult_valid_d2; valid_l2 <= valid_l1; valid_acc <= valid_l2;
            end
            if (mult_valid_d2) begin
                if (!frame_start) begin pixel_l1 <= mult_pixel_d2; token_l1 <= mult_token_d2; channel_l1 <= mult_channel_d2; end
                spe_bias_l1<=spe_bias_d2;
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1) begin
                    sum_l1[add_lane*2]
                        <= $signed({product[add_lane*4][15], product[add_lane*4]})
                         + $signed({product[add_lane*4+1][15], product[add_lane*4+1]});
                    sum_l1[add_lane*2+1]
                        <= $signed({product[add_lane*4+2][15], product[add_lane*4+2]})
                         + $signed({product[add_lane*4+3][15], product[add_lane*4+3]});
                end
            end
            if (valid_l1) begin
                if (!frame_start) begin pixel_l2 <= pixel_l1; token_l2 <= token_l1; channel_l2 <= channel_l1; end
                spe_bias_l2<=spe_bias_l1;
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1)
                    sum_l2[add_lane]
                        <= $signed({sum_l1[add_lane*2][16], sum_l1[add_lane*2]})
                         + $signed({sum_l1[add_lane*2+1][16], sum_l1[add_lane*2+1]});
            end
            if (valid_l2) begin
                if (!frame_start) begin pixel_acc <= pixel_l2; token_acc <= token_l2; channel_acc <= channel_l2; end
                for (add_lane = 0; add_lane < 8; add_lane = add_lane + 1)
                    accumulator[add_lane]
                        <= {{14{sum_l2[add_lane][17]}}, sum_l2[add_lane]}
                         + $signed(spe_bias_l2[add_lane*32 +:32]);
            end
        end
    end

    wire signed [36:0] requant_product [0:7];
    genvar rq_lane;
    generate
        for (rq_lane = 0; rq_lane < 8; rq_lane = rq_lane + 1) begin : GEN_SPE_CONV_RQ
            requant_mult_21x16 u_requant (
                .CLK(clk), .A(accumulator[rq_lane][20:0]),
                .B(REQUANT_MULTIPLIER), .P(requant_product[rq_lane])
            );
        end
    endgenerate

    reg rq_valid_d0, rq_valid_d1, rq_valid_d2;
    reg [7:0] rq_pixel_d0, rq_pixel_d1, rq_pixel_d2;
    reg [1:0] rq_token_d0, rq_token_d1, rq_token_d2;
    reg [3:0] rq_channel_d0, rq_channel_d1, rq_channel_d2;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_valid_d0 <= 1'b0; rq_valid_d1 <= 1'b0; rq_valid_d2 <= 1'b0;
            rq_pixel_d0 <= 8'd0; rq_pixel_d1 <= 8'd0; rq_pixel_d2 <= 8'd0;
            rq_token_d0 <= 2'd0; rq_token_d1 <= 2'd0; rq_token_d2 <= 2'd0;
            rq_channel_d0 <= 4'd0; rq_channel_d1 <= 4'd0; rq_channel_d2 <= 4'd0;
        end else begin
            rq_valid_d0 <= valid_acc; rq_valid_d1 <= rq_valid_d0; rq_valid_d2 <= rq_valid_d1;
            if (valid_acc) begin rq_pixel_d0 <= pixel_acc; rq_token_d0 <= token_acc; rq_channel_d0 <= channel_acc; end
            if (rq_valid_d0) begin rq_pixel_d1 <= rq_pixel_d0; rq_token_d1 <= rq_token_d0; rq_channel_d1 <= rq_channel_d0; end
            if (rq_valid_d1) begin rq_pixel_d2 <= rq_pixel_d1; rq_token_d2 <= rq_token_d1; rq_channel_d2 <= rq_channel_d1; end
        end
    end

    function [7:0] round_shift_clip_int8;
        input signed [63:0] product_value;
        input signed [6:0] shift_value;
        reg signed [63:0] magnitude, half_lsb, rounded;
        begin
            if (shift_value > 0) begin
                half_lsb = 64'sd1 <<< (shift_value - 1'b1);
                magnitude = (product_value < 0) ? -product_value : product_value;
                rounded = (magnitude + half_lsb) >>> shift_value;
                if (product_value < 0) rounded = -rounded;
            end else if (shift_value < 0)
                rounded = product_value <<< (-shift_value);
            else
                rounded = product_value;
            if (rounded > 64'sd127) round_shift_clip_int8 = 8'h7f;
            else if (rounded < -64'sd128) round_shift_clip_int8 = 8'h80;
            else round_shift_clip_int8 = rounded[7:0];
        end
    endfunction

    integer output_lane;
    reg [7:0] raw_value;
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0; out_pixel_addr <= 8'd0; out_token <= 2'd0;
            out_channel_base <= 4'd0; out_raw_data <= 64'd0; out_data <= 64'd0; done <= 1'b0;
        end else begin
            if (frame_start) begin
                out_valid <= 1'b0; out_pixel_addr <= 8'd0; out_token <= 2'd0;
                out_channel_base <= 4'd0; done <= 1'b0;
            end else begin
                out_valid <= rq_valid_d2;
                done <= 1'b0;
            end
            if (rq_valid_d2) begin
                if (!frame_start) begin
                    out_pixel_addr <= rq_pixel_d2;
                    out_token <= rq_token_d2;
                    out_channel_base <= rq_channel_d2;
                end
                for (output_lane = 0; output_lane < 8; output_lane = output_lane + 1) begin
                    raw_value = round_shift_clip_int8(requant_product[output_lane], REQUANT_SHIFT);
                    out_raw_data[output_lane*8 +: 8] <= raw_value;
                    out_data[output_lane*8 +: 8] <= raw_value[7] ? 8'd0 : raw_value;
                end
                if (!frame_start && (rq_pixel_d2 == PIXEL_COUNT-1) && (rq_token_d2 == 2'd3)
                    && (rq_channel_d2 == 4'd8)) done <= 1'b1;
            end
        end
    end

endmodule


module both_conv1d_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] SPA_REQUANT_MULTIPLIER=16'sd32020,
    parameter signed [6:0]  SPA_REQUANT_SHIFT=7'sd18,
    parameter signed [15:0] SPE_REQUANT_MULTIPLIER=16'sd24383,
    parameter signed [6:0]  SPE_REQUANT_SHIFT=7'sd18
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 spa_frame_start,
    input  wire                 spe_frame_start,

    input  wire                 spa_in_valid,
    input  wire [7:0]           spa_in_pixel_addr,
    input  wire [5:0]           spa_in_channel_base,
    input  wire [63:0]          spa_in_data,
    input  wire                 spe_in_valid,
    input  wire [7:0]           spe_in_pixel_addr,
    input  wire [1:0]           spe_in_token,
    input  wire [3:0]           spe_in_channel_base,
    input  wire [63:0]          spe_in_data,

    input  wire                 spa_weight_we,
    input  wire [7:0]           spa_weight_addr,
    input  wire signed [7:0]    spa_weight_data,
    input  wire                 spa_bias_we,
    input  wire [5:0]           spa_bias_addr,
    input  wire signed [31:0]   spa_bias_data,
    input  wire signed [31:0]   spa_requant_multiplier,
    input  wire signed [6:0]    spa_requant_shift,
    input  wire                 spe_weight_we,
    input  wire [5:0]           spe_weight_addr,
    input  wire signed [7:0]    spe_weight_data,
    input  wire                 spe_bias_we,
    input  wire [3:0]           spe_bias_addr,
    input  wire signed [31:0]   spe_bias_data,
    input  wire signed [31:0]   spe_requant_multiplier,
    input  wire signed [6:0]    spe_requant_shift,

    output wire                 spa_out_valid,
    output wire [7:0]           spa_out_pixel_addr,
    output wire [5:0]           spa_out_channel_base,
    output wire [63:0]          spa_out_raw_data,
    output wire [63:0]          spa_out_data,
    output wire                 spa_done,
    output wire                 spe_out_valid,
    output wire [7:0]           spe_out_pixel_addr,
    output wire [1:0]           spe_out_token,
    output wire [3:0]           spe_out_channel_base,
    output wire [63:0]          spe_out_raw_data,
    output wire [63:0]          spe_out_data,
    output wire                 spe_done
);

    spa_conv1d_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),
        .REQUANT_MULTIPLIER(SPA_REQUANT_MULTIPLIER),
        .REQUANT_SHIFT(SPA_REQUANT_SHIFT)) u_spa_conv1d (
        .clk(clk), .rst_n(rst_n), .frame_start(spa_frame_start),
        .in_valid(spa_in_valid), .in_pixel_addr(spa_in_pixel_addr),
        .in_channel_base(spa_in_channel_base), .in_data(spa_in_data),
        .cfg_weight_we(spa_weight_we), .cfg_weight_addr(spa_weight_addr),
        .cfg_weight_data(spa_weight_data), .cfg_bias_we(spa_bias_we),
        .cfg_bias_addr(spa_bias_addr), .cfg_bias_data(spa_bias_data),
        .cfg_requant_multiplier(spa_requant_multiplier), .cfg_requant_shift(spa_requant_shift),
        .out_valid(spa_out_valid), .out_pixel_addr(spa_out_pixel_addr),
        .out_channel_base(spa_out_channel_base), .out_raw_data(spa_out_raw_data),
        .out_data(spa_out_data), .done(spa_done)
    );

    spe_conv1d_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),
        .REQUANT_MULTIPLIER(SPE_REQUANT_MULTIPLIER),
        .REQUANT_SHIFT(SPE_REQUANT_SHIFT)) u_spe_conv1d (
        .clk(clk), .rst_n(rst_n), .frame_start(spe_frame_start),
        .in_valid(spe_in_valid), .in_pixel_addr(spe_in_pixel_addr),
        .in_token(spe_in_token), .in_channel_base(spe_in_channel_base), .in_data(spe_in_data),
        .cfg_weight_we(spe_weight_we), .cfg_weight_addr(spe_weight_addr),
        .cfg_weight_data(spe_weight_data), .cfg_bias_we(spe_bias_we),
        .cfg_bias_addr(spe_bias_addr), .cfg_bias_data(spe_bias_data),
        .cfg_requant_multiplier(spe_requant_multiplier), .cfg_requant_shift(spe_requant_shift),
        .out_valid(spe_out_valid), .out_pixel_addr(spe_out_pixel_addr),
        .out_token(spe_out_token), .out_channel_base(spe_out_channel_base),
        .out_raw_data(spe_out_raw_data), .out_data(spe_out_data), .done(spe_done)
    );

endmodule

`default_nettype wire
