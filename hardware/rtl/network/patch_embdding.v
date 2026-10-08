`timescale 1ns / 1ps
`default_nettype none

// Two signed INT8 weights share one unsigned INT8 activation in one explicit
// three-cycle DSP Macro IP.  The IP is configured for (A+D)*B so the packed
// weight addition stays in the DSP48E2 pre-adder instead of a LUT/CARRY tree.
//
// P[17:0] is the low product. P[35:18] is the high product minus one whenever
// P[17] is set. The enclosing dot-product tree applies that correction once at
// its final reduction stage.
(* keep_hierarchy = "yes" *)
module patch_packed_mult_2x8_dsp (
    input  wire                    clk,
    input  wire signed [7:0]       weight_low,
    input  wire signed [7:0]       weight_high,
    input  wire        [7:0]       activation,
    output wire signed [47:0]      packed_product
);
    wire signed [26:0] dsp_a
        = $signed({weight_high[7], weight_high, 18'b0});
    wire signed [26:0] dsp_d
        = $signed({{19{weight_low[7]}}, weight_low});
    wire signed [8:0] dsp_b = $signed({1'b0, activation});
    wire signed [35:0] dsp_p;

    patch_packed_dsp_3cyc u_patch_packed_dsp_3cyc (
        .CLK (clk),
        .A   (dsp_a),
        .B   (dsp_b),
        .D   (dsp_d),
        .P   (dsp_p)
    );

    assign packed_product = {{12{dsp_p[35]}}, dsp_p};
endmodule

// -----------------------------------------------------------------------------
// Patch embedding for the current dense16 software deployment.
//
// Software / hardware contract
//   input tile       : 16 x 16 pixels
//   input channels   : 16, unsigned INT8 (0 ... 255)
//   operation        : 1 x 1 convolution, 16 -> 32 channels
//   output boundary  : signed INT8 requantization followed by ReLU (0 ... 127)
//   quantization     : per-tensor multiplier and signed shift
//
// Parallel structure
//   - One pixel is stored in one logical 128-bit BRAM word.
//   - Four output channels are calculated in parallel.
//   - Every DSP48E2 computes two products with one shared activation.
//   - Therefore the dot-product core contains 2 x 16 = 32 DSP48E2 slices.
//   - Eight output groups are required for each pixel (32 / 4).
//
// Input word layout
//   tile_wr_data[8*c +: 8] = input channel c, c = 0 ... 15.
//
// Weight address layout
//   cfg_weight_addr = output_channel * 16 + input_channel.
//   All weights and biases must be written before start is asserted.
//
// Output word layout
//   out_data[ 7: 0] = output channel out_channel_base + 0
//   out_data[15: 8] = output channel out_channel_base + 1
//   out_data[23:16] = output channel out_channel_base + 2
//   out_data[31:24] = output channel out_channel_base + 3
//
// There is no output back-pressure in this first version.  The downstream
// module must accept one 32-bit word on every cycle for which out_valid is high.
// -----------------------------------------------------------------------------
module patch_embdding #(
    parameter signed [15:0] FIXED_REQUANT_MULTIPLIER =16'sd28478,
    parameter signed [6:0]  FIXED_REQUANT_SHIFT =7'sd19
) (
    input  wire                     clk,
    input  wire                     rst_n,

    // 256 x 128-bit logical input BRAM write port.
    input  wire                     tile_wr_en,
    input  wire [7:0]               tile_wr_addr,
    input  wire [127:0]             tile_wr_data,

    // Dynamic INT8 weight loading. Address = oc * 16 + ic.
    input  wire                     cfg_weight_we,
    input  wire [8:0]               cfg_weight_addr,
    input  wire signed [7:0]        cfg_weight_data,

    // Bias is stored in the INT32 accumulator domain.
    input  wire                     cfg_bias_we,
    input  wire [4:0]               cfg_bias_addr,
    input  wire signed [31:0]       cfg_bias_data,

    // QAT per-tensor requantization parameters.
    input  wire signed [31:0]       cfg_requant_multiplier,
    input  wire signed [6:0]        cfg_requant_shift,

    input  wire                     start,
    output wire                     busy,
    output reg                      done,

    output reg                      out_valid,
    output reg  [7:0]               out_pixel_addr,
    output reg  [4:0]               out_channel_base,
    output reg  [31:0]              out_data
);

    localparam [2:0] ST_IDLE   = 3'd0;
    localparam [2:0] ST_PRIME0 = 3'd1;
    localparam [2:0] ST_PRIME1 = 3'd2;
    localparam [2:0] ST_RUN    = 3'd3;
    localparam [2:0] ST_DRAIN  = 3'd4;

    reg [2:0] state;

    // Vivado treats this as one logical 256 x 128-bit memory.  Because a
    // Xilinx BRAM primitive is narrower than 128 bits, implementation will
    // automatically use several physical BRAM primitives in parallel.
    (* ram_style = "block" *) reg [127:0] input_bram [0:255];
    reg [7:0] input_read_address;
    reg [127:0] input_bram_q;
    reg [127:0] current_pixel_data;

    // Fixed deployment parameters are baked into the block instance.  The
    // legacy cfg ports remain in the interface so the surrounding loader can
    // be removed incrementally; they are intentionally not used here.
    wire signed [15:0] requant_multiplier_fixed
        = FIXED_REQUANT_MULTIPLIER;
    wire signed [6:0] requant_shift_fixed = FIXED_REQUANT_SHIFT;

    reg [7:0] pixel_issue;
    reg [2:0] group_issue;

    wire issue_valid;
    assign issue_valid = (state == ST_RUN);
    assign busy = (state != ST_IDLE);

    // -------------------------------------------------------------------------
    // Memories and configuration registers
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (tile_wr_en)
            input_bram[tile_wr_addr] <= tile_wr_data;

        // Synchronous read port used by the compute pipeline.
        input_bram_q <= input_bram[input_read_address];

    end

    // Four row reads are supplied by two replicated true-dual-port ROMs.
    // Each address stores all 16 INT8 weights of one output channel.
    wire [127:0] weight_row0;
    wire [127:0] weight_row1;
    wire [127:0] weight_row2;
    wire [127:0] weight_row3;
    wire [127:0] bias_group_word;
    wire [4:0] rom_row_base = {group_issue, 2'b00};

    patch_weight_rom_32x128_copy01 u_patch_weight_rom_copy01 (
        .clka(clk), .ena(issue_valid), .addra(rom_row_base),
        .douta(weight_row0),
        .clkb(clk), .enb(issue_valid), .addrb(rom_row_base + 5'd1),
        .doutb(weight_row1)
    );

    patch_weight_rom_32x128_copy23 u_patch_weight_rom_copy23 (
        .clka(clk), .ena(issue_valid), .addra(rom_row_base + 5'd2),
        .douta(weight_row2),
        .clkb(clk), .enb(issue_valid), .addrb(rom_row_base + 5'd3),
        .doutb(weight_row3)
    );

    patch_bias_rom_8x128 u_patch_bias_rom (
        .clka(clk), .ena(issue_valid), .addra(group_issue),
        .douta(bias_group_word)
    );

    // BMG primitive/core output registers are disabled, therefore the ROM has
    // one synchronous read cycle.  Delay activation and metadata by one cycle
    // before presenting them to the multiplier bank.
    reg rom_valid;
    reg [7:0] rom_pixel;
    reg [2:0] rom_group;
    reg [127:0] rom_activation;
    always @(posedge clk) begin
        if (!rst_n) begin
            rom_valid <= 1'b0;
            rom_pixel <= 8'd0;
            rom_group <= 3'd0;
            rom_activation <= 128'd0;
        end else begin
            rom_valid <= issue_valid;
            if (issue_valid) begin
                rom_pixel <= pixel_issue;
                rom_group <= group_issue;
                rom_activation <= current_pixel_data;
            end
        end
    end

    // -------------------------------------------------------------------------
    // 32 DSP48E2 packed multipliers preserve the original 64-product/cycle
    // throughput. Each DSP handles two output channels sharing one activation;
    // the four existing INT8 weight-ROM rows remain unchanged.
    // -------------------------------------------------------------------------
    wire [7:0] input_channel [0:15];
    wire signed [16:0] product_mult [0:63];
    wire packed_low_negative [0:31];

    genvar input_index;
    generate
        for (input_index = 0; input_index < 16; input_index = input_index + 1) begin : GEN_INPUT_CHANNEL
            assign input_channel[input_index]
                = rom_activation[input_index*8 +: 8];
        end
    endgenerate

    genvar output_pair;
    genvar channel_index;
    generate
        for (output_pair = 0; output_pair < 2; output_pair = output_pair + 1) begin : GEN_OUTPUT_PAIR
            wire [127:0] selected_weight_low_row;
            wire [127:0] selected_weight_high_row;
            assign selected_weight_low_row
                = (output_pair == 0) ? weight_row0 : weight_row2;
            assign selected_weight_high_row
                = (output_pair == 0) ? weight_row1 : weight_row3;

            for (channel_index = 0; channel_index < 16; channel_index = channel_index + 1) begin : GEN_CHANNEL_MULTIPLIER
                wire signed [47:0] packed_product;

                patch_packed_mult_2x8_dsp u_patch_packed_mult (
                    .clk           (clk),
                    .weight_low    (selected_weight_low_row[channel_index*8 +: 8]),
                    .weight_high   (selected_weight_high_row[channel_index*8 +: 8]),
                    .activation    (input_channel[channel_index]),
                    .packed_product(packed_product)
                );

                assign product_mult[(output_pair*2)*16 + channel_index]
                    = packed_product[16:0];
                assign product_mult[(output_pair*2+1)*16 + channel_index]
                    = packed_product[34:18];
                assign packed_low_negative[output_pair*16 + channel_index]
                    = packed_product[17];
            end
        end
    endgenerate

    // Metadata delay aligned to the three registered multiplier stages.  The
    // downstream adder registers consume the IP output on the following edge.
    reg mult_valid_d0;
    reg mult_valid_d1;
    reg mult_valid_d2;
    reg [7:0] mult_pixel_d0;
    reg [7:0] mult_pixel_d1;
    reg [7:0] mult_pixel_d2;
    reg [2:0] mult_group_d0;
    reg [2:0] mult_group_d1;
    reg [2:0] mult_group_d2;

    always @(posedge clk) begin
        if (!rst_n) begin
            mult_valid_d0 <= 1'b0;
            mult_valid_d1 <= 1'b0;
            mult_valid_d2 <= 1'b0;
            mult_pixel_d0 <= 8'd0;
            mult_pixel_d1 <= 8'd0;
            mult_pixel_d2 <= 8'd0;
            mult_group_d0 <= 3'd0;
            mult_group_d1 <= 3'd0;
            mult_group_d2 <= 3'd0;
        end else begin
            mult_valid_d0 <= rom_valid;
            mult_valid_d1 <= mult_valid_d0;
            mult_valid_d2 <= mult_valid_d1;
            if (rom_valid) begin
                mult_pixel_d0 <= rom_pixel;
                mult_group_d0 <= rom_group;
            end
            if (mult_valid_d0) begin
                mult_pixel_d1 <= mult_pixel_d0;
                mult_group_d1 <= mult_group_d0;
            end
            if (mult_valid_d1) begin
                mult_pixel_d2 <= mult_pixel_d1;
                mult_group_d2 <= mult_group_d1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Fully registered binary adder tree.  No stage contains a four/five-input
    // wide addition; every register input is formed from at most two operands.
    // Bias is added only after all sixteen products have been reduced.
    // -------------------------------------------------------------------------
    reg [127:0] bias_mult_d0, bias_mult_d1, bias_mult_d2;
    reg signed [16:0] sum_l1 [0:31];
    reg signed [17:0] sum_l2 [0:15];
    reg signed [18:0] sum_l3 [0:7];
    reg signed [19:0] sum_l4 [0:3];
    reg [1:0] borrow_l1 [0:15];
    reg [2:0] borrow_l2 [0:7];
    reg [3:0] borrow_l3 [0:3];
    reg [4:0] borrow_l4 [0:1];
    reg signed [20:0] corrected_sum_l5 [0:3];
    reg signed [31:0] accumulator_l6 [0:3];
    reg valid_l1, valid_l2, valid_l3, valid_l4, valid_l5, valid_l6;
    reg [7:0] pixel_l1, pixel_l2, pixel_l3, pixel_l4, pixel_l5, pixel_l6;
    reg [2:0] group_l1, group_l2, group_l3, group_l4, group_l5, group_l6;
    reg [127:0] bias_l1, bias_l2, bias_l3, bias_l4, bias_l5;
    integer add_index;
    integer bias_lane;
    integer borrow_index;

    always @(posedge clk) begin
        if (!rst_n) begin
            bias_mult_d0 <= 128'd0;
            bias_mult_d1 <= 128'd0;
            bias_mult_d2 <= 128'd0;
            valid_l1 <= 1'b0; valid_l2 <= 1'b0; valid_l3 <= 1'b0;
            valid_l4 <= 1'b0; valid_l5 <= 1'b0; valid_l6 <= 1'b0;
            pixel_l1 <= 8'd0; pixel_l2 <= 8'd0; pixel_l3 <= 8'd0;
            pixel_l4 <= 8'd0; pixel_l5 <= 8'd0; pixel_l6 <= 8'd0;
            group_l1 <= 3'd0; group_l2 <= 3'd0; group_l3 <= 3'd0;
            group_l4 <= 3'd0; group_l5 <= 3'd0; group_l6 <= 3'd0;
            bias_l1 <= 128'd0; bias_l2 <= 128'd0;
            bias_l3 <= 128'd0; bias_l4 <= 128'd0; bias_l5 <= 128'd0;
        end else begin
            if (rom_valid)
                bias_mult_d0 <= bias_group_word;
            if (mult_valid_d0)
                bias_mult_d1 <= bias_mult_d0;
            if (mult_valid_d1)
                bias_mult_d2 <= bias_mult_d1;

            valid_l1 <= mult_valid_d2;
            valid_l2 <= valid_l1;
            valid_l3 <= valid_l2;
            valid_l4 <= valid_l3;
            valid_l5 <= valid_l4;
            valid_l6 <= valid_l5;

            if (mult_valid_d2) begin
                pixel_l1 <= mult_pixel_d2;
                group_l1 <= mult_group_d2;
                bias_l1 <= bias_mult_d2;
                for (add_index = 0; add_index < 32;
                     add_index = add_index + 1)
                    sum_l1[add_index]
                        <= $signed({product_mult[add_index*2][16],
                                    product_mult[add_index*2]})
                         + $signed({product_mult[add_index*2+1][16],
                                    product_mult[add_index*2+1]});
                for (borrow_index = 0; borrow_index < 16;
                     borrow_index = borrow_index + 1)
                    borrow_l1[borrow_index]
                        <= {1'b0, packed_low_negative[borrow_index*2]}
                         + {1'b0, packed_low_negative[borrow_index*2+1]};
            end
            if (valid_l1) begin
                pixel_l2 <= pixel_l1; group_l2 <= group_l1;
                bias_l2 <= bias_l1;
                for (add_index = 0; add_index < 16;
                     add_index = add_index + 1)
                    sum_l2[add_index]
                        <= $signed({sum_l1[add_index*2][16],
                                    sum_l1[add_index*2]})
                         + $signed({sum_l1[add_index*2+1][16],
                                    sum_l1[add_index*2+1]});
                for (borrow_index = 0; borrow_index < 8;
                     borrow_index = borrow_index + 1)
                    borrow_l2[borrow_index]
                        <= {1'b0, borrow_l1[borrow_index*2]}
                         + {1'b0, borrow_l1[borrow_index*2+1]};
            end
            if (valid_l2) begin
                pixel_l3 <= pixel_l2; group_l3 <= group_l2;
                bias_l3 <= bias_l2;
                for (add_index = 0; add_index < 8;
                     add_index = add_index + 1)
                    sum_l3[add_index]
                        <= $signed({sum_l2[add_index*2][17],
                                    sum_l2[add_index*2]})
                         + $signed({sum_l2[add_index*2+1][17],
                                    sum_l2[add_index*2+1]});
                for (borrow_index = 0; borrow_index < 4;
                     borrow_index = borrow_index + 1)
                    borrow_l3[borrow_index]
                        <= {1'b0, borrow_l2[borrow_index*2]}
                         + {1'b0, borrow_l2[borrow_index*2+1]};
            end
            if (valid_l3) begin
                pixel_l4 <= pixel_l3; group_l4 <= group_l3;
                bias_l4 <= bias_l3;
                for (add_index = 0; add_index < 4;
                     add_index = add_index + 1)
                    sum_l4[add_index]
                        <= $signed({sum_l3[add_index*2][18],
                                    sum_l3[add_index*2]})
                         + $signed({sum_l3[add_index*2+1][18],
                                    sum_l3[add_index*2+1]});
                for (borrow_index = 0; borrow_index < 2;
                     borrow_index = borrow_index + 1)
                    borrow_l4[borrow_index]
                        <= {1'b0, borrow_l3[borrow_index*2]}
                         + {1'b0, borrow_l3[borrow_index*2+1]};
            end
            if (valid_l4) begin
                pixel_l5 <= pixel_l4; group_l5 <= group_l4;
                bias_l5 <= bias_l4;
                corrected_sum_l5[0]
                    <= $signed({sum_l4[0][19], sum_l4[0]});
                corrected_sum_l5[1]
                    <= $signed({sum_l4[1][19], sum_l4[1]})
                     + $signed({16'd0, borrow_l4[0]});
                corrected_sum_l5[2]
                    <= $signed({sum_l4[2][19], sum_l4[2]});
                corrected_sum_l5[3]
                    <= $signed({sum_l4[3][19], sum_l4[3]})
                     + $signed({16'd0, borrow_l4[1]});
            end
            if (valid_l5) begin
                pixel_l6 <= pixel_l5; group_l6 <= group_l5;
                for (bias_lane = 0; bias_lane < 4;
                     bias_lane = bias_lane + 1)
                    accumulator_l6[bias_lane]
                        <= $signed({{11{corrected_sum_l5[bias_lane][20]}},
                                    corrected_sum_l5[bias_lane]})
                         + $signed(bias_l5[bias_lane*32 +: 32]);
            end
        end
    end

    // Match the current software boundary exactly:
    // symmetric round-to-nearest -> signed INT8 saturation -> ReLU.
    // Combining the last two operations is equivalent to saturation to 0..127.
    function [7:0] round_shift_clip_relu_int8;
        input signed [63:0] product_value;
        input signed [6:0] shift_value;
        reg signed [63:0] absolute_product;
        reg signed [63:0] half_lsb;
        reg signed [63:0] rounded_value;
        begin
            if (shift_value > 0) begin
                half_lsb = 64'sd1 <<< (shift_value - 1'b1);
                if (product_value >= 0) begin
                    rounded_value = (product_value + half_lsb) >>> shift_value;
                end else begin
                    absolute_product = -product_value;
                    rounded_value = -((absolute_product + half_lsb) >>> shift_value);
                end
            end else if (shift_value < 0) begin
                rounded_value = product_value <<< (-shift_value);
            end else begin
                rounded_value = product_value;
            end

            if (rounded_value > 64'sd127)
                round_shift_clip_relu_int8 = 8'd127;
            else if (rounded_value <= 0)
                round_shift_clip_relu_int8 = 8'd0;
            else
                round_shift_clip_relu_int8 = rounded_value[7:0];
        end
    endfunction

    // -------------------------------------------------------------------------
    // Four signed 21 x 16, three-stage Multiplier Generator IPs implement the
    // per-tensor requantization multiply.  M_int is positive and can be as
    // large as 32767, so signed 16-bit is required.
    // -------------------------------------------------------------------------
    wire signed [36:0] requant_product [0:3];

    genvar requant_lane;
    generate
        for (requant_lane = 0; requant_lane < 4; requant_lane = requant_lane + 1) begin : GEN_REQUANT_MULTIPLIER
            requant_mult_21x16 u_requant_mult_21x16 (
                .CLK (clk),
                .A   (accumulator_l6[requant_lane][20:0]),
                .B   (requant_multiplier_fixed),
                .P   (requant_product[requant_lane])
            );
        end
    endgenerate

    reg requant_valid_d0;
    reg requant_valid_d1;
    reg requant_valid_d2;
    reg [7:0] requant_pixel_d0;
    reg [7:0] requant_pixel_d1;
    reg [7:0] requant_pixel_d2;
    reg [2:0] requant_group_d0;
    reg [2:0] requant_group_d1;
    reg [2:0] requant_group_d2;

    always @(posedge clk) begin
        if (!rst_n) begin
            requant_valid_d0 <= 1'b0;
            requant_valid_d1 <= 1'b0;
            requant_valid_d2 <= 1'b0;
            requant_pixel_d0 <= 8'd0;
            requant_pixel_d1 <= 8'd0;
            requant_pixel_d2 <= 8'd0;
            requant_group_d0 <= 3'd0;
            requant_group_d1 <= 3'd0;
            requant_group_d2 <= 3'd0;
        end else begin
            requant_valid_d0 <= valid_l6;
            requant_valid_d1 <= requant_valid_d0;
            requant_valid_d2 <= requant_valid_d1;
            if (valid_l6) begin
                requant_pixel_d0 <= pixel_l6;
                requant_group_d0 <= group_l6;
            end
            if (requant_valid_d0) begin
                requant_pixel_d1 <= requant_pixel_d0;
                requant_group_d1 <= requant_group_d0;
            end
            if (requant_valid_d1) begin
                requant_pixel_d2 <= requant_pixel_d1;
                requant_group_d2 <= requant_group_d1;
            end
        end
    end

    // -------------------------------------------------------------------------
    // Output stage: rounding, shift, INT8 saturation, ReLU and four-lane pack.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_pixel_addr <= 8'd0;
            out_channel_base <= 5'd0;
            out_data <= 32'd0;
        end else begin
            out_valid <= requant_valid_d2;
            if (requant_valid_d2) begin
                out_pixel_addr <= requant_pixel_d2;
                out_channel_base <= {requant_group_d2, 2'b00};
                out_data[7:0]
                    <= round_shift_clip_relu_int8(
                        requant_product[0],
                        requant_shift_fixed
                    );
                out_data[15:8]
                    <= round_shift_clip_relu_int8(
                        requant_product[1],
                        requant_shift_fixed
                    );
                out_data[23:16]
                    <= round_shift_clip_relu_int8(
                        requant_product[2],
                        requant_shift_fixed
                    );
                out_data[31:24]
                    <= round_shift_clip_relu_int8(
                        requant_product[3],
                        requant_shift_fixed
                    );
            end
        end
    end

    // -------------------------------------------------------------------------
    // Scheduler and one-pixel-ahead BRAM prefetch.
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= ST_IDLE;
            done <= 1'b0;
            input_read_address <= 8'd0;
            current_pixel_data <= 128'd0;
            pixel_issue <= 8'd0;
            group_issue <= 3'd0;
        end else begin
            done <= 1'b0;

            case (state)
                ST_IDLE: begin
                    pixel_issue <= 8'd0;
                    group_issue <= 3'd0;
                    if (start) begin
                        input_read_address <= 8'd0;
                        state <= ST_PRIME0;
                    end
                end

                // Two cycles prime the synchronous BRAM read path.
                ST_PRIME0: begin
                    state <= ST_PRIME1;
                end

                ST_PRIME1: begin
                    current_pixel_data <= input_bram_q;
                    state <= ST_RUN;
                end

                ST_RUN: begin
                    // Start fetching the next pixel early enough that it is
                    // available when group 7 of the current pixel is issued.
                    if ((group_issue == 3'd5) && (pixel_issue != 8'd255))
                        input_read_address <= pixel_issue + 1'b1;

                    if (group_issue == 3'd7) begin
                        if (pixel_issue == 8'd255) begin
                            state <= ST_DRAIN;
                        end else begin
                            pixel_issue <= pixel_issue + 1'b1;
                            group_issue <= 3'd0;
                            current_pixel_data <= input_bram_q;
                        end
                    end else begin
                        group_issue <= group_issue + 1'b1;
                    end
                end

                ST_DRAIN: begin
                    // Wait for the final result to pass through the three-stage
                    // requantization multiplier before asserting done.
                    if (requant_valid_d2
                        && (requant_pixel_d2 == 8'd255)
                        && (requant_group_d2 == 3'd7)) begin
                        done <= 1'b1;
                        state <= ST_IDLE;
                    end
                end

                default: begin
                    state <= ST_IDLE;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
