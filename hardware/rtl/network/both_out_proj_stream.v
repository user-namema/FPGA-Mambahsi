`timescale 1ns / 1ps
`default_nettype none

// Eight parallel 16-element signed INT8 dot products.
module out_proj_dot16x8_pipeline (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 issue_valid,
    input  wire [127:0]         activation_vector,
    input  wire [1023:0]        weight_vectors,
    input  wire [19:0]          issue_metadata,
    output reg                  sum_valid,
    output wire [167:0]         sum_values,
    output reg [19:0]           sum_metadata
);
    wire signed [15:0] products [0:127];
    genvar output_lane;
    genvar input_lane;

    generate
        for (output_lane = 0; output_lane < 4;
             output_lane = output_lane + 1) begin : GEN_OUTPUT
            for (input_lane = 0; input_lane < 16;
                 input_lane = input_lane + 1) begin : GEN_INPUT
                mamba_packed_signed_mult_2x8_3cyc u_multiplier (
                    .clk(clk),.activation(activation_vector[input_lane*8 +: 8]),
                    .weight_low(weight_vectors[((output_lane*2)*16+input_lane)*8 +: 8]),
                    .weight_high(weight_vectors[((output_lane*2+1)*16+input_lane)*8 +: 8]),
                    .product_low(products[(output_lane*2)*16+input_lane]),
                    .product_high(products[(output_lane*2+1)*16+input_lane]));
            end
        end
    endgenerate

    reg mult_valid_d0, mult_valid_d1, mult_valid_d2;
    reg [19:0] mult_meta_d0, mult_meta_d1, mult_meta_d2;
    reg signed [16:0] sum_l1 [0:63];
    reg signed [17:0] sum_l2 [0:31];
    (* use_dsp = "no" *) reg signed [18:0] sum_l3 [0:15];
    (* use_dsp = "no" *) reg signed [19:0] sum_l4 [0:7];
    reg valid_l1, valid_l2, valid_l3, valid_l4;
    reg [19:0] meta_l1, meta_l2, meta_l3, meta_l4;
    integer add_index;
    genvar sum_lane;

    generate
        for (sum_lane = 0; sum_lane < 8;
             sum_lane = sum_lane + 1) begin : GEN_SUM_PORT
            assign sum_values[sum_lane*21 +: 21]
                = {{1{sum_l4[sum_lane][19]}}, sum_l4[sum_lane]};
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            mult_valid_d0 <= 1'b0;
            mult_valid_d1 <= 1'b0;
            mult_valid_d2 <= 1'b0;
            valid_l1 <= 1'b0;
            valid_l2 <= 1'b0;
            valid_l3 <= 1'b0;
            valid_l4 <= 1'b0;
            sum_valid <= 1'b0;
        end else begin
            if (frame_start) begin
                mult_valid_d0 <= 1'b0;
                mult_valid_d1 <= 1'b0;
                mult_valid_d2 <= 1'b0;
                valid_l1 <= 1'b0;
                valid_l2 <= 1'b0;
                valid_l3 <= 1'b0;
                valid_l4 <= 1'b0;
                sum_valid <= 1'b0;
            end else begin
                mult_valid_d0 <= issue_valid;
                mult_valid_d1 <= mult_valid_d0;
                mult_valid_d2 <= mult_valid_d1;
                valid_l1 <= mult_valid_d2;
                valid_l2 <= valid_l1;
                valid_l3 <= valid_l2;
                valid_l4 <= valid_l3;
                sum_valid <= valid_l4;
            end
        end
        // Reduction payload and metadata free-run every clock.  The valid
        // pipeline is the sole architectural qualifier.  This removes the
        // 1088/576-load clock-enable trees formerly rooted in the first two
        // Spe out_proj reduction levels while preserving all pipeline stages.
        mult_meta_d0 <= issue_metadata;
        mult_meta_d1 <= mult_meta_d0;
        mult_meta_d2 <= mult_meta_d1;
        meta_l1 <= mult_meta_d2;
        meta_l2 <= meta_l1;
        meta_l3 <= meta_l2;
        meta_l4 <= meta_l3;
        sum_metadata <= meta_l4;
        for (add_index = 0; add_index < 64;
             add_index = add_index + 1)
            sum_l1[add_index]
                <= $signed({products[add_index*2][15],
                            products[add_index*2]})
                 + $signed({products[add_index*2+1][15],
                            products[add_index*2+1]});
        for (add_index = 0; add_index < 32;
             add_index = add_index + 1)
            sum_l2[add_index]
                <= $signed({sum_l1[add_index*2][16],
                            sum_l1[add_index*2]})
                 + $signed({sum_l1[add_index*2+1][16],
                            sum_l1[add_index*2+1]});
        for (add_index = 0; add_index < 16;
             add_index = add_index + 1)
            sum_l3[add_index]
                <= $signed({sum_l2[add_index*2][17],
                            sum_l2[add_index*2]})
                 + $signed({sum_l2[add_index*2+1][17],
                            sum_l2[add_index*2+1]});
        for (add_index = 0; add_index < 8;
             add_index = add_index + 1)
            sum_l4[add_index]
                <= $signed({sum_l3[add_index*2][18],
                            sum_l3[add_index*2]})
                 + $signed({sum_l3[add_index*2+1][18],
                            sum_l3[add_index*2+1]});
    end
endmodule


module spa_out_proj_stream (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [9:0]           in_record_addr,
    input  wire [5:0]           in_channel_base,
    input  wire [31:0]          in_data,
    input  wire                 cfg_weight_we,
    input  wire [10:0]          cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [4:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [15:0]   cfg_multiplier,
    input  wire signed [6:0]    cfg_shift,
    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [4:0]            out_channel_base,
    output reg [31:0]           out_raw_data,
    output reg [31:0]           out_data,
    output reg                  done
);
    reg signed [7:0] activation_mem [0:63];
    reg signed [7:0] weight_mem [0:2047];
    reg signed [31:0] bias_mem [0:31];

    integer capture_lane;
    always @(posedge clk) begin
        if (cfg_weight_we)
            weight_mem[cfg_weight_addr] <= cfg_weight_data;
        if (cfg_bias_we)
            bias_mem[cfg_bias_addr] <= cfg_bias_data;
        if (in_valid) begin
            for (capture_lane = 0; capture_lane < 4;
                 capture_lane = capture_lane + 1) begin
                activation_mem[in_channel_base + capture_lane]
                    <= in_data[capture_lane*8 +: 8];
            end
        end
    end

    reg active;
    reg [9:0] active_record;
    reg [2:0] issue_group;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            active <= 1'b0;
            active_record <= 10'd0;
            issue_group <= 3'd0;
        end else begin
            if (active) begin
                if (issue_group == 3'd7) begin
                    active <= 1'b0;
                    issue_group <= 3'd0;
                end else begin
                    issue_group <= issue_group + 1'b1;
                end
            end
            if (in_valid && (in_channel_base == 6'd60)) begin
                active <= 1'b1;
                active_record <= in_record_addr;
                issue_group <= 3'd0;
            end
        end
    end

    wire [511:0] activation_vector;
    wire [2047:0] weight_vectors;
    genvar activation_lane;
    genvar weight_output_lane;
    genvar weight_input_lane;
    generate
        for (activation_lane = 0; activation_lane < 64;
             activation_lane = activation_lane + 1) begin : GEN_ACTIVATION
            assign activation_vector[activation_lane*8 +: 8]
                = activation_mem[activation_lane];
        end
        for (weight_output_lane = 0; weight_output_lane < 4;
             weight_output_lane = weight_output_lane + 1) begin : GEN_WEIGHT_O
            for (weight_input_lane = 0; weight_input_lane < 64;
                 weight_input_lane = weight_input_lane + 1) begin : GEN_WEIGHT_I
                assign weight_vectors[
                    (weight_output_lane*64+weight_input_lane)*8 +: 8
                ] = weight_mem[
                    (issue_group*4+weight_output_lane)*64+weight_input_lane
                ];
            end
        end
    endgenerate

    wire sum_valid;
    wire signed [20:0] sum_0, sum_1, sum_2, sum_3;
    wire [19:0] sum_metadata;
    x_proj_dot64x4_pipeline u_dot_product (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .issue_valid(active),
        .activation_vector(activation_vector),
        .weight_vectors(weight_vectors),
        .issue_metadata({7'd0, active_record, issue_group}),
        .sum_valid(sum_valid),
        .sum_0(sum_0),
        .sum_1(sum_1),
        .sum_2(sum_2),
        .sum_3(sum_3),
        .sum_metadata(sum_metadata)
    );

    reg accumulator_valid;
    (* use_dsp = "no" *) reg signed [20:0] accumulator [0:3];
    reg [19:0] accumulator_metadata;
    integer accumulator_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            accumulator_valid <= 1'b0;
            accumulator_metadata <= 20'd0;
        end else begin
            accumulator_valid <= sum_valid;
            if (sum_valid) begin
                accumulator_metadata <= sum_metadata;
                accumulator[0] <= sum_0
                    + $signed(bias_mem[{sum_metadata[2:0], 2'b00}][20:0]);
                accumulator[1] <= sum_1
                    + $signed(bias_mem[{sum_metadata[2:0], 2'b00}+1][20:0]);
                accumulator[2] <= sum_2
                    + $signed(bias_mem[{sum_metadata[2:0], 2'b00}+2][20:0]);
                accumulator[3] <= sum_3
                    + $signed(bias_mem[{sum_metadata[2:0], 2'b00}+3][20:0]);
            end
        end
    end

    wire signed [36:0] requant_product [0:3];
    genvar requant_lane;
    generate
        for (requant_lane = 0; requant_lane < 4;
             requant_lane = requant_lane + 1) begin : GEN_REQUANT
            requant_mult_21x16 u_requant (
                .CLK(clk),
                .A(accumulator[requant_lane]),
                .B(cfg_multiplier),
                .P(requant_product[requant_lane])
            );
        end
    endgenerate

    reg rq_valid_d0, rq_valid_d1, rq_valid_d2;
    reg [19:0] rq_meta_d0, rq_meta_d1, rq_meta_d2;
    function [7:0] round_shift_clip_int8;
        input signed [36:0] value;
        input signed [6:0] shift_value;
        reg signed [63:0] magnitude;
        reg signed [63:0] rounded;
        reg signed [63:0] signed_rounded;
        begin
            magnitude = (value < 0)
                ? -$signed({{27{value[36]}}, value})
                :  $signed({{27{value[36]}}, value});
            if (shift_value > 0)
                rounded = (magnitude + (64'sd1 <<< (shift_value-1)))
                          >>> shift_value;
            else if (shift_value < 0)
                rounded = magnitude <<< (-shift_value);
            else
                rounded = magnitude;
            signed_rounded = (value < 0) ? -rounded : rounded;
            if (signed_rounded > 64'sd127)
                round_shift_clip_int8 = 8'h7f;
            else if (signed_rounded < -64'sd128)
                round_shift_clip_int8 = 8'h80;
            else
                round_shift_clip_int8 = signed_rounded[7:0];
        end
    endfunction

    integer output_lane;
    reg [7:0] raw_code;
    always @(posedge clk) begin
        if (!rst_n) begin
            rq_valid_d0 <= 1'b0;
            rq_valid_d1 <= 1'b0;
            rq_valid_d2 <= 1'b0;
            rq_meta_d0 <= 20'd0;
            rq_meta_d1 <= 20'd0;
            rq_meta_d2 <= 20'd0;
            out_valid <= 1'b0;
            out_pixel_addr <= 8'd0;
            out_channel_base <= 5'd0;
            out_raw_data <= 32'd0;
            out_data <= 32'd0;
            done <= 1'b0;
        end else begin
            if (frame_start) begin
                rq_valid_d0 <= 1'b0;
                rq_valid_d1 <= 1'b0;
                rq_valid_d2 <= 1'b0;
                out_valid <= 1'b0;
                out_pixel_addr <= 8'd0;
                out_channel_base <= 5'd0;
                done <= 1'b0;
            end else begin
                done <= 1'b0;
                rq_valid_d0 <= accumulator_valid;
                rq_valid_d1 <= rq_valid_d0;
                rq_valid_d2 <= rq_valid_d1;
                out_valid <= rq_valid_d2;
            end
            if (accumulator_valid)
                rq_meta_d0 <= accumulator_metadata;
            if (rq_valid_d0)
                rq_meta_d1 <= rq_meta_d0;
            if (rq_valid_d1)
                rq_meta_d2 <= rq_meta_d1;

            if (rq_valid_d2) begin
                if (!frame_start) begin
                    out_pixel_addr <= rq_meta_d2[10:3];
                    out_channel_base <= {rq_meta_d2[2:0], 2'b00};
                end
                for (output_lane = 0; output_lane < 4;
                     output_lane = output_lane + 1) begin
                    raw_code = round_shift_clip_int8(
                        requant_product[output_lane], cfg_shift
                    );
                    out_raw_data[output_lane*8 +: 8] <= raw_code;
                    out_data[output_lane*8 +: 8]
                        <= raw_code[7] ? 8'd0 : raw_code;
                end
                if (!frame_start && (rq_meta_d2[12:3] == 10'd255)
                    && (rq_meta_d2[2:0] == 3'd7)) begin
                    done <= 1'b1;
                end
            end
        end
    end
endmodule


module spe_out_proj_stream (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [9:0]           in_record_addr,
    input  wire [5:0]           in_channel_base,
    input  wire [31:0]          in_data,
    input  wire                 cfg_weight_we,
    input  wire [6:0]           cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [2:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [15:0]   cfg_multiplier,
    input  wire signed [6:0]    cfg_shift,
    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [1:0]            out_token,
    output reg [63:0]           out_raw_data,
    output reg [63:0]           out_data,
    output reg                  done
);
    reg signed [7:0] activation_mem [0:15];
    reg signed [7:0] weight_mem [0:127];
    reg signed [31:0] bias_mem [0:7];
    integer capture_lane;

    always @(posedge clk) begin
        if (cfg_weight_we)
            weight_mem[cfg_weight_addr] <= cfg_weight_data;
        if (cfg_bias_we)
            bias_mem[cfg_bias_addr] <= cfg_bias_data;
        if (in_valid) begin
            for (capture_lane = 0; capture_lane < 4;
                 capture_lane = capture_lane + 1) begin
                activation_mem[in_channel_base + capture_lane]
                    <= in_data[capture_lane*8 +: 8];
            end
        end
    end

    reg issue_valid;
    reg [9:0] issue_record;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            issue_valid <= 1'b0;
            issue_record <= 10'd0;
        end else begin
            issue_valid <= in_valid && (in_channel_base == 6'd12);
            if (in_valid && (in_channel_base == 6'd12))
                issue_record <= in_record_addr;
        end
    end

    wire [127:0] activation_vector;
    wire [1023:0] weight_vectors;
    genvar activation_lane;
    genvar weight_lane;
    generate
        for (activation_lane = 0; activation_lane < 16;
             activation_lane = activation_lane + 1) begin : GEN_ACTIVATION
            assign activation_vector[activation_lane*8 +: 8]
                = activation_mem[activation_lane];
        end
        for (weight_lane = 0; weight_lane < 128;
             weight_lane = weight_lane + 1) begin : GEN_WEIGHT
            assign weight_vectors[weight_lane*8 +: 8]
                = weight_mem[weight_lane];
        end
    endgenerate

    wire sum_valid;
    wire [167:0] sum_values;
    wire [19:0] sum_metadata;
    out_proj_dot16x8_pipeline u_dot_product (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .issue_valid(issue_valid),
        .activation_vector(activation_vector),
        .weight_vectors(weight_vectors),
        .issue_metadata({10'd0, issue_record}),
        .sum_valid(sum_valid),
        .sum_values(sum_values),
        .sum_metadata(sum_metadata)
    );

    reg accumulator_valid;
    (* use_dsp = "no" *) reg signed [20:0] accumulator [0:7];
    reg [9:0] accumulator_record;
    integer accumulator_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            accumulator_valid <= 1'b0;
            accumulator_record <= 10'd0;
        end else begin
            accumulator_valid <= sum_valid;
            if (sum_valid) begin
                accumulator_record <= sum_metadata[9:0];
                for (accumulator_lane = 0; accumulator_lane < 8;
                     accumulator_lane = accumulator_lane + 1) begin
                    accumulator[accumulator_lane]
                        <= $signed(sum_values[
                               accumulator_lane*21 +: 21
                           ])
                         + $signed(bias_mem[accumulator_lane][20:0]);
                end
            end
        end
    end

    wire signed [36:0] requant_product [0:7];
    genvar requant_lane;
    generate
        for (requant_lane = 0; requant_lane < 8;
             requant_lane = requant_lane + 1) begin : GEN_REQUANT
            requant_mult_21x16 u_requant (
                .CLK(clk),
                .A(accumulator[requant_lane]),
                .B(cfg_multiplier),
                .P(requant_product[requant_lane])
            );
        end
    endgenerate

    reg rq_valid_d0, rq_valid_d1, rq_valid_d2;
    reg [9:0] rq_record_d0, rq_record_d1, rq_record_d2;
    function [7:0] round_shift_clip_int8;
        input signed [36:0] value;
        input signed [6:0] shift_value;
        reg signed [63:0] magnitude;
        reg signed [63:0] rounded;
        reg signed [63:0] signed_rounded;
        begin
            magnitude = (value < 0)
                ? -$signed({{27{value[36]}}, value})
                :  $signed({{27{value[36]}}, value});
            if (shift_value > 0)
                rounded = (magnitude + (64'sd1 <<< (shift_value-1)))
                          >>> shift_value;
            else if (shift_value < 0)
                rounded = magnitude <<< (-shift_value);
            else
                rounded = magnitude;
            signed_rounded = (value < 0) ? -rounded : rounded;
            if (signed_rounded > 64'sd127)
                round_shift_clip_int8 = 8'h7f;
            else if (signed_rounded < -64'sd128)
                round_shift_clip_int8 = 8'h80;
            else
                round_shift_clip_int8 = signed_rounded[7:0];
        end
    endfunction

    integer output_lane;
    reg [7:0] raw_code;
    always @(posedge clk) begin
        if (!rst_n) begin
            rq_valid_d0 <= 1'b0;
            rq_valid_d1 <= 1'b0;
            rq_valid_d2 <= 1'b0;
            rq_record_d0 <= 10'd0;
            rq_record_d1 <= 10'd0;
            rq_record_d2 <= 10'd0;
            out_valid <= 1'b0;
            out_pixel_addr <= 8'd0;
            out_token <= 2'd0;
            out_raw_data <= 64'd0;
            out_data <= 64'd0;
            done <= 1'b0;
        end else begin
            if (frame_start) begin
                rq_valid_d0 <= 1'b0;
                rq_valid_d1 <= 1'b0;
                rq_valid_d2 <= 1'b0;
                out_valid <= 1'b0;
                out_pixel_addr <= 8'd0;
                out_token <= 2'd0;
                done <= 1'b0;
            end else begin
                done <= 1'b0;
                rq_valid_d0 <= accumulator_valid;
                rq_valid_d1 <= rq_valid_d0;
                rq_valid_d2 <= rq_valid_d1;
                out_valid <= rq_valid_d2;
            end
            if (accumulator_valid)
                rq_record_d0 <= accumulator_record;
            if (rq_valid_d0)
                rq_record_d1 <= rq_record_d0;
            if (rq_valid_d1)
                rq_record_d2 <= rq_record_d1;

            if (rq_valid_d2) begin
                if (!frame_start) begin
                    out_pixel_addr <= rq_record_d2[9:2];
                    out_token <= rq_record_d2[1:0];
                end
                for (output_lane = 0; output_lane < 8;
                     output_lane = output_lane + 1) begin
                    raw_code = round_shift_clip_int8(
                        requant_product[output_lane], cfg_shift
                    );
                    out_raw_data[output_lane*8 +: 8] <= raw_code;
                    out_data[output_lane*8 +: 8]
                        <= raw_code[7] ? 8'd0 : raw_code;
                end
                if (!frame_start && rq_record_d2 == 10'd1023)
                    done <= 1'b1;
            end
        end
    end
endmodule

`default_nettype wire
