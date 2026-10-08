`timescale 1ns / 1ps
`default_nettype none

// Spa dt_proj: rank-2 INT9 input -> 64 INT8 channels.
// Eight output channels are issued per clock, so one time point takes 8 clocks.
module spa_dt_proj_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] REQUANT_MULTIPLIER=16'sd21177,
    parameter signed [6:0] REQUANT_SHIFT=7'sd22
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [7:0]           in_pixel_addr,
    input  wire [17:0]          in_data,
    input  wire                 cfg_weight_we,
    input  wire [6:0]           cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [5:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [31:0]   cfg_multiplier,
    input  wire signed [6:0]    cfg_shift,
    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [5:0]            out_channel_base,
    output reg [63:0]           out_data,
    output reg                  done
);
    wire [383:0] param_word;
    mamba_spa_dt_param_rom #(.BLOCK_ID(BLOCK_ID)) u_param_rom(
        .clk(clk),.en(1'b1),.addr(issue_group),.data(param_word));

    reg active;
    reg [7:0] pixel;
    reg [17:0] dt_codes;
    reg [2:0] issue_group;
    reg rom_valid;reg[7:0]rom_pixel;reg[2:0]rom_group;reg[17:0]rom_dt_codes;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            active <= 1'b0;
            issue_group <= 3'd0;
            rom_valid<=0;
        end else begin
            rom_valid<=active;
            // ROM and metadata payload are free-running; rom_valid is the
            // only architectural qualifier.
            rom_pixel<=pixel;rom_group<=issue_group;rom_dt_codes<=dt_codes;
            if (active) begin
                if (issue_group == 3'd7) begin
                    active <= 1'b0;
                    issue_group <= 3'd0;
                end else begin
                    issue_group <= issue_group + 1'b1;
                end
            end

            if (in_valid) begin
                active <= 1'b1;
                pixel <= in_pixel_addr;
                dt_codes <= in_data;
                issue_group <= 3'd0;
            end
        end
    end

    wire signed [16:0] product_0 [0:7];
    wire signed [16:0] product_1 [0:7];
    genvar lane;

    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : GEN_DOT_PAIR
            mamba_packed_signed_mult_2x9_3cyc u_mult_0 (
                .clk(clk),.activation(rom_dt_codes[8:0]),
                .weight_low(param_word[(lane*2)*16 +:8]),
                .weight_high(param_word[(lane*2+1)*16 +:8]),
                .product_low(product_0[lane*2]),
                .product_high(product_0[lane*2+1]));

            mamba_packed_signed_mult_2x9_3cyc u_mult_1 (
                .clk(clk),.activation(rom_dt_codes[17:9]),
                .weight_low(param_word[(lane*2)*16+8 +:8]),
                .weight_high(param_word[(lane*2+1)*16+8 +:8]),
                .product_low(product_1[lane*2]),
                .product_high(product_1[lane*2+1]));
        end
    endgenerate

    reg mult_valid_d0;
    reg mult_valid_d1;
    reg mult_valid_d2;
    reg [7:0] mult_pixel_d0;
    reg [7:0] mult_pixel_d1;
    reg [7:0] mult_pixel_d2;
    reg [2:0] mult_group_d0;
    reg [2:0] mult_group_d1;
    reg [2:0] mult_group_d2;
    reg [255:0] bias_d0,bias_d1,bias_d2;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            mult_valid_d0 <= 1'b0;
            mult_valid_d1 <= 1'b0;
            mult_valid_d2 <= 1'b0;
        end else begin
            mult_valid_d0 <= rom_valid;
            mult_valid_d1 <= mult_valid_d0;
            mult_valid_d2 <= mult_valid_d1;
        end
        // Payload advances unconditionally; mult_valid alone qualifies it.
        mult_pixel_d0 <= rom_pixel;
        mult_pixel_d1 <= mult_pixel_d0;
        mult_pixel_d2 <= mult_pixel_d1;
        mult_group_d0 <= rom_group;
        mult_group_d1 <= mult_group_d0;
        mult_group_d2 <= mult_group_d1;
        bias_d0<=param_word[383:128];
        bias_d1<=bias_d0;
        bias_d2<=bias_d1;
    end

    reg pair_valid;
    reg accumulator_valid;
    reg signed [17:0] pair_sum [0:7];
    reg [7:0] pair_pixel;
    reg [2:0] pair_group;
    reg [255:0] pair_bias;
    reg [7:0] accumulator_pixel;
    reg [2:0] accumulator_group;
    reg signed [20:0] accumulator [0:7];
    integer add_lane;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            pair_valid <= 1'b0;
            accumulator_valid <= 1'b0;
        end else begin
            pair_valid<=mult_valid_d2;
            accumulator_valid <= pair_valid;
        end
        pair_pixel<=mult_pixel_d2;pair_group<=mult_group_d2;pair_bias<=bias_d2;
        for (add_lane = 0; add_lane < 8;
             add_lane = add_lane + 1) begin
            pair_sum[add_lane]
                <= $signed({product_0[add_lane][16],product_0[add_lane]})
                 + $signed({product_1[add_lane][16],product_1[add_lane]});
        end
        accumulator_pixel<=pair_pixel;accumulator_group<=pair_group;
        for(add_lane=0;add_lane<8;add_lane=add_lane+1)
            accumulator[add_lane]
                <= $signed({{3{pair_sum[add_lane][17]}},pair_sum[add_lane]})
                 + $signed(pair_bias[add_lane*32 +:21]);
    end

    wire signed [36:0] requant_product [0:7];
    genvar requant_lane;

    generate
        for (requant_lane = 0; requant_lane < 8;
             requant_lane = requant_lane + 1) begin : GEN_REQUANT
            requant_mult_21x16 u_requant (
                .CLK(clk),
                .A(accumulator[requant_lane]),
                .B(REQUANT_MULTIPLIER),
                .P(requant_product[requant_lane])
            );
        end
    endgenerate

    reg rq_valid_d0;
    reg rq_valid_d1;
    reg rq_valid_d2;
    reg [7:0] rq_pixel_d0;
    reg [7:0] rq_pixel_d1;
    reg [7:0] rq_pixel_d2;
    reg [2:0] rq_group_d0;
    reg [2:0] rq_group_d1;
    reg [2:0] rq_group_d2;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_valid_d0 <= 1'b0;
            rq_valid_d1 <= 1'b0;
            rq_valid_d2 <= 1'b0;
        end else begin
            rq_valid_d0 <= accumulator_valid;
            rq_valid_d1 <= rq_valid_d0;
            rq_valid_d2 <= rq_valid_d1;
        end
        rq_pixel_d0 <= accumulator_pixel;
        rq_pixel_d1 <= rq_pixel_d0;
        rq_pixel_d2 <= rq_pixel_d1;
        rq_group_d0 <= accumulator_group;
        rq_group_d1 <= rq_group_d0;
        rq_group_d2 <= rq_group_d1;
    end

    function [7:0] round_shift_clip_int8;
        input signed [36:0] product_value;
        input signed [6:0] shift_value;
        reg signed [63:0] value;
        reg signed [63:0] magnitude;
        reg signed [63:0] half_lsb;
        reg signed [63:0] rounded;
        begin
            value = {{27{product_value[36]}}, product_value};
            if (shift_value > 0) begin
                half_lsb = 64'sd1 <<< (shift_value - 1'b1);
                magnitude = (value < 0) ? -value : value;
                rounded = (magnitude + half_lsb) >>> shift_value;
                if (value < 0)
                    rounded = -rounded;
            end else if (shift_value < 0) begin
                rounded = value <<< (-shift_value);
            end else begin
                rounded = value;
            end
            if (rounded > 127)
                round_shift_clip_int8 = 8'h7f;
            else if (rounded < -128)
                round_shift_clip_int8 = 8'h80;
            else
                round_shift_clip_int8 = rounded[7:0];
        end
    endfunction

    integer output_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            out_valid <= 1'b0;
            done <= 1'b0;
        end else begin
            out_valid <= rq_valid_d2;
            done <= rq_valid_d2&&(rq_pixel_d2==PIXEL_COUNT-1)
                              &&(rq_group_d2==3'd7);
        end
        out_pixel_addr <= rq_pixel_d2;
        out_channel_base <= {rq_group_d2, 3'b000};
        for (output_lane = 0; output_lane < 8;
             output_lane = output_lane + 1)
            out_data[output_lane*8 +: 8]
                <= round_shift_clip_int8(
                    requant_product[output_lane], REQUANT_SHIFT);
    end
endmodule


// Spe dt_proj: rank-1 INT9 input -> 16 INT8 channels.
// Eight output channels are issued per clock, so one token takes 2 clocks.
module spe_dt_proj_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] REQUANT_MULTIPLIER=16'sd18236,
    parameter signed [6:0] REQUANT_SHIFT=7'sd22
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 in_valid,
    input  wire [7:0]           in_pixel_addr,
    input  wire [1:0]           in_token,
    input  wire [8:0]           in_data,
    input  wire                 cfg_weight_we,
    input  wire [3:0]           cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire                 cfg_bias_we,
    input  wire [3:0]           cfg_bias_addr,
    input  wire signed [31:0]   cfg_bias_data,
    input  wire signed [31:0]   cfg_multiplier,
    input  wire signed [6:0]    cfg_shift,
    output reg                  out_valid,
    output reg [7:0]            out_pixel_addr,
    output reg [1:0]            out_token,
    output reg [3:0]            out_channel_base,
    output reg [63:0]           out_data,
    output reg                  done
);
    wire[319:0]param_word;
    mamba_spe_dt_param_rom #(.BLOCK_ID(BLOCK_ID))u_param_rom(
        .clk(clk),.en(1'b1),.addr(issue_group),.data(param_word));

    reg active;
    reg [7:0] pixel;
    reg [1:0] token;
    reg [8:0] dt_code;
    reg issue_group;
    reg rom_valid;reg[7:0]rom_pixel;reg[1:0]rom_token;reg rom_group;reg[8:0]rom_dt_code;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            active <= 1'b0;
            issue_group <= 1'b0;
            rom_valid<=0;
        end else begin
            rom_valid<=active;
            rom_pixel<=pixel;rom_token<=token;rom_group<=issue_group;
            rom_dt_code<=dt_code;
            if (active) begin
                if (issue_group) begin
                    active <= 1'b0;
                    issue_group <= 1'b0;
                end else begin
                    issue_group <= 1'b1;
                end
            end

            if (in_valid) begin
                active <= 1'b1;
                pixel <= in_pixel_addr;
                token <= in_token;
                dt_code <= in_data;
                issue_group <= 1'b0;
            end
        end
    end

    wire signed [16:0] product [0:7];
    genvar lane;

    generate
        for (lane = 0; lane < 4; lane = lane + 1) begin : GEN_DOT_PAIR
            mamba_packed_signed_mult_2x9_3cyc u_mult (
                .clk(clk),.activation(rom_dt_code),
                .weight_low(param_word[(lane*2)*8 +:8]),
                .weight_high(param_word[(lane*2+1)*8 +:8]),
                .product_low(product[lane*2]),
                .product_high(product[lane*2+1]));
        end
    endgenerate

    reg mult_valid_d0;
    reg mult_valid_d1;
    reg mult_valid_d2;
    reg [7:0] mult_pixel_d0;
    reg [7:0] mult_pixel_d1;
    reg [7:0] mult_pixel_d2;
    reg [1:0] mult_token_d0;
    reg [1:0] mult_token_d1;
    reg [1:0] mult_token_d2;
    reg mult_group_d0;
    reg mult_group_d1;
    reg mult_group_d2;
    reg[255:0]spe_bias_d0,spe_bias_d1,spe_bias_d2;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            mult_valid_d0 <= 1'b0;
            mult_valid_d1 <= 1'b0;
            mult_valid_d2 <= 1'b0;
        end else begin
            mult_valid_d0 <= rom_valid;
            mult_valid_d1 <= mult_valid_d0;
            mult_valid_d2 <= mult_valid_d1;
        end
        mult_pixel_d0 <= rom_pixel;
        mult_pixel_d1 <= mult_pixel_d0;
        mult_pixel_d2 <= mult_pixel_d1;
        mult_token_d0 <= rom_token;
        mult_token_d1 <= mult_token_d0;
        mult_token_d2 <= mult_token_d1;
        mult_group_d0 <= rom_group;
        mult_group_d1 <= mult_group_d0;
        mult_group_d2 <= mult_group_d1;
        spe_bias_d0<=param_word[319:64];
        spe_bias_d1<=spe_bias_d0;
        spe_bias_d2<=spe_bias_d1;
    end

    reg accumulator_valid;
    reg [7:0] accumulator_pixel;
    reg [1:0] accumulator_token;
    reg accumulator_group;
    reg signed [20:0] accumulator [0:7];
    integer add_lane;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            accumulator_valid <= 1'b0;
        end else begin
            accumulator_valid <= mult_valid_d2;
        end
        accumulator_pixel <= mult_pixel_d2;
        accumulator_token <= mult_token_d2;
        accumulator_group <= mult_group_d2;
        for (add_lane = 0; add_lane < 8;
             add_lane = add_lane + 1) begin
            accumulator[add_lane]
                <= $signed({{4{product[add_lane][16]}},
                            product[add_lane]})
                 + $signed(spe_bias_d2[add_lane*32 +:21]);
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
                .B(REQUANT_MULTIPLIER),
                .P(requant_product[requant_lane])
            );
        end
    endgenerate

    reg rq_valid_d0;
    reg rq_valid_d1;
    reg rq_valid_d2;
    reg [7:0] rq_pixel_d0;
    reg [7:0] rq_pixel_d1;
    reg [7:0] rq_pixel_d2;
    reg [1:0] rq_token_d0;
    reg [1:0] rq_token_d1;
    reg [1:0] rq_token_d2;
    reg rq_group_d0;
    reg rq_group_d1;
    reg rq_group_d2;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_valid_d0 <= 1'b0;
            rq_valid_d1 <= 1'b0;
            rq_valid_d2 <= 1'b0;
        end else begin
            rq_valid_d0 <= accumulator_valid;
            rq_valid_d1 <= rq_valid_d0;
            rq_valid_d2 <= rq_valid_d1;
        end
        rq_pixel_d0 <= accumulator_pixel;
        rq_pixel_d1 <= rq_pixel_d0;
        rq_pixel_d2 <= rq_pixel_d1;
        rq_token_d0 <= accumulator_token;
        rq_token_d1 <= rq_token_d0;
        rq_token_d2 <= rq_token_d1;
        rq_group_d0 <= accumulator_group;
        rq_group_d1 <= rq_group_d0;
        rq_group_d2 <= rq_group_d1;
    end

    function [7:0] round_shift_clip_int8;
        input signed [36:0] product_value;
        input signed [6:0] shift_value;
        reg signed [63:0] value;
        reg signed [63:0] magnitude;
        reg signed [63:0] half_lsb;
        reg signed [63:0] rounded;
        begin
            value = {{27{product_value[36]}}, product_value};
            if (shift_value > 0) begin
                half_lsb = 64'sd1 <<< (shift_value - 1'b1);
                magnitude = (value < 0) ? -value : value;
                rounded = (magnitude + half_lsb) >>> shift_value;
                if (value < 0)
                    rounded = -rounded;
            end else if (shift_value < 0) begin
                rounded = value <<< (-shift_value);
            end else begin
                rounded = value;
            end
            if (rounded > 127)
                round_shift_clip_int8 = 8'h7f;
            else if (rounded < -128)
                round_shift_clip_int8 = 8'h80;
            else
                round_shift_clip_int8 = rounded[7:0];
        end
    endfunction

    integer output_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            out_valid <= 1'b0;
            done <= 1'b0;
        end else begin
            out_valid <= rq_valid_d2;
            done <= rq_valid_d2&&(rq_pixel_d2==PIXEL_COUNT-1)
                              &&(rq_token_d2==2'd3)&&rq_group_d2;
        end
        out_pixel_addr <= rq_pixel_d2;
        out_token <= rq_token_d2;
        out_channel_base <= {rq_group_d2, 3'b000};
        for (output_lane = 0; output_lane < 8;
             output_lane = output_lane + 1)
            out_data[output_lane*8 +: 8]
                <= round_shift_clip_int8(
                    requant_product[output_lane], REQUANT_SHIFT);
    end
endmodule

`default_nettype wire
