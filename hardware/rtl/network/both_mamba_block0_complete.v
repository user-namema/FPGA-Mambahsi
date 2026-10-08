`timescale 1ns / 1ps
`default_nettype none

// Complete block-0 tail connected to patch_both_x_dt_pipeline outputs.
// Together these two modules form the complete first Mamba block RTL.
module both_mamba_block0_complete #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=(BLOCK_ID==0)?256:(BLOCK_ID==1)?64:16
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,

    input  wire                 block_input_valid,
    input  wire [7:0]           block_input_pixel_addr,
    input  wire [4:0]           block_input_channel_base,
    input  wire [((BLOCK_ID==0)?32:40)-1:0] block_input_data,

    input  wire                 spa_u_valid,
    input  wire [7:0]           spa_u_pixel_addr,
    input  wire [5:0]           spa_u_channel_base,
    input  wire [63:0]          spa_u_data,
    input  wire                 spa_u_done,
    input  wire                 spa_dt_valid,
    input  wire [7:0]           spa_dt_pixel_addr,
    input  wire [5:0]           spa_dt_channel_base,
    input  wire [63:0]          spa_dt_data,
    input  wire                 spa_dt_done,
    input  wire                 spa_b_valid,
    input  wire [7:0]           spa_b_pixel_addr,
    input  wire [127:0]         spa_b_data,
    input  wire                 spa_c_valid,
    input  wire [7:0]           spa_c_pixel_addr,
    input  wire [127:0]         spa_c_data,
    input  wire                 spa_x_done,

    input  wire                 spe_u_valid,
    input  wire [7:0]           spe_u_pixel_addr,
    input  wire [1:0]           spe_u_token,
    input  wire [3:0]           spe_u_channel_base,
    input  wire [63:0]          spe_u_data,
    input  wire                 spe_u_done,
    input  wire                 spe_dt_valid,
    input  wire [7:0]           spe_dt_pixel_addr,
    input  wire [1:0]           spe_dt_token,
    input  wire [3:0]           spe_dt_channel_base,
    input  wire [63:0]          spe_dt_data,
    input  wire                 spe_dt_done,
    input  wire                 spe_b_valid,
    input  wire [7:0]           spe_b_pixel_addr,
    input  wire [1:0]           spe_b_token,
    input  wire [127:0]         spe_b_data,
    input  wire                 spe_c_valid,
    input  wire [7:0]           spe_c_pixel_addr,
    input  wire [1:0]           spe_c_token,
    input  wire [127:0]         spe_c_data,
    input  wire                 spe_x_done,

    input  wire                 spa_cfg_lut_we,
    input  wire [7:0]           spa_cfg_lut_addr,
    input  wire [418:0]         spa_cfg_lut_data,
    input  wire                 spe_cfg_lut_we,
    input  wire [7:0]           spe_cfg_lut_addr,
    input  wire [418:0]         spe_cfg_lut_data,

    input  wire signed [15:0]   cfg_spa_ssm_multiplier,
    input  wire signed [6:0]    cfg_spa_ssm_shift,
    input  wire signed [15:0]   cfg_spe_ssm_multiplier,
    input  wire signed [6:0]    cfg_spe_ssm_shift,
    input  wire                 cfg_spa_out_weight_we,
    input  wire [10:0]          cfg_spa_out_weight_addr,
    input  wire signed [7:0]    cfg_spa_out_weight_data,
    input  wire                 cfg_spa_out_bias_we,
    input  wire [4:0]           cfg_spa_out_bias_addr,
    input  wire signed [31:0]   cfg_spa_out_bias_data,
    input  wire signed [15:0]   cfg_spa_out_multiplier,
    input  wire signed [6:0]    cfg_spa_out_shift,
    input  wire                 cfg_spe_out_weight_we,
    input  wire [6:0]           cfg_spe_out_weight_addr,
    input  wire signed [7:0]    cfg_spe_out_weight_data,
    input  wire                 cfg_spe_out_bias_we,
    input  wire [2:0]           cfg_spe_out_bias_addr,
    input  wire signed [31:0]   cfg_spe_out_bias_data,
    input  wire signed [15:0]   cfg_spe_out_multiplier,
    input  wire signed [6:0]    cfg_spe_out_shift,
    input  wire signed [15:0]   cfg_fusion_a_multiplier,
    input  wire signed [6:0]    cfg_fusion_a_shift,
    input  wire signed [15:0]   cfg_fusion_b_multiplier,
    input  wire signed [6:0]    cfg_fusion_b_shift,
    input  wire signed [15:0]   cfg_block_a_multiplier,
    input  wire signed [6:0]    cfg_block_a_shift,
    input  wire signed [15:0]   cfg_block_b_multiplier,
    input  wire signed [6:0]    cfg_block_b_shift,

    output wire                 out_valid,
    output wire [7:0]           out_pixel_addr,
    output wire [4:0]           out_channel_base,
    output wire [63:0]          out_data,
    output wire                 spa_input_record_ready,
    output wire                 spa_input_record_consumed,
    output wire                 done
);
    wire spa_ssm_valid;
    wire [9:0] spa_ssm_record;
    wire [5:0] spa_ssm_channel;
    wire [191:0] spa_ssm_y;
    wire spa_ssm_done;
    wire spe_ssm_valid;
    wire [9:0] spe_ssm_record;
    wire [5:0] spe_ssm_channel;
    wire [191:0] spe_ssm_y;
    wire spe_ssm_done;
    wire both_ssm_done;

    // The post-SSM cluster gets its own retained pulse root.  The SSM core
    // continues to use the block-local frame_start supplied by its parent.
    (* KEEP = "TRUE", MAX_FANOUT = 64 *) reg post_frame_start_q;
    always @(posedge clk) begin
        if (!rst_n)
            post_frame_start_q <= 1'b0;
        else
            post_frame_start_q <= frame_start;
    end

    both_ssm64_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT)) u_both_ssm (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .spa_u_valid(spa_u_valid),
        .spa_u_pixel_addr(spa_u_pixel_addr),
        .spa_u_channel_base(spa_u_channel_base),
        .spa_u_data(spa_u_data),
        .spa_u_done(spa_u_done),
        .spa_dt_valid(spa_dt_valid),
        .spa_dt_pixel_addr(spa_dt_pixel_addr),
        .spa_dt_channel_base(spa_dt_channel_base),
        .spa_dt_data(spa_dt_data),
        .spa_dt_done(spa_dt_done),
        .spa_b_valid(spa_b_valid),
        .spa_b_pixel_addr(spa_b_pixel_addr),
        .spa_b_data(spa_b_data),
        .spa_c_valid(spa_c_valid),
        .spa_c_pixel_addr(spa_c_pixel_addr),
        .spa_c_data(spa_c_data),
        .spa_x_done(spa_x_done),
        .spe_u_valid(spe_u_valid),
        .spe_u_pixel_addr(spe_u_pixel_addr),
        .spe_u_token(spe_u_token),
        .spe_u_channel_base(spe_u_channel_base),
        .spe_u_data(spe_u_data),
        .spe_u_done(spe_u_done),
        .spe_dt_valid(spe_dt_valid),
        .spe_dt_pixel_addr(spe_dt_pixel_addr),
        .spe_dt_token(spe_dt_token),
        .spe_dt_channel_base(spe_dt_channel_base),
        .spe_dt_data(spe_dt_data),
        .spe_dt_done(spe_dt_done),
        .spe_b_valid(spe_b_valid),
        .spe_b_pixel_addr(spe_b_pixel_addr),
        .spe_b_token(spe_b_token),
        .spe_b_data(spe_b_data),
        .spe_c_valid(spe_c_valid),
        .spe_c_pixel_addr(spe_c_pixel_addr),
        .spe_c_token(spe_c_token),
        .spe_c_data(spe_c_data),
        .spe_x_done(spe_x_done),
        .spa_cfg_lut_we(spa_cfg_lut_we),
        .spa_cfg_lut_addr(spa_cfg_lut_addr),
        .spa_cfg_lut_data(spa_cfg_lut_data),
        .spe_cfg_lut_we(spe_cfg_lut_we),
        .spe_cfg_lut_addr(spe_cfg_lut_addr),
        .spe_cfg_lut_data(spe_cfg_lut_data),
        .spa_out_valid(spa_ssm_valid),
        .spa_out_record_addr(spa_ssm_record),
        .spa_out_channel_base(spa_ssm_channel),
        .spa_out_y_q24(spa_ssm_y),
        .spa_done(spa_ssm_done),
        .spe_out_valid(spe_ssm_valid),
        .spe_out_record_addr(spe_ssm_record),
        .spe_out_channel_base(spe_ssm_channel),
        .spe_out_y_q24(spe_ssm_y),
        .spe_done(spe_ssm_done),
        .spa_input_record_ready(spa_input_record_ready),
        .spa_input_record_consumed(spa_input_record_consumed),
        .all_done(both_ssm_done)
    );

    wire unused_spa_ssm_int_valid;
    wire [9:0] unused_spa_ssm_int_record;
    wire [5:0] unused_spa_ssm_int_channel;
    wire [31:0] unused_spa_ssm_int_data;
    wire unused_spe_ssm_int_valid;
    wire [9:0] unused_spe_ssm_int_record;
    wire [5:0] unused_spe_ssm_int_channel;
    wire [31:0] unused_spe_ssm_int_data;
    wire unused_spa_out_valid;
    wire [7:0] unused_spa_out_pixel;
    wire [4:0] unused_spa_out_channel;
    wire [31:0] unused_spa_out_raw;
    wire [31:0] unused_spa_out_relu;
    wire unused_spe_out_valid;
    wire [7:0] unused_spe_out_pixel;
    wire [1:0] unused_spe_out_token;
    wire [63:0] unused_spe_out_raw;
    wire [63:0] unused_spe_out_relu;
    wire unused_spa_result_valid;
    wire [15:0] unused_spa_result_meta;
    wire [31:0] unused_spa_result_data;
    wire unused_spe_result_valid;
    wire [15:0] unused_spe_result_meta;
    wire [63:0] unused_spe_result_data;
    wire unused_fusion_valid;
    wire [15:0] unused_fusion_meta;
    wire [63:0] unused_fusion_data;

    both_mamba_block0_post_ssm #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT)) u_post_ssm (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(post_frame_start_q),
        .input_valid(block_input_valid),
        .input_pixel_addr(block_input_pixel_addr),
        .input_channel_base(block_input_channel_base),
        .input_data(block_input_data),
        .spa_ssm_valid(spa_ssm_valid),
        .spa_ssm_record_addr(spa_ssm_record),
        .spa_ssm_channel_base(spa_ssm_channel),
        .spa_ssm_y_q24(spa_ssm_y),
        .spe_ssm_valid(spe_ssm_valid),
        .spe_ssm_record_addr(spe_ssm_record),
        .spe_ssm_channel_base(spe_ssm_channel),
        .spe_ssm_y_q24(spe_ssm_y),
        .cfg_spa_ssm_multiplier(cfg_spa_ssm_multiplier),
        .cfg_spa_ssm_shift(cfg_spa_ssm_shift),
        .cfg_spe_ssm_multiplier(cfg_spe_ssm_multiplier),
        .cfg_spe_ssm_shift(cfg_spe_ssm_shift),
        .cfg_spa_out_weight_we(cfg_spa_out_weight_we),
        .cfg_spa_out_weight_addr(cfg_spa_out_weight_addr),
        .cfg_spa_out_weight_data(cfg_spa_out_weight_data),
        .cfg_spa_out_bias_we(cfg_spa_out_bias_we),
        .cfg_spa_out_bias_addr(cfg_spa_out_bias_addr),
        .cfg_spa_out_bias_data(cfg_spa_out_bias_data),
        .cfg_spa_out_multiplier(cfg_spa_out_multiplier),
        .cfg_spa_out_shift(cfg_spa_out_shift),
        .cfg_spe_out_weight_we(cfg_spe_out_weight_we),
        .cfg_spe_out_weight_addr(cfg_spe_out_weight_addr),
        .cfg_spe_out_weight_data(cfg_spe_out_weight_data),
        .cfg_spe_out_bias_we(cfg_spe_out_bias_we),
        .cfg_spe_out_bias_addr(cfg_spe_out_bias_addr),
        .cfg_spe_out_bias_data(cfg_spe_out_bias_data),
        .cfg_spe_out_multiplier(cfg_spe_out_multiplier),
        .cfg_spe_out_shift(cfg_spe_out_shift),
        .cfg_fusion_a_multiplier(cfg_fusion_a_multiplier),
        .cfg_fusion_a_shift(cfg_fusion_a_shift),
        .cfg_fusion_b_multiplier(cfg_fusion_b_multiplier),
        .cfg_fusion_b_shift(cfg_fusion_b_shift),
        .cfg_block_a_multiplier(cfg_block_a_multiplier),
        .cfg_block_a_shift(cfg_block_a_shift),
        .cfg_block_b_multiplier(cfg_block_b_multiplier),
        .cfg_block_b_shift(cfg_block_b_shift),
        .spa_ssm_int_valid(unused_spa_ssm_int_valid),
        .spa_ssm_int_record(unused_spa_ssm_int_record),
        .spa_ssm_int_channel(unused_spa_ssm_int_channel),
        .spa_ssm_int_data(unused_spa_ssm_int_data),
        .spe_ssm_int_valid(unused_spe_ssm_int_valid),
        .spe_ssm_int_record(unused_spe_ssm_int_record),
        .spe_ssm_int_channel(unused_spe_ssm_int_channel),
        .spe_ssm_int_data(unused_spe_ssm_int_data),
        .spa_out_valid(unused_spa_out_valid),
        .spa_out_pixel(unused_spa_out_pixel),
        .spa_out_channel(unused_spa_out_channel),
        .spa_out_raw_data(unused_spa_out_raw),
        .spa_out_relu_data(unused_spa_out_relu),
        .spe_out_valid(unused_spe_out_valid),
        .spe_out_pixel(unused_spe_out_pixel),
        .spe_out_token(unused_spe_out_token),
        .spe_out_raw_data(unused_spe_out_raw),
        .spe_out_relu_data(unused_spe_out_relu),
        .spa_result_valid(unused_spa_result_valid),
        .spa_result_metadata(unused_spa_result_meta),
        .spa_result_data(unused_spa_result_data),
        .spe_result_valid(unused_spe_result_valid),
        .spe_result_metadata(unused_spe_result_meta),
        .spe_result_data(unused_spe_result_data),
        .fusion_valid(unused_fusion_valid),
        .fusion_metadata(unused_fusion_meta),
        .fusion_data(unused_fusion_data),
        .out_valid(out_valid),
        .out_pixel_addr(out_pixel_addr),
        .out_channel_base(out_channel_base),
        .out_data(out_data),
        .done(done)
    );
endmodule

`default_nettype wire
