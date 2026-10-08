`timescale 1ns / 1ps
`default_nettype none

// Complete block-0 datapath after the two selective scans:
// SSM Q24 -> SSM requant -> out_proj -> ReLU
//         -> Spa/Spe fusion -> fusion + 2*x block residual.
module both_mamba_block0_post_ssm #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=(BLOCK_ID==0)?256:(BLOCK_ID==1)?64:16,
    parameter signed [15:0] SPA_SSM_MULT=(BLOCK_ID==0)?16'sd16734:(BLOCK_ID==1)?16'sd27810:16'sd22551,
    parameter signed [6:0]  SPA_SSM_SHIFT=(BLOCK_ID==0)?7'sd37:(BLOCK_ID==1)?7'sd38:7'sd38,
    parameter signed [15:0] SPE_SSM_MULT=(BLOCK_ID==0)?16'sd26812:(BLOCK_ID==1)?16'sd25026:16'sd30765,
    parameter signed [6:0]  SPE_SSM_SHIFT=(BLOCK_ID==0)?7'sd39:(BLOCK_ID==1)?7'sd38:7'sd39,
    parameter signed [15:0] SPA_OUT_MULT=(BLOCK_ID==0)?16'sd30169:(BLOCK_ID==1)?16'sd21144:16'sd20037,
    parameter signed [6:0]  SPA_OUT_SHIFT=(BLOCK_ID==0)?7'sd19:(BLOCK_ID==1)?7'sd19:7'sd19,
    parameter signed [15:0] SPE_OUT_MULT=(BLOCK_ID==0)?16'sd18683:(BLOCK_ID==1)?16'sd29670:16'sd22737,
    parameter signed [6:0]  SPE_OUT_SHIFT=(BLOCK_ID==0)?7'sd17:(BLOCK_ID==1)?7'sd18:7'sd18,
    parameter signed [15:0] FUSION_A_MULT=(BLOCK_ID==0)?16'sd17376:(BLOCK_ID==1)?16'sd18154:16'sd18548,
    parameter signed [6:0]  FUSION_A_SHIFT=(BLOCK_ID==0)?7'sd15:(BLOCK_ID==1)?7'sd15:7'sd15,
    parameter signed [15:0] FUSION_B_MULT=(BLOCK_ID==0)?16'sd31143:(BLOCK_ID==1)?16'sd29382:16'sd28535,
    parameter signed [6:0]  FUSION_B_SHIFT=(BLOCK_ID==0)?7'sd16:(BLOCK_ID==1)?7'sd16:7'sd16,
    parameter signed [15:0] BLOCK_A_MULT=(BLOCK_ID==0)?16'sd31651:(BLOCK_ID==1)?16'sd25163:16'sd29183,
    parameter signed [6:0]  BLOCK_A_SHIFT=(BLOCK_ID==0)?7'sd16:(BLOCK_ID==1)?7'sd17:7'sd17,
    parameter signed [15:0] BLOCK_B_MULT=(BLOCK_ID==0)?16'sd16988:(BLOCK_ID==1)?16'sd29741:16'sd28560,
    parameter signed [6:0]  BLOCK_B_SHIFT=(BLOCK_ID==0)?7'sd15:(BLOCK_ID==1)?7'sd17:7'sd18
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,

    input  wire                 input_valid,
    input  wire [7:0]           input_pixel_addr,
    input  wire [4:0]           input_channel_base,
    input  wire [((BLOCK_ID==0)?32:40)-1:0] input_data,

    input  wire                 spa_ssm_valid,
    input  wire [9:0]           spa_ssm_record_addr,
    input  wire [5:0]           spa_ssm_channel_base,
    input  wire [191:0]         spa_ssm_y_q24,
    input  wire                 spe_ssm_valid,
    input  wire [9:0]           spe_ssm_record_addr,
    input  wire [5:0]           spe_ssm_channel_base,
    input  wire [191:0]         spe_ssm_y_q24,

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

    output wire                 spa_ssm_int_valid,
    output wire [9:0]           spa_ssm_int_record,
    output wire [5:0]           spa_ssm_int_channel,
    output wire [31:0]          spa_ssm_int_data,
    output wire                 spe_ssm_int_valid,
    output wire [9:0]           spe_ssm_int_record,
    output wire [5:0]           spe_ssm_int_channel,
    output wire [31:0]          spe_ssm_int_data,
    output wire                 spa_out_valid,
    output wire [7:0]           spa_out_pixel,
    output wire [4:0]           spa_out_channel,
    output wire [31:0]          spa_out_raw_data,
    output wire [31:0]          spa_out_relu_data,
    output wire                 spe_out_valid,
    output wire [7:0]           spe_out_pixel,
    output wire [1:0]           spe_out_token,
    output wire [63:0]          spe_out_raw_data,
    output wire [63:0]          spe_out_relu_data,
    output wire                 spa_result_valid,
    output wire [15:0]          spa_result_metadata,
    output wire [31:0]          spa_result_data,
    output wire                 spe_result_valid,
    output wire [15:0]          spe_result_metadata,
    output wire [63:0]          spe_result_data,
    output wire                 fusion_valid,
    output wire [15:0]          fusion_metadata,
    output wire [63:0]          fusion_data,
    output wire                 out_valid,
    output wire [7:0]           out_pixel_addr,
    output wire [4:0]           out_channel_base,
    output wire [63:0]          out_data,
    output wire                 done
);
    // Keep the frame control physically local to each post-SSM consumer.
    // These are intentionally independent registers rather than one replicated
    // top-level net: placement can now keep each reset pulse beside the logic
    // whose valid/FSM state it clears.
    (* keep = "true", max_fanout = 32 *) reg spa_rq_frame_start_q;
    (* keep = "true", max_fanout = 32 *) reg spe_rq_frame_start_q;
    (* keep = "true", max_fanout = 32 *) reg spa_out_frame_start_q;
    (* keep = "true", max_fanout = 32 *) reg spe_out_frame_start_q;
    (* keep = "true", max_fanout = 32 *) reg fusion_frame_start_q;
    always @(posedge clk) begin
        if (!rst_n) begin
            spa_rq_frame_start_q  <= 1'b0;
            spe_rq_frame_start_q  <= 1'b0;
            spa_out_frame_start_q <= 1'b0;
            spe_out_frame_start_q <= 1'b0;
            fusion_frame_start_q  <= 1'b0;
        end else begin
            spa_rq_frame_start_q  <= frame_start;
            spe_rq_frame_start_q  <= frame_start;
            spa_out_frame_start_q <= frame_start;
            spe_out_frame_start_q <= frame_start;
            fusion_frame_start_q  <= frame_start;
        end
    end

    ssm_requant_stream u_spa_ssm_requant (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(spa_rq_frame_start_q),
        .in_valid(spa_ssm_valid),
        .in_record_addr(spa_ssm_record_addr),
        .in_channel_base(spa_ssm_channel_base),
        .in_y_q24(spa_ssm_y_q24),
        .cfg_multiplier(SPA_SSM_MULT),
        .cfg_shift(SPA_SSM_SHIFT),
        .out_valid(spa_ssm_int_valid),
        .out_record_addr(spa_ssm_int_record),
        .out_channel_base(spa_ssm_int_channel),
        .out_data(spa_ssm_int_data)
    );

    ssm_requant_stream u_spe_ssm_requant (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(spe_rq_frame_start_q),
        .in_valid(spe_ssm_valid),
        .in_record_addr(spe_ssm_record_addr),
        .in_channel_base(spe_ssm_channel_base),
        .in_y_q24(spe_ssm_y_q24),
        .cfg_multiplier(SPE_SSM_MULT),
        .cfg_shift(SPE_SSM_SHIFT),
        .out_valid(spe_ssm_int_valid),
        .out_record_addr(spe_ssm_int_record),
        .out_channel_base(spe_ssm_int_channel),
        .out_data(spe_ssm_int_data)
    );

    wire spa_out_done;
    spa_out_proj_rom_stream #(
        .BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(SPA_OUT_MULT),
        .FIXED_SHIFT(SPA_OUT_SHIFT)
    ) u_spa_out_proj (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(spa_out_frame_start_q),
        .in_valid(spa_ssm_int_valid),
        .in_record_addr(spa_ssm_int_record),
        .in_channel_base(spa_ssm_int_channel),
        .in_data(spa_ssm_int_data),
        .cfg_weight_we(cfg_spa_out_weight_we),
        .cfg_weight_addr(cfg_spa_out_weight_addr),
        .cfg_weight_data(cfg_spa_out_weight_data),
        .cfg_bias_we(cfg_spa_out_bias_we),
        .cfg_bias_addr(cfg_spa_out_bias_addr),
        .cfg_bias_data(cfg_spa_out_bias_data),
        .cfg_multiplier(cfg_spa_out_multiplier),
        .cfg_shift(cfg_spa_out_shift),
        .out_valid(spa_out_valid),
        .out_pixel_addr(spa_out_pixel),
        .out_channel_base(spa_out_channel),
        .out_raw_data(spa_out_raw_data),
        .out_data(spa_out_relu_data),
        .done(spa_out_done)
    );

    wire spe_out_done;
    spe_out_proj_rom_stream #(
        .BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.FIXED_MULTIPLIER(SPE_OUT_MULT),
        .FIXED_SHIFT(SPE_OUT_SHIFT)
    ) u_spe_out_proj (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(spe_out_frame_start_q),
        .in_valid(spe_ssm_int_valid),
        .in_record_addr(spe_ssm_int_record),
        .in_channel_base(spe_ssm_int_channel),
        .in_data(spe_ssm_int_data),
        .cfg_weight_we(cfg_spe_out_weight_we),
        .cfg_weight_addr(cfg_spe_out_weight_addr),
        .cfg_weight_data(cfg_spe_out_weight_data),
        .cfg_bias_we(cfg_spe_out_bias_we),
        .cfg_bias_addr(cfg_spe_out_bias_addr),
        .cfg_bias_data(cfg_spe_out_bias_data),
        .cfg_multiplier(cfg_spe_out_multiplier),
        .cfg_shift(cfg_spe_out_shift),
        .out_valid(spe_out_valid),
        .out_pixel_addr(spe_out_pixel),
        .out_token(spe_out_token),
        .out_raw_data(spe_out_raw_data),
        .out_data(spe_out_relu_data),
        .done(spe_out_done)
    );

    block0_residual_fusion_stream #(
        .BLOCK_ID(BLOCK_ID), .PIXEL_COUNT(PIXEL_COUNT),
        .FUSION_A_MULT(FUSION_A_MULT), .FUSION_A_SHIFT(FUSION_A_SHIFT),
        .FUSION_B_MULT(FUSION_B_MULT), .FUSION_B_SHIFT(FUSION_B_SHIFT),
        .BLOCK_A_MULT(BLOCK_A_MULT), .BLOCK_A_SHIFT(BLOCK_A_SHIFT),
        .BLOCK_B_MULT(BLOCK_B_MULT), .BLOCK_B_SHIFT(BLOCK_B_SHIFT)
    ) u_residual_fusion (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(fusion_frame_start_q),
        .input_valid(input_valid),
        .input_pixel_addr(input_pixel_addr),
        .input_channel_base(input_channel_base),
        .input_data(input_data),
        .spa_valid(spa_out_valid),
        .spa_pixel_addr(spa_out_pixel),
        .spa_channel_base(spa_out_channel),
        .spa_data(spa_out_relu_data),
        .spa_done(spa_out_done),
        .spe_valid(spe_out_valid),
        .spe_pixel_addr(spe_out_pixel),
        .spe_token(spe_out_token),
        .spe_data(spe_out_relu_data),
        .spe_done(spe_out_done),
        .cfg_fusion_a_multiplier(FUSION_A_MULT),
        .cfg_fusion_a_shift(FUSION_A_SHIFT),
        .cfg_fusion_b_multiplier(FUSION_B_MULT),
        .cfg_fusion_b_shift(FUSION_B_SHIFT),
        .cfg_block_a_multiplier(BLOCK_A_MULT),
        .cfg_block_a_shift(BLOCK_A_SHIFT),
        .cfg_block_b_multiplier(BLOCK_B_MULT),
        .cfg_block_b_shift(BLOCK_B_SHIFT),
        .spa_result_valid(spa_result_valid),
        .spa_result_metadata(spa_result_metadata),
        .spa_result_data(spa_result_data),
        .spe_result_valid(spe_result_valid),
        .spe_result_metadata(spe_result_metadata),
        .spe_result_data(spe_result_data),
        .fusion_valid(fusion_valid),
        .fusion_metadata(fusion_metadata),
        .fusion_data(fusion_data),
        .out_valid(out_valid),
        .out_pixel_addr(out_pixel_addr),
        .out_channel_base(out_channel_base),
        .out_data(out_data),
        .done(done)
    );
endmodule

`default_nettype wire
