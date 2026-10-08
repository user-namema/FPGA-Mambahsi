`timescale 1ns / 1ps
`default_nettype none

// Connect this module directly to patch_both_x_dt_pipeline's exposed u, B/C,
// and dt streams.  Each core tracks four ordered completion watermarks and
// starts record r as soon as U/DT/B/C for r are all present; it no longer waits
// for the complete branch input frame.
module both_ssm64_stream #(
    parameter integer BLOCK_ID = 0,
    parameter integer PIXEL_COUNT = (BLOCK_ID==0)?256:(BLOCK_ID==1)?64:16
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,

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

    output wire                 spa_out_valid,
    output wire [9:0]           spa_out_record_addr,
    output wire [5:0]           spa_out_channel_base,
    output wire [191:0]         spa_out_y_q24,
    output wire                 spa_done,
    output wire                 spe_out_valid,
    output wire [9:0]           spe_out_record_addr,
    output wire [5:0]           spe_out_channel_base,
    output wire [191:0]         spe_out_y_q24,
    output wire                 spe_done,
    output wire                 spa_input_record_ready,
    output wire                 spa_input_record_consumed,
    output wire                 all_done
);
    reg spa_done_finished;
    reg spe_done_finished;
    // Branch-local endpoints preserve same-cycle frame semantics.  MAX_FANOUT
    // permits physical replication without delaying the first capture beat.
    wire spa_frame_start_local;
    wire spe_frame_start_local;
    ssm_local_control_buffer u_spa_frame_start_buffer (
        .control_in(frame_start), .control_out(spa_frame_start_local));
    ssm_local_control_buffer u_spe_frame_start_buffer (
        .control_in(frame_start), .control_out(spe_frame_start_local));

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            spa_done_finished <= 1'b0;
            spe_done_finished <= 1'b0;
        end else begin
            if (spa_done)
                spa_done_finished <= 1'b1;
            if (spe_done)
                spe_done_finished <= 1'b1;
        end
    end

    wire spa_busy;
    wire spe_busy;
    ssm64_parallel_core #(
        .BLOCK_ID(BLOCK_ID),
        .IS_SPE(0),
        .CHANNEL_COUNT(64),
        .RECORD_COUNT(PIXEL_COUNT),
        .GROUP_COUNT(16),
        .K_FRACTION_BITS((BLOCK_ID == 2) ? 23 : 24),
        .RESET_PERIOD(0),
        .STREAM_INPUT(BLOCK_ID == 0),
        .CONTINUOUS_TILES(1)
    ) u_spa_ssm (
        .clk(clk),
        .rst_n(rst_n),
        .u_valid(spa_u_valid),
        .u_record_addr({2'd0, spa_u_pixel_addr}),
        .u_channel_base(spa_u_channel_base),
        .u_data(spa_u_data),
        .dt_valid(spa_dt_valid),
        .dt_record_addr({2'd0, spa_dt_pixel_addr}),
        .dt_channel_base(spa_dt_channel_base),
        .dt_data(spa_dt_data),
        .b_valid(spa_b_valid),
        .b_record_addr({2'd0, spa_b_pixel_addr}),
        .b_data(spa_b_data),
        .c_valid(spa_c_valid),
        .c_record_addr({2'd0, spa_c_pixel_addr}),
        .c_data(spa_c_data),
        .cfg_lut_we(spa_cfg_lut_we),
        .cfg_lut_addr(spa_cfg_lut_addr),
        .cfg_lut_data(spa_cfg_lut_data),
        .frame_start(spa_frame_start_local),
        .compute_start(1'b0),
        .out_valid(spa_out_valid),
        .out_record_addr(spa_out_record_addr),
        .out_channel_base(spa_out_channel_base),
        .out_y_q24(spa_out_y_q24),
        .done(spa_done),
        .busy(spa_busy),
        .input_record_ready(spa_input_record_ready),
        .input_record_consumed(spa_input_record_consumed)
    );

    ssm64_parallel_core #(
        .BLOCK_ID(BLOCK_ID),
        .IS_SPE(1),
        .CHANNEL_COUNT(16),
        .RECORD_COUNT(PIXEL_COUNT*4),
        .GROUP_COUNT(4),
        .K_FRACTION_BITS(24),
        .RESET_PERIOD(4),
        // Spe U is produced before x/dt and has no four-slot credit return.
        // Keep its full record ring; a four-slot FIFO would overwrite U(0)
        // with U(4) before DT/B/C(0) arrive and deadlock ordered consumption.
        .STREAM_INPUT(0),
        .CONTINUOUS_TILES(1)
    ) u_spe_ssm (
        .clk(clk),
        .rst_n(rst_n),
        .u_valid(spe_u_valid),
        .u_record_addr({spe_u_pixel_addr, spe_u_token}),
        .u_channel_base({2'd0, spe_u_channel_base}),
        .u_data(spe_u_data),
        .dt_valid(spe_dt_valid),
        .dt_record_addr({spe_dt_pixel_addr, spe_dt_token}),
        .dt_channel_base({2'd0, spe_dt_channel_base}),
        .dt_data(spe_dt_data),
        .b_valid(spe_b_valid),
        .b_record_addr({spe_b_pixel_addr, spe_b_token}),
        .b_data(spe_b_data),
        .c_valid(spe_c_valid),
        .c_record_addr({spe_c_pixel_addr, spe_c_token}),
        .c_data(spe_c_data),
        .cfg_lut_we(spe_cfg_lut_we),
        .cfg_lut_addr(spe_cfg_lut_addr),
        .cfg_lut_data(spe_cfg_lut_data),
        .frame_start(spe_frame_start_local),
        .compute_start(1'b0),
        .out_valid(spe_out_valid),
        .out_record_addr(spe_out_record_addr),
        .out_channel_base(spe_out_channel_base),
        .out_y_q24(spe_out_y_q24),
        .done(spe_done),
        .busy(spe_busy),
        .input_record_consumed()
    );

    assign all_done = (spa_done_finished || spa_done)
                   && (spe_done_finished || spe_done);
endmodule

`default_nettype wire
