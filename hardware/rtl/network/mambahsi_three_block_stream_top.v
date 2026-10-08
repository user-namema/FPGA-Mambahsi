`timescale 1ns/1ps
`default_nettype none

// Complete streaming feature extractor for one 16x16 tile:
// patch -> block0 -> 2x2 sum pool -> block1 -> 2x2 sum pool -> block2.
// All blocks receive the same frame_start pulse and independently wait for
// their first complete input record, so adjacent layers overlap naturally.
(* KEEP_HIERARCHY = "yes" *)
module mambahsi_three_block_stream_top(
    input wire clk,input wire rst_n,
    input wire tile_wr_en,input wire[7:0]tile_wr_addr,input wire[127:0]tile_wr_data,
    input wire start,
    output wire out_valid,output wire[7:0]out_pixel_addr,
    output wire[4:0]out_channel_base,output wire[63:0]out_data,
    output wire out_sof,output wire out_eof,
    output wire out_tile_context,output wire done,output wire patch_busy
);
    // Global start is a tile admission event.  Only the first accepted tile
    // initializes the streaming pipeline; subsequent tiles are separated by
    // local address-derived SOF/EOF markers and use alternating storage
    // contexts, so an arriving tile never clears the preceding tile's tail.
    reg pipeline_initialized;
    always @(posedge clk) begin
        if(!rst_n) pipeline_initialized<=1'b0;
        else if(start) pipeline_initialized<=1'b1;
    end
    wire pipeline_frame_start=start&&!pipeline_initialized;
    wire patch_valid;wire[7:0]patch_pixel;wire[4:0]patch_channel;
    wire[31:0]patch_data;wire patch_done;
    patch_embdding u_patch(
        .clk(clk),.rst_n(rst_n),.tile_wr_en(tile_wr_en),.tile_wr_addr(tile_wr_addr),
        .tile_wr_data(tile_wr_data),.cfg_weight_we(1'b0),.cfg_weight_addr(9'd0),
        .cfg_weight_data(8'd0),.cfg_bias_we(1'b0),.cfg_bias_addr(5'd0),
        .cfg_bias_data(32'd0),.cfg_requant_multiplier(32'd0),.cfg_requant_shift(7'd0),
        .start(start),.busy(patch_busy),.done(patch_done),.out_valid(patch_valid),
        .out_pixel_addr(patch_pixel),.out_channel_base(patch_channel),.out_data(patch_data));

    wire block0_valid;wire[7:0]block0_pixel;wire[4:0]block0_channel;
    wire[63:0]block0_data;wire block0_done;
    (* KEEP_HIERARCHY = "yes" *)
    mamba_block_rom_top #(.BLOCK_ID(0),.PIXEL_COUNT(256))u_block0(
        .clk(clk),.rst_n(rst_n),.frame_start(pipeline_frame_start),.in_valid(patch_valid),
        .in_pixel_addr(patch_pixel),.in_channel_base(patch_channel),.in_data(patch_data),
        .out_valid(block0_valid),.out_pixel_addr(block0_pixel),
        .out_channel_base(block0_channel),.out_data(block0_data),.done(block0_done));

    wire pool1_valid;wire[7:0]pool1_pixel;wire[4:0]pool1_channel;
    wire[39:0]pool1_data;wire pool1_done;
    (* KEEP_HIERARCHY = "yes" *)
    avgpool2x2_sum_stream #(.IN_SIDE(16))u_pool1(
        .clk(clk),.rst_n(rst_n),.frame_start(pipeline_frame_start),.in_valid(block0_valid),
        .in_pixel_addr(block0_pixel),.in_channel_base(block0_channel),.in_data(block0_data),
        .out_valid(pool1_valid),.out_pixel_addr(pool1_pixel),
        .out_channel_base(pool1_channel),.out_data(pool1_data),.done(pool1_done));

    wire block1_valid;wire[7:0]block1_pixel;wire[4:0]block1_channel;
    wire[63:0]block1_data;wire block1_done;
    (* KEEP_HIERARCHY = "yes" *)
    mamba_block_rom_top #(.BLOCK_ID(1),.PIXEL_COUNT(64))u_block1(
        .clk(clk),.rst_n(rst_n),.frame_start(pipeline_frame_start),.in_valid(pool1_valid),
        .in_pixel_addr(pool1_pixel),.in_channel_base(pool1_channel),.in_data(pool1_data),
        .out_valid(block1_valid),.out_pixel_addr(block1_pixel),
        .out_channel_base(block1_channel),.out_data(block1_data),.done(block1_done));

    wire pool2_valid;wire[7:0]pool2_pixel;wire[4:0]pool2_channel;
    wire[39:0]pool2_data;wire pool2_done;
    (* KEEP_HIERARCHY = "yes" *)
    avgpool2x2_sum_stream #(.IN_SIDE(8),.UNCONDITIONAL_PAYLOAD(1))u_pool2(
        .clk(clk),.rst_n(rst_n),.frame_start(pipeline_frame_start),.in_valid(block1_valid),
        .in_pixel_addr(block1_pixel),.in_channel_base(block1_channel),.in_data(block1_data),
        .out_valid(pool2_valid),.out_pixel_addr(pool2_pixel),
        .out_channel_base(pool2_channel),.out_data(pool2_data),.done(pool2_done));

    wire block2_valid;wire[7:0]block2_pixel;wire[4:0]block2_channel;
    wire[63:0]block2_data;wire block2_done;
    (* KEEP_HIERARCHY = "yes" *)
    mamba_block_rom_top #(.BLOCK_ID(2),.PIXEL_COUNT(16))u_block2(
        .clk(clk),.rst_n(rst_n),.frame_start(pipeline_frame_start),.in_valid(pool2_valid),
        .in_pixel_addr(pool2_pixel),.in_channel_base(pool2_channel),.in_data(pool2_data),
        .out_valid(block2_valid),.out_pixel_addr(block2_pixel),
        .out_channel_base(block2_channel),.out_data(block2_data),.done(block2_done));

    assign out_valid=block2_valid;
    assign out_pixel_addr=block2_pixel;
    assign out_channel_base=block2_channel;
    assign out_data=block2_data;
    assign out_sof=block2_valid&&(block2_pixel==0)&&(block2_channel==0);
    assign out_eof=block2_valid&&(block2_pixel==15)&&(block2_channel==24);
    reg output_tile_context_q;
    always @(posedge clk) begin
        if(!rst_n||pipeline_frame_start) output_tile_context_q<=1'b0;
        else if(out_eof) output_tile_context_q<=~output_tile_context_q;
    end
    assign out_tile_context=output_tile_context_q;
    assign done=block2_done;
endmodule

`default_nettype wire
