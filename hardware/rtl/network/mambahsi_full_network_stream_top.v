`timescale 1ns/1ps
`default_nettype none

// Complete one-tile inference pipeline, including the QAT classifier head.
(* KEEP_HIERARCHY = "yes" *)
module mambahsi_full_network_stream_top(
    input wire clk,input wire rst_n,
    input wire tile_wr_en,input wire[7:0]tile_wr_addr,
    input wire[127:0]tile_wr_data,input wire start,
    output wire out_valid,output wire[3:0]out_pixel_addr,
    output wire[71:0]out_logits,output wire[3:0]out_class,
    output wire out_sof,output wire out_eof,
    output wire out_tile_context,
    output wire feature_done,output wire done,
    output wire patch_busy
);
    wire feature_valid;wire[7:0]feature_pixel;wire[4:0]feature_channel;
    wire[63:0]feature_data;wire feature_sof,feature_eof,feature_context;
    reg network_initialized;
    always @(posedge clk) begin
        if(!rst_n) network_initialized<=1'b0;
        else if(start) network_initialized<=1'b1;
    end
    wire network_frame_start=start&&!network_initialized;
    (* KEEP_HIERARCHY = "yes" *)
    mambahsi_three_block_stream_top u_features(
        .clk(clk),.rst_n(rst_n),.tile_wr_en(tile_wr_en),
        .tile_wr_addr(tile_wr_addr),.tile_wr_data(tile_wr_data),.start(start),
        .out_valid(feature_valid),.out_pixel_addr(feature_pixel),
        .out_channel_base(feature_channel),.out_data(feature_data),
        .out_sof(feature_sof),.out_eof(feature_eof),
        .out_tile_context(feature_context), .patch_busy(patch_busy),
        .done(feature_done));

    (* KEEP_HIERARCHY = "yes" *)
    mamba_classifier_head_stream u_head(
        .clk(clk),.rst_n(rst_n),.frame_start(network_frame_start),
        .in_valid(feature_valid),.in_pixel_addr(feature_pixel),
        .in_channel_base(feature_channel),.in_data(feature_data),
        .out_valid(out_valid),.out_pixel_addr(out_pixel_addr),
        .out_logits(out_logits),.out_class(out_class),.done(done),.busy());
    assign out_sof=out_valid&&(out_pixel_addr==0);
    assign out_eof=out_valid&&(out_pixel_addr==15);
    reg output_tile_context_q;
    always @(posedge clk) begin
        if(!rst_n||network_frame_start) output_tile_context_q<=1'b0;
        else if(out_eof) output_tile_context_q<=~output_tile_context_q;
    end
    assign out_tile_context=output_tile_context_q;
    wire unused_feature_boundary=feature_sof^feature_eof^feature_context;
endmodule

`default_nettype wire
