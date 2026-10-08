`timescale 1ns/1ps
// Full Patch -> B0 -> Pool1 -> B1 -> Pool2 -> B2 -> classifier network.
// Registered fabric boundary, NOT a PCIe/DDR/HDMI board top.
module nl_fullnet_shell (
    input wire clk, rst_n, tile_wr_en, start,
    input wire [7:0] tile_wr_addr,
    input wire [127:0] tile_wr_data,
    output reg out_valid, out_sof, out_eof, out_tile_context,
    output reg [3:0] out_pixel_addr, out_class,
    output reg [71:0] out_logits,
    output reg feature_done, done, patch_busy
);
    wire core_clk;
    BUFGCE u_clk (.I(clk), .CE(1'b1), .O(core_clk));
    (* keep = "true" *) reg rst_q;
    reg wr_q, start_q;
    reg [7:0] addr_q;
    reg [127:0] data_q;
    wire v, sof, eof, ctx, fd, dn, busy;
    wire [3:0] pixel, cls;
    wire [71:0] logits;
    always @(posedge core_clk) begin
        rst_q <= rst_n;
        wr_q <= tile_wr_en;
        start_q <= start;
        addr_q <= tile_wr_addr;
        data_q <= tile_wr_data;
        out_valid <= v;
        out_sof <= sof;
        out_eof <= eof;
        out_tile_context <= ctx;
        out_pixel_addr <= pixel;
        out_class <= cls;
        out_logits <= logits;
        feature_done <= fd;
        done <= dn;
        patch_busy <= busy;
    end
    mambahsi_full_network_stream_top u_network (
        .clk(core_clk), .rst_n(rst_q),
        .tile_wr_en(wr_q), .tile_wr_addr(addr_q), .tile_wr_data(data_q),
        .start(start_q), .out_valid(v), .out_pixel_addr(pixel),
        .out_logits(logits), .out_class(cls), .out_sof(sof), .out_eof(eof),
        .out_tile_context(ctx), .feature_done(fd), .done(dn), .patch_busy(busy)
    );
endmodule
