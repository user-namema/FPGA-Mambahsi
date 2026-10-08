`timescale 1ns/1ps

module hdmi_async_fifo # (
    parameter WIDTH = 8, DEPTH = 64
) (
    input wire rst, wr_clk, rd_clk,
    input wire [WIDTH - 1 : 0] din,
    input wire push, pop,
    output wire [WIDTH - 1 : 0] dout,
    output wire full, empty
);

    // rst is a global asynchronous request, NOT the XPM FIFO reset input.
    // Use the vendor synchronizer and its SCOPED asynchronous-input constraints.
    // Catch a short request and stretch it across eight write-clock edges.
    wire wr_reset_pending;
    xpm_cdc_async_rst # (
        .DEST_SYNC_FF (8),
        .INIT_SYNC_FF (0),
        .RST_ACTIVE_HIGH (1)
    ) u_wr_reset_sync (
        .src_arst (rst),
        .dest_clk (wr_clk),
        .dest_arst (wr_reset_pending)
    );

    // XPM CDC asserts asynchronously; XPM FIFO requires synchronous assertion
    // too. Keep this final register WITHOUT asynchronous reset.
    reg fifo_rst = 1'b1;
    always @(posedge wr_clk) begin
        fifo_rst <= wr_reset_pending;
    end

    // Use ONE registered source: no combinational reset merge across domains.
    // wr_reset_pending asserts even when wr_clk is stopped. On release, the
    // FIFO's rd_rst_busy covers its final write-clock reset cycle and recovery;
    // rd_reset_pending independently guards asynchronous reset entry.
    wire rd_reset_pending;
    xpm_cdc_async_rst # (
        .DEST_SYNC_FF (3),
        .INIT_SYNC_FF (0),
        .RST_ACTIVE_HIGH (1)
    ) u_rd_reset_sync (
        .src_arst (wr_reset_pending),
        .dest_clk (rd_clk),
        .dest_arst (rd_reset_pending)
    );

    wire raw_full, raw_empty, wb, rb;
    assign full = raw_full | wb | wr_reset_pending | fifo_rst;
    assign empty = raw_empty | rb | rd_reset_pending;
    xpm_fifo_async # (
        .FIFO_MEMORY_TYPE ("block"),
        .FIFO_WRITE_DEPTH (DEPTH),
        .WRITE_DATA_WIDTH (WIDTH),
        .READ_DATA_WIDTH (WIDTH),
        .READ_MODE ("fwft"),
        .FIFO_READ_LATENCY (0),
        .CDC_SYNC_STAGES (2),
        .DOUT_RESET_VALUE ("0"),
        .ECC_MODE ("no_ecc"),
        .USE_ADV_FEATURES ("0000"),
        .RD_DATA_COUNT_WIDTH ($clog2 (DEPTH) + 1),
        .WR_DATA_COUNT_WIDTH ($clog2 (DEPTH) + 1)
    ) u_fifo (
        .rst (fifo_rst),
        .wr_clk (wr_clk),
        .rd_clk (rd_clk),
        .din (din),
        .wr_en (push && ! full),
        .rd_en (pop && ! empty),
        .dout (dout),
        .full (raw_full),
        .empty (raw_empty),
        .wr_rst_busy (wb),
        .rd_rst_busy (rb),
        .sleep (1'b0),
        .injectsbiterr (1'b0),
        .injectdbiterr (1'b0),
        .almost_empty (),
        .almost_full (),
        .data_valid (),
        .dbiterr (),
        .sbiterr (),
        .overflow (),
        .underflow (),
        .prog_empty (),
        .prog_full (),
        .rd_data_count (),
        .wr_data_count (),
        .wr_ack ()
    );
endmodule
