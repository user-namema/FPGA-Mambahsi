`timescale 1ns/1ps
// Application subsystem. ui_clk/pixel_clk supplied by ONE DDR4 MIG controller.
// rst_n must assert globally and release synchronously in each clock domain.

module up_output_ddr_hdmi # (
    parameter integer PROGRESSIVE = 0,
    parameter WIDTH = 340, HEIGHT = 610, PITCH = (
        (
            WIDTH + 63
        ) / 64
    ) * 64,
    parameter H_ACTIVE = 1280, H_SYNC = 40, H_BACK = 220, H_TOTAL = 1650,
    parameter V_ACTIVE = 720, V_SYNC = 5, V_BACK = 20, V_TOTAL = 750
) (
    input wire net_clk, ui_clk, pixel_clk, rst_n, calib, scene_start,
    input wire out_valid, out_sof, out_eof,
    input wire [3 : 0] out_pixel_addr,
    input wire [71 : 0] out_logits,
    output wire scene_ready,
    output wire frame_stored,
    output wire net_error, ddr_error, display_underflow,
    output wire video_hs, video_vs, video_de,
    output wire [23 : 0] video_rgb,
    output wire [28 : 0] app_addr,
    output wire [2 : 0] app_cmd,
    output wire app_en,
    input wire app_rdy,
    output wire [511 : 0] app_wdf_data,
    output wire [63 : 0] app_wdf_mask,
    output wire app_wdf_wren, app_wdf_end,
    input wire app_wdf_rdy,
    input wire [511 : 0] app_rd_data,
    input wire app_rd_data_valid,
    input wire rb_valid,
    output wire rb_ready,
    input wire [31:0] rb_byte_addr,
    output wire rb_response_valid,
    input wire rb_response_ready,
    output wire [511:0] rb_response_data
);

    // Each hdmi_async_fifo synchronizes this common reset request into its
    // own write domain: wrq=net_clk, requestq=pixel_clk, responseq=ui_clk.
    // nr/ur/pr below serve the surrounding logic, not the XPM FIFO rst pin.
    wire nr, ur, pr;
    hdmi_reset_sync rn (net_clk, rst_n, nr), ru (ui_clk, rst_n, ur), rp (pixel_clk, rst_n, pr);
    (* ASYNC_REG = "TRUE" *) reg cal0, cal1;
    reg started;
    wire wf, we, wp, wpush;
    wire [176 : 0] wd, wo;
    wire rv, rr, last, captured;
    wire [31 : 0] addr;
    wire [127 : 0] data;
    wire [15 : 0] keep;
    always @ (posedge net_clk) begin
        if (! nr) begin
            cal0 <= 0;
            cal1 <= 0;
            started <= 0;
        end
        else begin
            cal0 <= calib;
            cal1 <= cal0;
            if (scene_start && scene_ready) started <= 1;
        end
    end
    assign scene_ready = cal1 && ! started && ! wf;
    up_logits_to_rows # (
        .WIDTH (WIDTH),
        .HEIGHT (HEIGHT),
        .PITCH (PITCH)
    ) u_post (
        .clk (net_clk),
        .rst_n (nr),
        .scene_start (scene_start && scene_ready),
        .in_valid (out_valid),
        .in_sof (out_sof),
        .in_eof (out_eof),
        .in_pixel (out_pixel_addr),
        .in_logits (out_logits),
        .row_valid (rv),
        .row_ready (rr),
        .row_addr (addr),
        .row_data (data),
        .row_keep (keep),
        .row_last (last),
        .capture_done (captured),
        .error (net_error)
    );
    assign rr = ! wf;
    assign wd = { last, keep, addr, data };
    assign wpush = rv && rr;
    hdmi_async_fifo # (
        .WIDTH (177),
        .DEPTH (64)
    ) wrq (
        .rst (! rst_n),
        .wr_clk (net_clk),
        .rd_clk (ui_clk),
        .din (wd),
        .push (wpush),
        .pop (wp),
        .dout (wo),
        .full (wf),
        .empty (we)
    );
    wire qf, qe, qp, qpush;
    wire [9 : 0] qi, qo;
    hdmi_async_fifo # (
        .WIDTH (10),
        .DEPTH (16)
    ) requestq (
        .rst (! rst_n),
        .wr_clk (pixel_clk),
        .rd_clk (ui_clk),
        .din (qi),
        .push (qpush),
        .pop (qp),
        .dout (qo),
        .full (qf),
        .empty (qe)
    );
    wire rf, re, rp_pop, rpush;
    wire [524 : 0] ri, ro;
    hdmi_async_fifo # (
        .WIDTH (525),
        .DEPTH (16)
    ) responseq (
        .rst (! rst_n),
        .wr_clk (ui_clk),
        .rd_clk (pixel_clk),
        .din (ri),
        .push (rpush),
        .pop (rp_pop),
        .dout (ro),
        .full (rf),
        .empty (re)
    );
    up_ddr_label_store # (
        .PITCH (PITCH), .WIDTH(WIDTH), .HEIGHT(HEIGHT), .PROGRESSIVE(PROGRESSIVE)
    ) u_store (
        .clk (ui_clk),
        .rst_n (ur),
        .calib (calib),
        .cmd_empty (we),
        .cmd_pop (wp),
        .cmd (wo),
        .req_empty (qe),
        .req_pop (qp),
        .req_row (qo),
        .res_push (rpush),
        .res_full (rf),
        .res_data (ri),
        .frame_valid (frame_stored),
        .error (ddr_error),
        .app_addr (app_addr),
        .app_cmd (app_cmd),
        .app_en (app_en),
        .app_rdy (app_rdy),
        .app_wdf_data (app_wdf_data),
        .app_wdf_mask (app_wdf_mask),
        .app_wdf_wren (app_wdf_wren),
        .app_wdf_end (app_wdf_end),
        .app_wdf_rdy (app_wdf_rdy),
        .app_rd_data (app_rd_data),
        .app_rd_data_valid (app_rd_data_valid),
        .rb_valid(rb_valid), .rb_ready(rb_ready), .rb_byte_addr(rb_byte_addr),
        .rb_response_valid(rb_response_valid), .rb_response_ready(rb_response_ready),
        .rb_response_data(rb_response_data)
    );
    up_hdmi_scan # (
        .PROGRESSIVE(PROGRESSIVE),
        .WIDTH (WIDTH),
        .HEIGHT (HEIGHT),
        .PITCH (PITCH),
        .H_ACTIVE(H_ACTIVE), .H_SYNC(H_SYNC), .H_BACK(H_BACK), .H_TOTAL(H_TOTAL),
        .V_ACTIVE(V_ACTIVE), .V_SYNC(V_SYNC), .V_BACK(V_BACK), .V_TOTAL(V_TOTAL)
    ) u_scan (
        .clk (pixel_clk),
        .rst_n (pr),
        .frame_valid_async (PROGRESSIVE ? calib : frame_stored),
        .req_push (qpush),
        .req_full (qf),
        .req_row (qi),
        .res_empty (re),
        .res_pop (rp_pop),
        .res_data (ro),
        .hs (video_hs),
        .vs (video_vs),
        .de (video_de),
        .rgb (video_rgb),
        .underflow (display_underflow)
    );
endmodule

module hdmi_reset_sync (
    input wire clk, arst_n,
    output wire rst_n
);

    // Async assertion, destination-clock synchronous release. XPM ships
    // scoped reset CDC constraints; do not false-path entire clock domains.
    xpm_cdc_async_rst #(.DEST_SYNC_FF(4), .RST_ACTIVE_HIGH(0)) u_reset (
        .src_arst(arst_n), .dest_clk(clk), .dest_arst(rst_n)
    );
endmodule
