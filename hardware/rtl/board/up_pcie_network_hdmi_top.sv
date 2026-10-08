`timescale 1ns/1ps
// PCIe bank input feeding the verified progress_v4 display datapath.
module up_pcie_network_hdmi_top (
    input wire sys_clk_p, sys_clk_n, sys_rst_n,
    input wire [7:0] pcie_mgt_rxp, pcie_mgt_rxn,
    output wire [7:0] pcie_mgt_txp, pcie_mgt_txn,
    input wire [0:0] pcie_ref_clk_p, pcie_ref_clk_n,
    input wire pcie_rst_n,
    output wire [1 : 0] led,
    output wire rstn_out, iic_scl,
    inout wire iic_sda,
    output wire video_clk, video_hs, video_vs, video_de,
    output wire [23 : 0] video_rgb,
    output wire c0_ddr4_act_n,
    output wire [16 : 0] c0_ddr4_adr,
    output wire [1 : 0] c0_ddr4_ba,
    output wire [0 : 0] c0_ddr4_bg, c0_ddr4_cke, c0_ddr4_odt, c0_ddr4_cs_n, c0_ddr4_ck_t, c0_ddr4_ck_c,
    output wire c0_ddr4_reset_n,
    inout wire [7 : 0] c0_ddr4_dm_dbi_n,
    inout wire [63 : 0] c0_ddr4_dq,
    inout wire [7 : 0] c0_ddr4_dqs_c, c0_ddr4_dqs_t
);

    wire pcie_clk, pcie_reset_n, pcie_link;
    wire [31:0] gpio_command, gpio_status;
    wire load_cmd_valid, load_cmd_ready, load_bank, load_hold, load_wr_en;
    wire [3:0] load_tile;
    wire [7:0] load_wr_addr;
    wire [127:0] load_wr_data;
    wire tile_wr_en, network_start;
    wire [7:0] tile_wr_addr;
    wire [127:0] tile_wr_data;
    wire [31:0] tiles_started, tiles_completed;
    wire patch_busy;
    wire rb_start, rb_busy, rb_done, rb_error, rb_mem_wr_en;
    wire [7:0] rb_page;
    wire [31:0] rb_mem_byte_addr;
    wire [127:0] rb_mem_data;

    design_1_wrapper u_pcie (
        .pcie_mgt_rxp(pcie_mgt_rxp), .pcie_mgt_rxn(pcie_mgt_rxn),
        .pcie_mgt_txp(pcie_mgt_txp), .pcie_mgt_txn(pcie_mgt_txn),
        .pcie_ref_clk_p(pcie_ref_clk_p), .pcie_ref_clk_n(pcie_ref_clk_n),
        .pcie_rst_n(pcie_rst_n), .lnk_up_led(pcie_link),
        .pcie_clk_out(pcie_clk), .pcie_reset_n_out(pcie_reset_n),
        .gpio_command(gpio_command), .gpio_status(gpio_status),
        .load_cmd_valid(load_cmd_valid), .load_cmd_ready(load_cmd_ready),
        .load_bank(load_bank), .load_tile(load_tile), .load_hold(load_hold),
        .load_wr_en(load_wr_en), .load_wr_addr(load_wr_addr), .load_wr_data(load_wr_data),
        .diag_wr_en(rb_mem_wr_en), .diag_byte_addr(rb_mem_byte_addr),
        .diag_data(rb_mem_data)
    );
    wire ui_clk, ui_rst, calib, clk50, cfg_clk, net_locked;
    wire scene_start, net_clk_out, scene_ready, frame_stored, net_error, ddr_error, display_underflow;
    wire init_over, pixel_clk;
    wire [28 : 0] app_addr;
    wire [2 : 0] app_cmd;
    wire app_en, app_rdy, app_wdf_wren, app_wdf_end, app_wdf_rdy;
    wire [511 : 0] app_wdf_data, app_rd_data;
    wire [63 : 0] app_wdf_mask;
    wire app_rd_data_valid;
    ddr4_0 u_ddr4 (
        .sys_rst (! sys_rst_n),
        .c0_sys_clk_p (sys_clk_p),
        .c0_sys_clk_n (sys_clk_n),
        .c0_init_calib_complete (calib),
        .dbg_clk (),
        .dbg_bus (),
        .c0_ddr4_act_n (c0_ddr4_act_n),
        .c0_ddr4_adr (c0_ddr4_adr),
        .c0_ddr4_ba (c0_ddr4_ba),
        .c0_ddr4_bg (c0_ddr4_bg),
        .c0_ddr4_cke (c0_ddr4_cke),
        .c0_ddr4_odt (c0_ddr4_odt),
        .c0_ddr4_cs_n (c0_ddr4_cs_n),
        .c0_ddr4_ck_t (c0_ddr4_ck_t),
        .c0_ddr4_ck_c (c0_ddr4_ck_c),
        .c0_ddr4_reset_n (c0_ddr4_reset_n),
        .c0_ddr4_dm_dbi_n (c0_ddr4_dm_dbi_n),
        .c0_ddr4_dq (c0_ddr4_dq),
        .c0_ddr4_dqs_c (c0_ddr4_dqs_c),
        .c0_ddr4_dqs_t (c0_ddr4_dqs_t),
        .c0_ddr4_ui_clk (ui_clk),
        .c0_ddr4_ui_clk_sync_rst (ui_rst),
        .addn_ui_clkout1 (clk50),
        .addn_ui_clkout2 (pixel_clk),
        .addn_ui_clkout3 (cfg_clk),
        .c0_ddr4_app_addr (app_addr),
        .c0_ddr4_app_cmd (app_cmd),
        .c0_ddr4_app_en (app_en),
        .c0_ddr4_app_hi_pri (1'b0),
        .c0_ddr4_app_rdy (app_rdy),
        .c0_ddr4_app_wdf_data (app_wdf_data),
        .c0_ddr4_app_wdf_mask (app_wdf_mask),
        .c0_ddr4_app_wdf_wren (app_wdf_wren),
        .c0_ddr4_app_wdf_end (app_wdf_end),
        .c0_ddr4_app_wdf_rdy (app_wdf_rdy),
        .c0_ddr4_app_rd_data (app_rd_data),
        .c0_ddr4_app_rd_data_valid (app_rd_data_valid),
        .c0_ddr4_app_rd_data_end ()
    );
    hdmi_net_clock u_net_clock (
        .clk_in1 (clk50),
        .reset (! sys_rst_n || ui_rst),
        .clk_out1 (net_clk_out),
        .locked (net_locked)
    );
    wire common_reset_n = sys_rst_n && ! ui_rst && calib && net_locked && pcie_reset_n;
    wire net_reset_n, cfg_reset_n, host_reset_n;
    hdmi_reset_sync rn (net_clk_out, common_reset_n, net_reset_n);
    hdmi_reset_sync rc (cfg_clk, common_reset_n, cfg_reset_n);
    hdmi_reset_sync rp (pcie_clk, pcie_reset_n && sys_rst_n, host_reset_n);
    wire ov, sof, eof, ctx;
    wire [3 : 0] op, oc;
    wire [71 : 0] ol;
    wire inference_error;
    pcie_batch_ingress #(.TILE_COUNT(858)) u_ingress (
        .pcie_clk(pcie_clk), .pcie_rst_n(host_reset_n),
        .net_clk(net_clk_out), .net_rst_n(net_reset_n),
        .gpio_command(gpio_command), .gpio_status(gpio_status),
        .load_cmd_valid(load_cmd_valid), .load_cmd_ready(load_cmd_ready),
        .load_bank(load_bank), .load_tile(load_tile), .load_hold(load_hold),
        .load_wr_en(load_wr_en), .load_wr_addr(load_wr_addr), .load_wr_data(load_wr_data),
        .scene_ready(scene_ready), .start_enable(init_over), .patch_busy(patch_busy),
        .output_error(net_error | ddr_error), .frame_stored(frame_stored),
        .display_underflow(display_underflow),
        .out_valid(ov), .out_sof(sof), .out_eof(eof), .out_pixel(op),
        .scene_start(scene_start), .tile_wr_en(tile_wr_en), .tile_wr_addr(tile_wr_addr),
        .tile_wr_data(tile_wr_data), .start(network_start), .error(inference_error),
        .tiles_started(tiles_started), .tiles_completed(tiles_completed),
        .rb_start(rb_start), .rb_page(rb_page), .rb_busy(rb_busy),
        .rb_done(rb_done), .rb_error(rb_error)
    );
    mambahsi_full_network_stream_top u_network (
        .clk(net_clk_out), .rst_n(net_reset_n),
        .tile_wr_en(tile_wr_en), .tile_wr_addr(tile_wr_addr), .tile_wr_data(tile_wr_data),
        .start(network_start), .out_valid(ov), .out_pixel_addr(op), .out_logits(ol),
        .out_class(oc), .out_sof(sof), .out_eof(eof), .out_tile_context(ctx),
        .feature_done(), .done(), .patch_busy(patch_busy)
    );
    wire rb_valid, rb_ready, rb_response_valid, rb_response_ready;
    wire [31:0] rb_byte_addr;
    wire [511:0] rb_response_data;
    wire rb_ui_reset_n;
    hdmi_reset_sync rrb (ui_clk, common_reset_n, rb_ui_reset_n);
    ddr_readback_page u_readback (
        .pcie_clk(pcie_clk), .pcie_rst_n(host_reset_n),
        .ui_clk(ui_clk), .ui_rst_n(rb_ui_reset_n),
        .start(rb_start), .page(rb_page), .busy(rb_busy), .done(rb_done), .error(rb_error),
        .mem_wr_en(rb_mem_wr_en), .mem_byte_addr(rb_mem_byte_addr), .mem_data(rb_mem_data),
        .request_valid(rb_valid), .request_ready(rb_ready), .request_byte_addr(rb_byte_addr),
        .response_valid(rb_response_valid), .response_ready(rb_response_ready),
        .response_data(rb_response_data)
    );
    up_output_ddr_hdmi #(.PROGRESSIVE(1)) u_output (
        .net_clk (net_clk_out),
        .ui_clk (ui_clk),
        .pixel_clk (pixel_clk),
        .rst_n (common_reset_n),
        .calib (calib),
        .scene_start (scene_start),
        .out_valid (ov),
        .out_sof (sof),
        .out_eof (eof),
        .out_pixel_addr (op),
        .out_logits (ol),
        .scene_ready (scene_ready),
        .frame_stored (frame_stored),
        .net_error (net_error),
        .ddr_error (ddr_error),
        .display_underflow (display_underflow),
        .video_hs (video_hs),
        .video_vs (video_vs),
        .video_de (video_de),
        .video_rgb (video_rgb),
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
    hdmiout_ms72xx_ctl u_config (
        .clk (cfg_clk),
        .rst_n (cfg_reset_n),
        .rstn_out (rstn_out),
        .init_over (init_over),
        .iic_scl (iic_scl),
        .iic_sda (iic_sda)
    );
    // DDR video clock forwarded with rising sample edge half a cycle after RGB launch.
    ODDRE1 u_video_clk (
        .C (pixel_clk),
        .D1 (1'b0),
        .D2 (1'b1),
        .SR (! common_reset_n),
        .Q (video_clk)
    );
    // Board LEDs are ACTIVE HIGH: T22=physical LED2, T23=physical LED1.
    // Preserve existing polarity: LED2 OFF means calibrated;
    // LED1 blinks on error, otherwise OFF means frame stored.
    (* ASYNC_REG = "TRUE" *) reg [1 : 0] fs_sync, err_sync;
    reg [25 : 0] heartbeat;
    always @ (posedge net_clk_out) begin
        if (! net_reset_n) begin
            fs_sync <= 0;
            err_sync <= 0;
            heartbeat <= 0;
        end
        else begin
            fs_sync <= { fs_sync [0], frame_stored };
            err_sync <= { err_sync [0], inference_error | net_error | ddr_error | display_underflow };
            heartbeat <= heartbeat + 1'b1;
        end
    end
    assign led [0] = ! calib;
    assign led [1] = err_sync [1] ? heartbeat [25] : ! fs_sync [1];
endmodule
