`timescale 1ns/1ps
`default_nettype none

// Real loader/network/postprocessor and XPM CDC; modeled bank ports and MIG.
// Host GPIO commands are emulated; PCIe physical/link/AXI behavior is not modeled.
// Shortened VIDEO BLANKING ONLY keeps the joint test below 1ms. Network
// dimensions, arithmetic and output CDC are unchanged; actual start II=4096.
module tb_d1_network_ddr_hdmi #(
    parameter integer REAL_FIVE = 0
);
    reg net_clk = 0, ui_clk = 0, pixel_clk = 0;
    always #5 net_clk = ~net_clk;
    always #1.666 ui_clk = ~ui_clk;
    always #6.667 pixel_clk = ~pixel_clk;
    reg rst_n = 0, calib = 0, enable = 0;
    wire scene_start, ready, stored, inference_error, net_error, ddr_error, underflow;
    wire valid, sof, eof;
    wire [3:0] pixel;
    wire [71:0] logits;
    wire [31:0] launched, completed;
    wire patch_busy;
    wire rb_start, rb_busy, rb_done, rb_error, rb_mem_wr;
    wire [7:0] rb_page;
    wire [31:0] rb_mem_addr, rb_addr;
    wire [127:0] rb_mem_data;
    wire rb_valid, rb_ready, rb_response_valid, rb_response_ready;
    wire [511:0] rb_response_data;
    reg [127:0] page_memory [0:255];
    reg [511:0] memory [0:63];
    reg checked_readback = 0;
    wire hs, vs, de;
    wire [23:0] rgb;
    wire [28:0] app_addr;
    wire [2:0] app_cmd;
    wire app_en, wren, wend;
    reg app_rdy = 0, wdf_rdy = 0;
    wire [511:0] wdata;
    wire [63:0] wmask;
    reg [511:0] rdata = 0;
    reg rvalid = 0;


    reg pcie_clk=0;
    always #2 pcie_clk=~pcie_clk;
    reg [31:0] command=0;
    wire [31:0] status;
    wire cmd_valid, cmd_ready, bank, hold_load;
    wire [3:0] slot;
    wire a_en, b_en, load_wr, patch_wr, network_start;
    wire [31:0] bank_addr;
    wire [7:0] load_addr, patch_addr;
    wire [127:0] load_data, patch_data;
    reg [255:0] a_data=0, b_data=0;
    reg [255:0] bank_a [0:2047], bank_b [0:2047];
    reg [127:0] input_tile [0:1279];
    integer loaded=0;
    bank_to_patch_loader loader (
        .clk(pcie_clk), .rst_n(rst_n), .cmd_valid(cmd_valid), .cmd_ready(cmd_ready),
        .cmd_bank(bank), .cmd_tile(slot), .patch_busy(hold_load),
        .busy(), .done(), .bank_a_en(a_en), .bank_b_en(b_en),
        .bank_byte_addr(bank_addr), .bank_word_addr(),
        .bank_a_data(a_data), .bank_b_data(b_data),
        .tile_wr_en(load_wr), .tile_wr_addr(load_addr), .tile_wr_byte_addr(),
        .tile_wr_data(load_data), .words_written()
    );
    always @(posedge pcie_clk) begin
        if(a_en) a_data <= bank_a[bank_addr >> 5];
        if(b_en) b_data <= bank_b[bank_addr >> 5];
    end
    pcie_batch_ingress #(.TILE_COUNT(5)) ingress (
        .pcie_clk(pcie_clk), .pcie_rst_n(rst_n), .net_clk(net_clk), .net_rst_n(rst_n),
        .gpio_command(command), .gpio_status(status), .load_cmd_valid(cmd_valid),
        .load_cmd_ready(cmd_ready), .load_bank(bank), .load_tile(slot), .load_hold(hold_load),
        .load_wr_en(load_wr), .load_wr_addr(load_addr), .load_wr_data(load_data),
        .scene_ready(ready), .start_enable(enable), .patch_busy(patch_busy),
        .output_error(net_error | ddr_error),
        .frame_stored(stored), .display_underflow(underflow),
        .out_valid(valid), .out_sof(sof), .out_eof(eof), .out_pixel(pixel),
        .scene_start(scene_start), .tile_wr_en(patch_wr), .tile_wr_addr(patch_addr),
        .tile_wr_data(patch_data), .start(network_start), .error(inference_error),
        .tiles_started(launched), .tiles_completed(completed),
        .rb_start(rb_start), .rb_page(rb_page), .rb_busy(rb_busy),
        .rb_done(rb_done), .rb_error(rb_error)
    );
    mambahsi_full_network_stream_top inference (
        .clk(net_clk), .rst_n(rst_n), .tile_wr_en(patch_wr),
        .tile_wr_addr(patch_addr), .tile_wr_data(patch_data), .start(network_start),
        .out_valid(valid), .out_pixel_addr(pixel), .out_logits(logits),
        .out_class(), .out_sof(sof), .out_eof(eof), .out_tile_context(),
        .feature_done(), .done(), .patch_busy(patch_busy)
    );
    ddr_readback_page readback (
        .pcie_clk(pcie_clk), .pcie_rst_n(rst_n), .ui_clk(ui_clk), .ui_rst_n(rst_n),
        .start(rb_start), .page(rb_page), .busy(rb_busy), .done(rb_done), .error(rb_error),
        .mem_wr_en(rb_mem_wr), .mem_byte_addr(rb_mem_addr), .mem_data(rb_mem_data),
        .request_valid(rb_valid), .request_ready(rb_ready), .request_byte_addr(rb_addr),
        .response_valid(rb_response_valid), .response_ready(rb_response_ready),
        .response_data(rb_response_data)
    );
    always @(posedge pcie_clk) if (rb_mem_wr) page_memory[rb_mem_addr >> 4] <= rb_mem_data;
    task host_command(input [31:0] value);
        @(negedge pcie_clk); command = 0;
        repeat (4) @(negedge pcie_clk);
        command = value | 1;
        repeat (4) @(negedge pcie_clk);
        command = 0;
        repeat (4) @(negedge pcie_clk);
    endtask
    initial begin : host
        if (REAL_FIVE)
            $readmemh("real5_input_128b.mem", input_tile);
        else
            $readmemh("network_input_tile.mem", input_tile, 0, 255);
        wait(enable);
        wait(status[0]);
        for(integer t=0;t<5;t=t+1) begin
            for(integer w=0;w<128;w=w+1) begin
                @(negedge pcie_clk);
                if(t>=3) bank_b[(t-3)*128+w]={input_tile[(REAL_FIVE?t*256:0)+2*w+1],input_tile[(REAL_FIVE?t*256:0)+2*w]};
                else bank_a[t*128+w]={input_tile[(REAL_FIVE?t*256:0)+2*w+1],input_tile[(REAL_FIVE?t*256:0)+2*w]};
            end
        end
        host_command(2 | (3 << 5));
        host_command(2 | (1 << 4) | (2 << 5));
        host_command(4);
        wait(status[4]);
        host_command(6);
        wait(rb_done && !rb_busy);
        repeat(4) @(negedge pcie_clk);
        for(integer word=0;word<256;word=word+1)
            if(page_memory[word] !== memory[word/4][(word%4)*128 +: 128])
                $fatal(1,"DDR -> PCIe page mismatch word=%0d",word);
        checked_readback=1;
        $display("READBACK PASS: 4096 actual DDR bytes in C2H page buffer, HDMI active.");
    end
    always @(posedge net_clk) if(rst_n && patch_wr) begin
        if(loaded>=1280 || patch_addr !== (loaded%256) || patch_data !== input_tile[REAL_FIVE?loaded:loaded%256])
            $fatal(1,"PCIe clock to Patch byte/address mismatch beat=%0d",loaded);
        loaded=loaded+1;
    end
    up_output_ddr_hdmi #(
        .WIDTH(80), .HEIGHT(16), .PITCH(128),
        .H_ACTIVE(128), .H_SYNC(4), .H_BACK(8), .H_TOTAL(192),
        .V_ACTIVE(32), .V_SYNC(2), .V_BACK(2), .V_TOTAL(48)
    ) output_path (
        .net_clk(net_clk), .ui_clk(ui_clk), .pixel_clk(pixel_clk),
        .rst_n(rst_n), .calib(calib), .scene_start(scene_start),
        .out_valid(valid), .out_sof(sof), .out_eof(eof),
        .out_pixel_addr(pixel), .out_logits(logits), .scene_ready(ready),
        .frame_stored(stored), .net_error(net_error), .ddr_error(ddr_error),
        .display_underflow(underflow), .video_hs(hs), .video_vs(vs),
        .video_de(de), .video_rgb(rgb), .app_addr(app_addr), .app_cmd(app_cmd),
        .app_en(app_en), .app_rdy(app_rdy), .app_wdf_data(wdata),
        .app_wdf_mask(wmask), .app_wdf_wren(wren), .app_wdf_end(wend),
        .app_wdf_rdy(wdf_rdy), .app_rd_data(rdata), .app_rd_data_valid(rvalid),
        .rb_valid(rb_valid), .rb_ready(rb_ready), .rb_byte_addr(rb_addr),
        .rb_response_valid(rb_response_valid), .rb_response_ready(rb_response_ready),
        .rb_response_data(rb_response_data)
    );

    reg [7:0] reference_logits [0:719];
    reg [7:0] reference_labels [0:1279];
    integer logit_errors = 0;
    integer tile_logit_errors [0:4];
    integer logit_dump;
    integer reference_tile;

    function automatic integer logit_index(input integer tile, channel, point);
        logit_index = (REAL_FIVE ? tile*144 : 0) + channel*16 + point;
    endfunction

    function automatic integer label_index(input integer xx, yy);
        label_index = (REAL_FIVE ? (xx/16)*256 : 0) + yy*16 + xx%16;
    endfunction
    reg command_pending = 0, data_pending = 0;
    reg [28:0] write_address;
    reg [511:0] write_data;
    reg [63:0] write_mask;
    integer ui_ticks = 0, read_delay = 0, read_index = 0, writes = 0;
    integer cycles = 0, beats = 0, last_start = -1, last_sof = -1;
    integer starts = 0, sofs = 0, eofs = 0, classes_checked = 0;
    integer n, x, y, c, ax, ay, fx, fy, score, best, winner, b;
    integer displays = 0, old_h, old_v, sx, sy;
    reg checked_ddr = 0;

    function automatic [23:0] palette(input [7:0] id);
        case (id)
            0: palette=24'hff0000; 1: palette=24'h00ff00; 2: palette=24'h0000ff;
            3: palette=24'hffff00; 4: palette=24'hff00ff; 5: palette=24'h00ffff;
            6: palette=24'hc86400; 7: palette=24'h00c864; 8: palette=24'h6400c8;
            default: palette=0;
        endcase
    endfunction

    // Independent mathematical reference: signed logits, rational bilinear
    // weights /25, no INT8 intermediate truncation; ties use lowest class ID.
    initial begin
        logit_dump = $fopen("actual_head_logits.csv", "w");
        if (!logit_dump) $fatal(1, "Cannot open actual_head_logits.csv");
        $fdisplay(logit_dump, "tile,pixel,class,actual_hex,expected_hex,match");
        for (n=0; n<5; n=n+1) tile_logit_errors[n]=0;
        if (REAL_FIVE)
            $readmemh("real5_expected_logits_chw.mem", reference_logits);
        else
            $readmemh("expected_head_logits.mem", reference_logits, 0, 143);
        for (n=0; n<(REAL_FIVE?720:144); n=n+1)
            if ((^reference_logits[n]) === 1'bx)
                $fatal(1, "Missing/unknown golden logit byte %0d", n);
        for (n=0; n<64; n=n+1) memory[n] = 0;
        for (reference_tile=0; reference_tile<(REAL_FIVE?5:1); reference_tile=reference_tile+1) begin
        for (y=0; y<16; y=y+1) begin
            for (x=0; x<16; x=x+1) begin
                ax=(x==15)?2:x/5; ay=(y==15)?2:y/5;
                fx=x-ax*5; fy=y-ay*5; best=-100000; winner=0;
                for (c=0; c<9; c=c+1) begin
                    score=(5-fx)*(5-fy)*$signed(reference_logits[logit_index(reference_tile,c,ay*4+ax)])
                        +fx*(5-fy)*$signed(reference_logits[logit_index(reference_tile,c,ay*4+ax+1)])
                        +(5-fx)*fy*$signed(reference_logits[logit_index(reference_tile,c,(ay+1)*4+ax)])
                        +fx*fy*$signed(reference_logits[logit_index(reference_tile,c,(ay+1)*4+ax+1)]);
                    if (score>best) begin best=score; winner=c; end
                end
                reference_labels[reference_tile*256+y*16+x]=winner;
            end
        end
        end
        repeat (20) @(negedge net_clk);
        rst_n=1; calib=1;
        repeat (10) @(negedge net_clk);
        enable=1;
    end

    always @(posedge net_clk) begin
        cycles=cycles+1;
        if (rst_n) begin
            if (rb_error || status[3] || inference_error || net_error || ddr_error || underflow)
                $fatal(1,"CHAIN error inference/post/DDR/scan=%b%b%b%b",
                       inference_error,net_error,ddr_error,underflow);
            if (network_start) begin
                if (last_start>=0 && cycles-last_start!=4096) $fatal(1,"Input II mismatch: %0d",cycles-last_start);
                last_start=cycles; starts=starts+1;
            end
            if (valid) begin
                if (beats>=80 || pixel!==(beats%16) || sof!==(pixel==0) || eof!==(pixel==15))
                    $fatal(1,"Network output ordering/count mismatch beat=%0d",beats);
                for (b=0; b<9; b=b+1) begin
                    $fdisplay(logit_dump, "%0d,%0d,%0d,%02h,%02h,%0d",
                        beats/16, pixel, b, logits[b*8+:8],
                        reference_logits[logit_index(beats/16,b,pixel)],
                        logits[b*8+:8] === reference_logits[logit_index(beats/16,b,pixel)]);
                    if (logits[b*8+:8] !== reference_logits[logit_index(beats/16,b,pixel)]) begin
                        if (logit_errors<40)
                            $display("LOGIT FAIL tile=%0d pixel=%0d class=%0d got=%0d expected=%0d",
                                beats/16,pixel,b,$signed(logits[b*8+:8]),
                                $signed(reference_logits[logit_index(beats/16,b,pixel)]));
                        logit_errors=logit_errors+1;
                        tile_logit_errors[beats/16]=tile_logit_errors[beats/16]+1;
                    end
                    classes_checked=classes_checked+1;
                end
                if (pixel==15) $fflush(logit_dump);
                beats=beats+1;
                if (beats==80) begin
                    $fclose(logit_dump);
                    for (integer t=0; t<5; t=t+1)
                        $display("TILE LOGIT SUMMARY tile=%0d bytes=144 errors=%0d", t, tile_logit_errors[t]);
                    if (logit_errors!=0)
                        $fatal(1,"720-byte logit comparison FAILED: %0d errors; see actual_head_logits.csv",logit_errors);
                    $display("LOGITS PASS: all 720 bytes / 5760 bits exact; REAL_FIVE=%0d", REAL_FIVE);
                end
            end
            if (sof) begin
                if (last_sof>=0 && cycles-last_sof!=4096) $fatal(1,"Network output II mismatch");
                last_sof=cycles; sofs=sofs+1;
                $display("REAL INFERENCE tile=%0d SOF cycle=%0d",sofs-1,cycles);
            end
            if (eof) eofs=eofs+1;
            if (cycles%5000==0)
                $display("CHAIN cycle=%0d starts/tiles/logits=%0d/%0d/%0d writes=%0d stored=%b HDMI_pixels=%0d",
                         cycles,starts,completed,beats,writes,stored,displays);
        end
    end

    // Native MIG model: command and write data accepted independently.
    always @(negedge ui_clk) begin
        ui_ticks=ui_ticks+1;
        app_rdy=rst_n && (ui_ticks%5!=0);
        wdf_rdy=rst_n && (ui_ticks%3!=0);
    end
    always @(posedge ui_clk) begin
        rvalid<=0;
        if (app_en && app_rdy) begin
            if (app_addr[2:0]!=0 || (app_addr>>3)>=64) $fatal(1,"DDR address out of range");
            case (app_cmd)
                0: begin
                    if (command_pending) $fatal(1,"Duplicate write command");
                    command_pending<=1; write_address<=app_addr;
                end
                1: begin
                    if (read_delay!=0) $fatal(1,"Multiple outstanding reads");
                    read_delay<=5; read_index<=app_addr>>3;
                end
                default: $fatal(1,"Unexpected MIG command");
            endcase
        end
        if (wren && wdf_rdy) begin
            if (data_pending || !wend) $fatal(1,"Invalid write data beat");
            data_pending<=1; write_data<=wdata; write_mask<=wmask;
        end
        if (command_pending && data_pending) begin
            for (integer j=0; j<64; j=j+1)
                if (!write_mask[j]) memory[write_address>>3][j*8+:8]<=write_data[j*8+:8];
            writes<=writes+1; command_pending<=0; data_pending<=0;
        end
        if (read_delay>0) begin
            read_delay<=read_delay-1;
            if (read_delay==1) begin rdata<=memory[read_index]; rvalid<=1; end
        end
    end

    initial begin
        wait (stored);
        repeat (20) @(negedge ui_clk);
        if (writes!=80 || starts!=5 || sofs!=5 || eofs!=5 || beats!=80 || completed!=5 || loaded!=1280 || classes_checked!=720)
            $fatal(1,"Final inference/write counts mismatch");
        for (integer yy=0; yy<16; yy=yy+1)
            for (integer xx=0; xx<80; xx=xx+1)
                if (memory[yy*2+xx/64][(xx%64)*8+:8] !== reference_labels[label_index(xx,yy)])
                    $fatal(1,"DDR label mismatch x/y=%0d/%0d",xx,yy);
        checked_ddr=1;
        $display("DDR PASS: 80 masked writes; 1280 labels match bilinear/argmax reference.");
    end

    always @(posedge pixel_clk) begin
        old_h=output_path.u_scan.h; old_v=output_path.u_scan.v;
        #1;
        if (output_path.u_scan.show_frame && old_h>=36 && old_h<116 && old_v>=12 && old_v<28) begin
            sx=old_h-36; sy=old_v-12;
            if (!checked_ddr || !de || rgb !== palette(reference_labels[label_index(sx,sy)]))
                $fatal(1,"HDMI pixel mismatch x/y=%0d/%0d rgb=%h",sx,sy,rgb);
            displays=displays+1;
            if (displays==2560) begin
                if (!checked_readback) $fatal(1,"C2H page was not checked");
                $display("PASS: D1 repeated-first-tile bank loader -> network -> DDR -> HDMI -> C2H.");
                if (REAL_FIVE) $display("PASS: first FIVE DIFFERENT scene tiles, independent software logits, bit-exact, II=4096.");
                $display("5 input tiles; 720 exact logit bytes; actual II=4096; 20480 Patch bytes; 1280 DDR labels; 2560 HDMI pixels; 4096 C2H page bytes.");
                $display("Display uses shortened simulation blanking; MIG is a native-interface behavioral model.");
                $finish;
            end
        end
    end
    initial begin #2000000; $fatal(1,"Joint chain timeout: beats=%0d stored=%b HDMI=%0d",beats,stored,displays); end
endmodule

`default_nettype wire
