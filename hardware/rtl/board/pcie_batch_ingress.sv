`timescale 1ns/1ps
`default_nettype none

// E2 ABI: commit complete banks, then GO. Ownership ends after the last
// source word enters the asynchronous FIFO. It does not wait for network EOF.
module pcie_batch_ingress #(
    parameter integer TILE_COUNT = 858,
    parameter integer TILE_II = 4096
) (
    input wire pcie_clk, pcie_rst_n, net_clk, net_rst_n,
    input wire [31:0] gpio_command,
    output reg [31:0] gpio_status,
    output reg load_cmd_valid,
    input wire load_cmd_ready,
    output reg load_bank,
    output reg [3:0] load_tile,
    output wire load_hold,
    input wire load_wr_en,
    input wire [7:0] load_wr_addr,
    input wire [127:0] load_wr_data,
    input wire scene_ready, start_enable, patch_busy,
    input wire output_error, frame_stored, display_underflow,
    input wire out_valid, out_sof, out_eof,
    input wire [3:0] out_pixel,
    output reg scene_start,
    output wire tile_wr_en,
    output wire [7:0] tile_wr_addr,
    output wire [127:0] tile_wr_data,
    output reg start, error,
    output reg [31:0] tiles_started, tiles_completed,
    output reg rb_start,
    output reg [7:0] rb_page,
    input wire rb_busy, rb_done, rb_error
);
    wire fifo_full, fifo_empty, fifo_pop;
    wire [135:0] fifo_data;
    hdmi_async_fifo #(.WIDTH(136), .DEPTH(512)) u_tiles (
        .rst(!pcie_rst_n || !net_rst_n), .wr_clk(pcie_clk), .rd_clk(net_clk),
        .din({load_wr_addr, load_wr_data}), .push(load_wr_en),
        .pop(fifo_pop), .dout(fifo_data), .full(fifo_full), .empty(fifo_empty)
    );
    wire enable_net, output_error_net, ready_pcie, frame_pcie;
    wire error_pcie, underflow_pcie, consumed_pcie;
    reg scene_begun, consumed_toggle;
    wire scene_available = net_rst_n && enable_net && !error && !output_error_net
                          && (scene_begun || scene_ready);
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_enable
        (.src_clk(1'b0), .src_in(start_enable), .dest_clk(net_clk), .dest_out(enable_net));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_output_error
        (.src_clk(1'b0), .src_in(output_error), .dest_clk(net_clk), .dest_out(output_error_net));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_ready
        (.src_clk(1'b0), .src_in(scene_available), .dest_clk(pcie_clk), .dest_out(ready_pcie));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_frame
        (.src_clk(1'b0), .src_in(frame_stored), .dest_clk(pcie_clk), .dest_out(frame_pcie));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_error
        (.src_clk(1'b0), .src_in(error), .dest_clk(pcie_clk), .dest_out(error_pcie));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_underflow
        (.src_clk(1'b0), .src_in(display_underflow), .dest_clk(pcie_clk), .dest_out(underflow_pcie));
    xpm_cdc_single #(.DEST_SYNC_FF(3), .SRC_INPUT_REG(0)) c_consumed
        (.src_clk(1'b0), .src_in(consumed_toggle), .dest_clk(pcie_clk), .dest_out(consumed_pcie));
    wire [31:0] completed_pcie;
    xpm_cdc_gray #(.WIDTH(32), .DEST_SYNC_FF(3)) c_completed
        (.src_clk(net_clk), .src_in_bin(tiles_completed), .dest_clk(pcie_clk),
         .dest_out_bin(completed_pcie));

    reg [1:0] owned;
    reg [4:0] bank_count [0:1];
    reg next_bank, go, previous_command, host_error, transfer_busy;
    reg waiting_consumed, consumed_previous;
    reg [31:0] committed, loaded;
    wire command_edge = gpio_command[0] && !previous_command;
    wire [2:0] opcode = gpio_command[3:1];
    wire [4:0] commit_count = gpio_command[9:5];
    assign load_hold = !ready_pcie || fifo_full;

    // One unconsumed 256-word transfer maximum. A command reserves the
    // entire transfer because the loader cannot stop in mid-record.
    always @(posedge pcie_clk) begin
        if (!pcie_rst_n) begin
            owned <= 0;
            bank_count[0] <= 0;
            bank_count[1] <= 0;
            next_bank <= 0;
            go <= 0;
            previous_command <= 0;
            host_error <= 0;
            transfer_busy <= 0;
            waiting_consumed <= 0;
            consumed_previous <= 0;
            committed <= 0;
            loaded <= 0;
            load_bank <= 0;
            load_tile <= 0;
            load_cmd_valid <= 0;
            rb_start <= 0;
            rb_page <= 0;
        end else begin
            previous_command <= gpio_command[0];
            load_cmd_valid <= 0;
            rb_start <= 0;
            consumed_previous <= consumed_pcie;
            if (consumed_previous != consumed_pcie) waiting_consumed <= 0;
            if (load_wr_en && fifo_full) host_error <= 1;
            if (command_edge) begin
                case (opcode)
                    1: begin
                        if (owned[gpio_command[4]] || commit_count == 0 ||
                            commit_count > 16 || committed + commit_count > TILE_COUNT)
                            host_error <= 1;
                        else begin
                            owned[gpio_command[4]] <= 1;
                            bank_count[gpio_command[4]] <= commit_count;
                            committed <= committed + commit_count;
                        end
                    end
                    2: if (go || !owned[0]) host_error <= 1; else go <= 1;
                    3: begin
                        if (!frame_pcie || rb_busy || gpio_command[15:8] >= 58)
                            host_error <= 1;
                        else begin
                            rb_page <= gpio_command[15:8];
                            rb_start <= 1;
                        end
                    end
                    default: host_error <= 1;
                endcase
            end
            if (go && ready_pcie && owned[next_bank] && !transfer_busy &&
                !waiting_consumed && !host_error && !error_pcie && !command_edge &&
                load_cmd_ready && loaded < TILE_COUNT) begin
                load_bank <= next_bank;
                load_cmd_valid <= 1;
                transfer_busy <= 1;
                waiting_consumed <= 1;
            end
            if (load_wr_en && load_wr_addr == 255) begin
                transfer_busy <= 0;
                loaded <= loaded + 1'b1;
                if ({1'b0, load_tile} + 1'b1 == bank_count[load_bank]) begin
                    owned[load_bank] <= 0;
                    next_bank <= !load_bank;
                    load_tile <= 0;
                end else load_tile <= load_tile + 1'b1;
            end
        end
    end

    localparam LOAD = 0, COMMIT = 1, WAIT_DEADLINE = 2, WAIT_PATCH = 3;
    reg [1:0] state;
    reg [7:0] expected_address;
    reg [3:0] expected_pixel;
    reg [31:0] cycle_count, last_start_cycle, min_ii, max_ii, late_starts;
    reg [31:0] cooldown;
    wire [31:0] interval_cycles = cycle_count - last_start_cycle;
    assign fifo_pop = state == LOAD && scene_available && !fifo_empty && !patch_busy
                      && tiles_started < TILE_COUNT;
    assign tile_wr_en = fifo_pop;
    assign tile_wr_addr = fifo_data[135:128];
    assign tile_wr_data = fifo_data[127:0];
    always @(posedge net_clk) begin
        if (!net_rst_n) begin
            state <= LOAD;
            scene_begun <= 0;
            scene_start <= 0;
            start <= 0;
            error <= 0;
            consumed_toggle <= 0;
            expected_address <= 0;
            expected_pixel <= 0;
            tiles_started <= 0;
            tiles_completed <= 0;
            cycle_count <= 0;
            last_start_cycle <= 0;
            min_ii <= 32'hffffffff;
            max_ii <= 0;
            late_starts <= 0;
            cooldown <= 0;
        end else begin
            cycle_count <= cycle_count + 1'b1;
            scene_start <= 0;
            start <= 0;
            if (cooldown != 0) cooldown <= cooldown - 1'b1;
            if (output_error_net) error <= 1;
            if (fifo_pop) begin
                if (tile_wr_addr != expected_address) error <= 1;
                expected_address <= expected_address + 1'b1;
                if (!scene_begun) begin
                    scene_start <= 1;
                    scene_begun <= 1;
                end
                if (expected_address == 255) begin
                    state <= COMMIT;
                    consumed_toggle <= !consumed_toggle;
                end
            end
            if (state == COMMIT) state <= WAIT_DEADLINE;
            if (state == WAIT_DEADLINE && cooldown == 0 && !patch_busy && !error) begin
                start <= 1;
                cooldown <= TILE_II - 1;
                tiles_started <= tiles_started + 1'b1;
                if (tiles_started != 0) begin
                    if (interval_cycles < min_ii) min_ii <= interval_cycles;
                    if (interval_cycles > max_ii) max_ii <= interval_cycles;
                    if (interval_cycles != TILE_II) late_starts <= late_starts + 1'b1;
                end
                last_start_cycle <= cycle_count;
                state <= WAIT_PATCH;
            end
            // Let Patch sample start before checking its busy output.
            if (state == WAIT_PATCH && !start && !patch_busy) state <= LOAD;
            if (out_valid) begin
                if (tiles_completed >= tiles_started || out_pixel != expected_pixel ||
                    out_sof != (expected_pixel == 0) || out_eof != (expected_pixel == 15))
                    error <= 1;
                expected_pixel <= expected_pixel + 1'b1;
                if (out_eof) tiles_completed <= tiles_completed + 1'b1;
            end
        end
    end

    // One coherent snapshot, transferred with a full CDC handshake. Do not
    // synchronize the changing binary min/max words independently.
    wire [127:0] statistics_pcie;
    reg statistics_send, statistics_sent, statistics_ready;
    wire statistics_ack, statistics_request;
    always @(posedge net_clk) begin
        if (!net_rst_n) begin
            statistics_send <= 0;
            statistics_sent <= 0;
        end else begin
            if (tiles_completed == TILE_COUNT && !statistics_sent) begin
                statistics_send <= 1;
                statistics_sent <= 1;
            end
            if (statistics_ack) statistics_send <= 0;
        end
    end
    always @(posedge pcie_clk) begin
        if (!pcie_rst_n) statistics_ready <= 0;
        else if (statistics_request) statistics_ready <= 1;
    end
    xpm_cdc_handshake #(.WIDTH(128), .DEST_EXT_HSK(0), .INIT_SYNC_FF(1),
                        .DEST_SYNC_FF(3), .SRC_SYNC_FF(3)) c_stats
        (.src_clk(net_clk), .src_in({tiles_started, late_starts, max_ii, min_ii}),
         .src_send(statistics_send), .src_rcv(statistics_ack),
         .dest_clk(pcie_clk), .dest_out(statistics_pcie),
         .dest_req(statistics_request), .dest_ack(1'b0));
    always @* begin
        case (gpio_command[31:28])
            1: gpio_status = statistics_ready ? statistics_pcie[31:0] : 0;
            2: gpio_status = statistics_ready ? statistics_pcie[63:32] : 0;
            3: gpio_status = statistics_ready ? statistics_pcie[95:64] : 0;
            4: gpio_status = statistics_ready ? statistics_pcie[127:96] : 0;
            5: gpio_status = {24'b0, rb_page};
            6: gpio_status = committed;
            7: gpio_status = {30'b0, statistics_ready, underflow_pcie};
            default: gpio_status = {completed_pcie[15:0], 8'hE2,
                rb_busy, rb_done, go, frame_pcie,
                (host_error || error_pcie || rb_error), owned, ready_pcie};
        endcase
    end
endmodule
`default_nettype wire
