`timescale 1ns/1ps
// Native MIG DDR4: 64-bit physical bus, BL8, 512-bit app data. Address unit=8 bytes.
// One write at a time; command and write-data ready are independent.

module up_ddr_label_store # (
    parameter PITCH = 384,
    parameter WIDTH = 340, HEIGHT = 610,
    parameter integer PROGRESSIVE = 0
) (
    input wire clk, rst_n, calib,
    input wire cmd_empty,
    output wire cmd_pop,
    input wire [176 : 0] cmd,
    input wire req_empty,
    output wire req_pop,
    input wire [9 : 0] req_row,
    output wire res_push,
    input wire res_full,
    output wire [524 : 0] res_data,
    output reg frame_valid,
    output reg error,
    output wire [28 : 0] app_addr,
    output wire [2 : 0] app_cmd,
    output wire app_en,
    input wire app_rdy,
    output wire [511 : 0] app_wdf_data,
    output wire [63 : 0] app_wdf_mask,
    output wire app_wdf_wren,
    output wire app_wdf_end,
    input wire app_wdf_rdy,
    input wire [511 : 0] app_rd_data,
    input wire app_rd_data_valid,
    input wire rb_valid,
    output wire rb_ready,
    input wire [31:0] rb_byte_addr,
    output reg rb_response_valid,
    input wire rb_response_ready,
    output reg [511:0] rb_response_data
);

    localparam IDLE = 0, WRITE = 1, READ_CMD = 2, READ_WAIT = 3,
               READ_FILL = 4, READ_PREP = 5, RB_CMD = 6, RB_WAIT = 7;
    localparam integer TILES_X = (WIDTH + 15) / 16;
    reg [2 : 0] state;
    reg [176 : 0] w;
    reg [28:0] rb_addr_q;
    // A whole 64-byte response has a reserved register before issuing.
    // HDMI requests win at IDLE; PCIe can delay a new line by at most one read.
    reg sent_cmd, sent_data;
    reg [9 : 0] row;
    reg [2 : 0] word_index;
    // Writes arrive tile-major, then row-major within each tile. Only a
    // COMPLETED tile becomes visible. No reset/clear of the DDR image needed.
    reg [15:0] committed_tiles, read_frontier;
    reg [9:0] write_tile_x, write_tile_y;
    reg [3:0] write_tile_row;
    reg [15:0] read_tile_base;
    reg last_grant_read;
    wire want_write = !frame_valid && !cmd_empty;
    wire want_read = (PROGRESSIVE || frame_valid) && !req_empty;
    assign rb_ready = calib && frame_valid && state == IDLE && !want_read
                      && !rb_response_valid;
    wire grant_read = want_read && (!want_write || !last_grant_read);
    wire [15:0] word_tile_base = read_tile_base + {word_index, 2'b00};
    wire [3:0] visible_lanes;
    reg [3:0] visible_lanes_q;
    reg read_visible_q;
    reg [28:0] read_app_addr_q;
    wire [511:0] visible_data;
    genvar lane_index;
    generate for (lane_index=0; lane_index<4; lane_index=lane_index+1) begin : G_VISIBLE
        assign visible_lanes[lane_index] = !PROGRESSIVE ||
            ((word_index*4 + lane_index < TILES_X) &&
             (word_tile_base + lane_index < read_frontier));
        assign visible_data[lane_index*128 +: 128] = visible_lanes_q[lane_index]
            ? app_rd_data[lane_index*128 +: 128] : {16{8'hff}};
    end endgenerate
    wire [31 : 0] byte_address = w [159 : 128];
    wire [1 : 0] lane = byte_address [5 : 4];
    assign cmd_pop = calib && state == IDLE && want_write && !grant_read;
    assign req_pop = calib && state == IDLE && grant_read;
    assign app_en = calib && ((state == WRITE && ! sent_cmd) ||
        (state == READ_CMD && !res_full && read_visible_q) || state == RB_CMD);
    assign app_cmd = (state == WRITE) ? 3'b000 : 3'b001;
    assign app_addr = (state == WRITE) ? { byte_address [31 : 6], 3'b000 }
                    : (state == RB_CMD ? rb_addr_q : read_app_addr_q);
    assign app_wdf_data = { { 384 { 1'b0 } }, w [127 : 0] } << (lane * 128);
    assign app_wdf_mask = ~ ({ 48'b0, w [175 : 160] } << (lane * 16));
    assign app_wdf_wren = calib && state == WRITE && ! sent_data;
    assign app_wdf_end = app_wdf_wren;
    assign res_push = !res_full && ((state == READ_WAIT && app_rd_data_valid) || state == READ_FILL);
    assign res_data = {row, word_index,
        (state == READ_FILL ? {64{8'hff}} : visible_data)};
    always @ (posedge clk) begin
        if (! rst_n) begin
            state <= IDLE;
            frame_valid <= 0;
            error <= 0;
            sent_cmd <= 0;
            sent_data <= 0;
            row <= 0;
            word_index <= 0;
            w <= 0;
            committed_tiles <= 0;
            read_frontier <= 0;
            write_tile_x <= 0;
            write_tile_y <= 0;
            write_tile_row <= 0;
            read_tile_base <= 0;
            last_grant_read <= 0;
            visible_lanes_q <= 0;
            read_visible_q <= 0;
            read_app_addr_q <= 0;
            rb_response_valid <= 0;
            rb_addr_q <= 0;
        end
        else begin
        if (rb_response_valid && rb_response_ready) rb_response_valid <= 0;
        case (state)
        IDLE : begin
            if (cmd_pop) begin
                w <= cmd;
                sent_cmd <= 0;
                sent_data <= 0;
                state <= WRITE;
                last_grant_read <= 0;
            end
            else if (req_pop) begin
                row <= req_row;
                word_index <= 0;
                state <= READ_PREP;
                read_frontier <= committed_tiles;
                read_tile_base <= (req_row >> 4) * TILES_X;
                last_grant_read <= 1;
            end
            else if (rb_valid && rb_ready) begin
                rb_addr_q <= rb_byte_addr[31:3];
                state <= RB_CMD;
                if (rb_byte_addr[5:0] != 0) error <= 1;
            end
        end
        WRITE : begin
            if (app_rdy && ! sent_cmd) sent_cmd <= 1;
            if (app_wdf_rdy && ! sent_data) sent_data <= 1;
            if ((sent_cmd || app_rdy) && (sent_data || app_wdf_rdy)) begin
                if (PROGRESSIVE) begin
                    if (write_tile_row == 15 ||
                        (write_tile_y*16 + write_tile_row == HEIGHT-1)) begin
                        committed_tiles <= committed_tiles + 1'b1;
                        write_tile_row <= 0;
                        if (write_tile_x == TILES_X-1) begin
                            write_tile_x <= 0;
                            write_tile_y <= write_tile_y + 1'b1;
                        end else write_tile_x <= write_tile_x + 1'b1;
                    end else write_tile_row <= write_tile_row + 1'b1;
                end
                if (w [176]) frame_valid <= 1;
                state <= IDLE;
            end
        end
        // The new request's row/frontier/index have settled before this
        // edge. Hold the snapshot through stalls and the read response.
        // No tile-range carry/compare path reaches MIG app_en in READ_CMD.
        READ_PREP : begin
            visible_lanes_q <= visible_lanes;
            read_visible_q <= |visible_lanes;
            read_app_addr_q <= (row * PITCH + word_index * 64) >> 3;
            state <= READ_CMD;
        end
        READ_CMD : begin
            if (!read_visible_q) state <= READ_FILL;
            else if (app_en && app_rdy) state <= READ_WAIT;
        end
        READ_FILL : if (!res_full) begin
            if (word_index == PITCH / 64 - 1) state <= IDLE;
            else begin
                word_index <= word_index + 1'b1;
                state <= READ_PREP;
            end
        end
        READ_WAIT : if (app_rd_data_valid) begin
            if (res_full) error <= 1;
            if (word_index == PITCH / 64 - 1) state <= IDLE;
            else begin
                word_index <= word_index + 1'b1;
                state <= READ_PREP;
            end
        end
        RB_CMD: if (app_en && app_rdy) state <= RB_WAIT;
        RB_WAIT: if (app_rd_data_valid) begin
            rb_response_data <= app_rd_data;
            rb_response_valid <= 1;
            state <= IDLE;
        end
        default : state <= IDLE;
    endcase
    end
end
endmodule
