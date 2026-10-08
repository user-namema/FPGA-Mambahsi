`timescale 1ns/1ps
`default_nettype none

// Source: BMG in BRAM_Controller mode, one synchronous read cycle.
// Address is a BYTE address: tile*4096 + word*32 (not a 256-bit word index).
// Destination: patch_embdding tile_wr_* contract, channel c at [8*c +: 8].
module bank_to_patch_loader (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         cmd_valid,
    output wire         cmd_ready,
    input  wire         cmd_bank,
    input  wire [3:0]   cmd_tile,
    input  wire         patch_busy,
    output wire         busy,
    output reg          done,
    output wire         bank_a_en,
    output wire         bank_b_en,
    output wire [31:0]  bank_byte_addr,
    output wire [10:0]  bank_word_addr,
    input  wire [255:0] bank_a_data,
    input  wire [255:0] bank_b_data,
    output wire         tile_wr_en,
    output wire [7:0]   tile_wr_addr,
    output wire [31:0]  tile_wr_byte_addr,
    output wire [127:0] tile_wr_data,
    output reg  [8:0]   words_written
);
    localparam IDLE=0, WAIT_PATCH=1, READ_WORD=2, CAPTURE=3,
               WRITE_LOW=4, WRITE_HIGH=5, FINISH=6;
    reg [2:0] state;
    reg bank_q;
    reg [3:0] tile_q;
    reg [6:0] word_q;
    reg [255:0] data_q;

    assign cmd_ready = state == IDLE;
    assign busy = state != IDLE;
    assign bank_a_en = state == READ_WORD && !bank_q;
    assign bank_b_en = state == READ_WORD && bank_q;
    assign bank_byte_addr = {16'd0, tile_q, word_q, 5'd0};
    // BMG is configured as 2048 x 256-bit. Its native address is a word
    // index, while the host/XDMA address above is a byte address.
    assign bank_word_addr = {tile_q, word_q};
    assign tile_wr_en = state == WRITE_LOW || state == WRITE_HIGH;
    assign tile_wr_addr = {word_q, (state == WRITE_HIGH)};
    assign tile_wr_byte_addr = {20'd0, tile_wr_addr, 4'd0};
    assign tile_wr_data = state == WRITE_HIGH ? data_q[255:128] : data_q[127:0];

    // Once started, the destination RAM is exclusively owned until done.
    // patch_busy is admission control, not per-beat backpressure.
    always @(posedge clk) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            words_written <= 0;
            bank_q <= 0;
            tile_q <= 0;
            word_q <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                IDLE: if (cmd_valid) begin
                    bank_q <= cmd_bank;
                    tile_q <= cmd_tile;
                    word_q <= 0;
                    words_written <= 0;
                    state <= WAIT_PATCH;
                end
                WAIT_PATCH: if (!patch_busy) state <= READ_WORD;
                READ_WORD: state <= CAPTURE;
                CAPTURE: begin
                    data_q <= bank_q ? bank_b_data : bank_a_data;
                    state <= WRITE_LOW;
                end
                WRITE_LOW: begin
                    words_written <= words_written + 1'b1;
                    state <= WRITE_HIGH;
                end
                WRITE_HIGH: begin
                    words_written <= words_written + 1'b1;
                    if (word_q == 127) state <= FINISH;
                    else begin
                        word_q <= word_q + 1'b1;
                        state <= READ_WORD;
                    end
                end
                FINISH: begin
                    done <= 1'b1;
                    state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
