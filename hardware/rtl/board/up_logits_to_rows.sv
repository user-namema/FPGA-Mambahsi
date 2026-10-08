`timescale 1ns/1ps

// Exact 4 -> 16 rational interpolation, then lowest-ID argmax.
// One class per cycle; four arithmetic registers, drain only at row boundaries.
module up_logits_to_rows #(
    parameter integer WIDTH = 340,
    parameter integer HEIGHT = 610,
    parameter integer CLASSES = 9,
    parameter integer PITCH = ((WIDTH + 63) / 64) * 64
) (
    input wire clk, rst_n, scene_start,
    input wire in_valid, in_sof, in_eof,
    input wire [3:0] in_pixel,
    input wire [CLASSES*8-1:0] in_logits,
    output wire row_valid,
    input wire row_ready,
    output wire [31:0] row_addr,
    output wire [127:0] row_data,
    output wire [15:0] row_keep,
    output wire row_last,
    output reg capture_done, error
);
    localparam integer TX = (WIDTH + 15) / 16, TY = (HEIGHT + 15) / 16;
    localparam [1:0] IDLE = 0, ISSUE = 1, DRAIN = 2, EMIT = 3;
    reg [CLASSES*8-1:0] mem [0:31];
    reg [15:0] produced, consumed, tile_x, tile_y;
    reg [3:0] expected_pixel, x, y;
    reg [7:0] c, best_id;
    reg active;
    reg [1:0] state;
    reg signed [12:0] best;
    reg [127:0] packed_row;

    // {base[1:0], fraction[2:0]}; coordinate 15 uses base=2, fraction=5.
    function automatic [4:0] axis(input [3:0] position);
        begin
            case (position)
                0: axis = {2'd0,3'd0};  1: axis = {2'd0,3'd1};
                2: axis = {2'd0,3'd2};  3: axis = {2'd0,3'd3};
                4: axis = {2'd0,3'd4};  5: axis = {2'd1,3'd0};
                6: axis = {2'd1,3'd1};  7: axis = {2'd1,3'd2};
                8: axis = {2'd1,3'd3};  9: axis = {2'd1,3'd4};
               10: axis = {2'd2,3'd0}; 11: axis = {2'd2,3'd1};
               12: axis = {2'd2,3'd2}; 13: axis = {2'd2,3'd3};
               14: axis = {2'd2,3'd4};
               default: axis = {2'd2,3'd5};
            endcase
        end
    endfunction

    wire [4:0] ax = axis(x), ay = axis(y);
    wire [4:0] address = {consumed[0], ay[4:3], ax[4:3]};
    wire [31:0] global_y = tile_y * 16 + y;
    wire [4:0] valid_width = (tile_x == TX-1) ? WIDTH-tile_x*16 : 16;
    wire issue = state == ISSUE;
    reg [3:0] valid_q;
    reg [3:0] pixel_q [0:3];
    reg [7:0] class_q [0:3];
    reg signed [7:0] q00, q01, q10, q11;
    reg signed [3:0] wa0, wa1, wb0, wb1, wb0_q, wb1_q;
    reg signed [10:0] horizontal0, horizontal1; // [-640, 635]
    reg signed [12:0] vertical0, vertical1;
    reg signed [12:0] score_q; // total weight 25: [-3200, 3175]
    integer k;

    // Free-running payload; reset/valid handled separately.
    always @(posedge clk) begin
        pixel_q[0] <= x;
        class_q[0] <= c;
        for (k = 1; k < 4; k = k + 1) begin
            pixel_q[k] <= pixel_q[k-1];
            class_q[k] <= class_q[k-1];
        end
        // S0: terminate the asynchronous tile reads here.
        q00 <= $signed(mem[address][c*8 +: 8]);
        q01 <= $signed(mem[address+5'd1][c*8 +: 8]);
        q10 <= $signed(mem[address+5'd4][c*8 +: 8]);
        q11 <= $signed(mem[address+5'd5][c*8 +: 8]);
        wa0 <= $signed({1'b0, 3'd5-ax[2:0]});
        wa1 <= $signed({1'b0, ax[2:0]});
        wb0 <= $signed({1'b0, 3'd5-ay[2:0]});
        wb1 <= $signed({1'b0, ay[2:0]});
        // S1: horizontal interpolation, explicitly signed narrow arithmetic.
        horizontal0 <= wa0*q00 + wa1*q01;
        horizontal1 <= wa0*q10 + wa1*q11;
        wb0_q <= wb0;
        wb1_q <= wb1;
        // S2: vertical products. S3: final sum, before argmax.
        vertical0 <= horizontal0*wb0_q;
        vertical1 <= horizontal1*wb1_q;
        score_q <= vertical0 + vertical1;
    end

    wire take_score = (class_q[3] == 0) || (score_q > best);
    wire [7:0] winner = take_score ? class_q[3] : best_id;
    assign row_valid = state == EMIT;
    assign row_addr = global_y*PITCH + tile_x*16;
    assign row_data = packed_row;
    assign row_keep = (17'h10000-(17'h10000 >> valid_width)) >> (16-valid_width);
    assign row_last = (tile_x == TX-1) && (global_y == HEIGHT-1);

    always @(posedge clk) begin
        if (!rst_n) begin
            produced <= 0;
            consumed <= 0;
            tile_x <= 0;
            tile_y <= 0;
            expected_pixel <= 0;
            x <= 0;
            y <= 0;
            c <= 0;
            best <= 0;
            best_id <= 0;
            packed_row <= 0;
            active <= 0;
            state <= IDLE;
            valid_q <= 0;
            capture_done <= 0;
            error <= 0;
        end else begin
            valid_q <= {valid_q[2:0], issue};
            if (scene_start) begin
                if (active || capture_done) error <= 1;
                else active <= 1;
            end
            if (in_valid) begin
                if (!active || produced >= TX*TY || produced-consumed >= 2 ||
                    in_pixel != expected_pixel || in_sof != (in_pixel == 0) ||
                    in_eof != (in_pixel == 15)) begin
                    error <= 1;
                end else begin
                    mem[{produced[0], in_pixel}] <= in_logits;
                    expected_pixel <= expected_pixel + 1'b1;
                    if (in_eof) produced <= produced + 1'b1;
                end
            end
            if (valid_q[3]) begin
                if (take_score) begin
                    best <= score_q;
                    best_id <= class_q[3];
                end
                if (class_q[3] == CLASSES-1)
                    packed_row[pixel_q[3]*8 +: 8] <= winner;
            end
            case (state)
                IDLE: if (active && produced != consumed) begin
                    x <= 0;
                    y <= 0;
                    c <= 0;
                    state <= ISSUE;
                end
                ISSUE: begin
                    if (c == CLASSES-1) begin
                        c <= 0;
                        if (x == 15) state <= DRAIN;
                        else x <= x + 1'b1;
                    end else c <= c + 1'b1;
                end
                DRAIN: if (valid_q[3] && pixel_q[3] == 15 && class_q[3] == CLASSES-1)
                    state <= EMIT;
                EMIT: if (row_ready) begin
                    if (y == 15 || global_y == HEIGHT-1) begin
                        consumed <= consumed + 1'b1;
                        state <= IDLE;
                        if (tile_x == TX-1) begin
                            tile_x <= 0;
                            tile_y <= tile_y + 1'b1;
                        end else tile_x <= tile_x + 1'b1;
                        if (consumed == TX*TY-1) begin
                            active <= 0;
                            capture_done <= 1;
                        end
                    end else begin
                        y <= y + 1'b1;
                        x <= 0;
                        c <= 0;
                        state <= ISSUE;
                    end
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
