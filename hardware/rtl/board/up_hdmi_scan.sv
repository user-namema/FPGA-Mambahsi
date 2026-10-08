`timescale 1ns/1ps
// 1280x720 timing, 1650x750 total; 75 MHz matches supplied demo (60.606 Hz).
// UP is displayed 1:1 centred, with black border. No DDR reads during visible pixel critical path.

module up_hdmi_scan # (
    parameter WIDTH = 340, HEIGHT = 610, PITCH = 384,
    parameter integer PROGRESSIVE = 0,
    // Defaults preserve the board-tested 720p timing. Small values are used
    // only by the joint network/DDR/display regression to shorten blanking.
    parameter H_ACTIVE = 1280, H_SYNC = 40, H_BACK = 220, H_TOTAL = 1650,
    parameter V_ACTIVE = 720, V_SYNC = 5, V_BACK = 20, V_TOTAL = 750
) (
    input wire clk, rst_n, frame_valid_async,
    output wire req_push,
    input wire req_full,
    output wire [9 : 0] req_row,
    input wire res_empty,
    output wire res_pop,
    input wire [524 : 0] res_data,
    output reg hs, vs, de,
    output reg [23 : 0] rgb,
    output reg underflow
);

    localparam XSTART = H_SYNC + H_BACK + (H_ACTIVE - WIDTH) / 2;
    localparam YSTART = V_SYNC + V_BACK + (V_ACTIVE - HEIGHT) / 2;
    reg [10 : 0] h;
    reg [9 : 0] v;
    (* ASYNC_REG = "TRUE" *) reg fv0, fv1;
    reg show_frame;
    reg [511 : 0] lines [0 : 2 * (PITCH / 64) - 1];
    reg [9 : 0] tags [0 : 1];
    reg [1 : 0] ready;
    wire [9 : 0] ret_row = res_data [524 : 515];
    wire [2 : 0] ret_word = res_data [514 : 512];
    wire kick0 = h == 0 && v == 0 && fv1;
    wire kicknext = h == 0 && v >= YSTART && v < YSTART + HEIGHT - 1 && show_frame;
    assign req_push = (kick0 || kicknext) && ! req_full;
    assign req_row = kick0 ? 10'd0 : v - YSTART + 1;
    assign res_pop = ! res_empty;
    integer sx, sy, slot;
    reg [7 : 0] label_value;
    reg [23 : 0] color;
    function automatic [23 : 0] palette (input [7 : 0] id);
        case (id)
            0 : palette = 24'hff0000;
            1 : palette = 24'h00ff00;
            2 : palette = 24'h0000ff;
            3 : palette = 24'hffff00;
            4 : palette = 24'hff00ff;
            5 : palette = 24'h00ffff;
            6 : palette = 24'hc86400;
            7 : palette = 24'h00c864;
            8 : palette = 24'h6400c8;
            255 : palette = PROGRESSIVE ? 24'h202020 : 24'h000000;
            default : palette = 24'h000000;
        endcase
    endfunction
    always @ * begin
        sx = h - XSTART;
        sy = v - YSTART;
        slot = sy & 1;
        label_value = 0;
        color = 0;
        // A missed line is diagnostic cyan, not a misleading black frame.
        if (PROGRESSIVE && show_frame && sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT)
            color = 24'h0080ff;
        if (show_frame && sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT && ready [slot] && tags [slot] == sy) begin
            label_value = lines [slot * (PITCH / 64) + (sx / 64)] [(sx % 64) * 8 +: 8];
            color = palette (label_value);
        end
    end
    always @ (posedge clk) begin
        if (! rst_n) begin
            h <= 0;
            v <= 0;
            fv0 <= 0;
            fv1 <= 0;
            show_frame <= 0;
            ready <= 0;
            tags [0] <= 0;
            tags [1] <= 0;
            hs <= 1;
            vs <= 1;
            de <= 0;
            rgb <= 0;
            underflow <= 0;
        end
        else begin
            fv0 <= frame_valid_async;
            fv1 <= fv0;
            if (h == H_TOTAL-1) begin
                h <= 0;
                if (v == V_TOTAL-1) v <= 0;
                else v <= v + 1'b1;
            end
            else h <= h + 1'b1;
            if (h == 0 && v == 0) show_frame <= fv1;
            if ((kick0 || kicknext) && req_full) underflow <= 1;
            if (res_pop) begin
                lines [ret_row [0] * (PITCH / 64) + ret_word] <= res_data [511 : 0];
                if (ret_word == PITCH / 64 - 1) begin
                    ready [ret_row [0]] <= 1;
                    tags [ret_row [0]] <= ret_row;
                end
            end
            if (show_frame && sx >= 0 && sx < WIDTH && sy >= 0 && sy < HEIGHT && (! ready [slot] || tags [slot] != sy)) underflow <= 1;
            hs <= ! (h < H_SYNC);
            vs <= ! (v < V_SYNC);
            de <= h >= H_SYNC+H_BACK && h < H_SYNC+H_BACK+H_ACTIVE &&
                  v >= V_SYNC+V_BACK && v < V_SYNC+V_BACK+V_ACTIVE;
            rgb <= color;
        end
    end
endmodule
