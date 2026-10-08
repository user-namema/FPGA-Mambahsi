`timescale 1ns/1ps
`default_nettype none

// Read 4 KiB from actual DDR, one outstanding 64-byte read at a time.
// Store the page in the EXISTING diagnostic RAM, C2H address 0x00020000.
// No frame-sized shadow RAM. The page remains unchanged until the next command.
module ddr_readback_page (
    input wire pcie_clk, pcie_rst_n, ui_clk, ui_rst_n,
    input wire start,
    input wire [7:0] page,
    output wire busy,
    output reg done, error,
    output wire mem_wr_en,
    output wire [31:0] mem_byte_addr,
    output wire [127:0] mem_data,
    output wire request_valid,
    input wire request_ready,
    output wire [31:0] request_byte_addr,
    input wire response_valid,
    output wire response_ready,
    input wire [511:0] response_data
);
    wire qfull, qempty, rfull, rempty;
    wire [31:0] qdata;
    wire [511:0] rdata;
    reg [1:0] state;
    localparam IDLE = 0, REQUEST = 1, RESPONSE = 2, STORE = 3;
    reg [7:0] page_q;
    reg [5:0] word_index;
    reg [1:0] lane;
    reg [511:0] payload;
    reg [27:0] watchdog;
    assign busy = state != IDLE;
    wire request_push = state == REQUEST && !qfull;
    wire response_pop = state == RESPONSE && !rempty;
    assign request_valid = !qempty;
    assign request_byte_addr = qdata;
    assign response_ready = !rfull;
    hdmi_async_fifo #(.WIDTH(32), .DEPTH(16)) requests (
        .rst(!pcie_rst_n || !ui_rst_n), .wr_clk(pcie_clk), .rd_clk(ui_clk),
        .din({12'b0, page_q, word_index, 6'b0}), .push(request_push),
        .full(qfull), .empty(qempty), .dout(qdata),
        .pop(request_valid && request_ready)
    );
    hdmi_async_fifo #(.WIDTH(512), .DEPTH(16)) responses (
        .rst(!pcie_rst_n || !ui_rst_n), .wr_clk(ui_clk), .rd_clk(pcie_clk),
        .din(response_data), .push(response_valid && response_ready),
        .full(rfull), .empty(rempty), .dout(rdata), .pop(response_pop)
    );
    assign mem_wr_en = state == STORE;
    assign mem_byte_addr = {20'b0, word_index, lane, 4'b0};
    assign mem_data = payload[127:0];
    always @(posedge pcie_clk) begin
        if (!pcie_rst_n) begin
            state <= IDLE;
            page_q <= 0;
            word_index <= 0;
            lane <= 0;
            done <= 0;
            error <= 0;
            watchdog <= 0;
        end else begin
            if (!busy) watchdog <= 0;
            else if (&watchdog) error <= 1;
            else watchdog <= watchdog + 1'b1;
            if (start && busy) error <= 1;
            case (state)
                IDLE: if (start) begin
                    page_q <= page;
                    word_index <= 0;
                    done <= 0;
                    state <= REQUEST;
                end
                REQUEST: if (request_push) state <= RESPONSE;
                RESPONSE: if (response_pop) begin
                    payload <= rdata;
                    lane <= 0;
                    state <= STORE;
                end
                STORE: begin
                    payload <= {128'b0, payload[511:128]};
                    if (lane == 3) begin
                        if (word_index == 63) begin
                            state <= IDLE;
                            done <= 1;
                        end else begin
                            word_index <= word_index + 1'b1;
                            state <= REQUEST;
                        end
                    end else lane <= lane + 1'b1;
                end
                default: state <= IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
