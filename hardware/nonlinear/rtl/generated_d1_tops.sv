`timescale 1ns/1ps

module nl_d1_n0_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk0_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk0_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk0_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk0_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk0_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk0_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk0_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk0_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk0_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n0_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk0_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk0_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk0_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk0_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk0_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk0_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk0_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(0), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk0_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk0_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n0_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk1_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk1_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk1_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk1_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk1_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk1_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk1_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk1_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk1_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n0_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk1_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk1_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk1_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk1_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk1_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk1_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk1_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(1), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk1_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk1_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n0_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk2_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk2_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk2_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk2_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk2_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk2_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk2_spa u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(0)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk2_spa (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk2_spa u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n0_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n0_blk2_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n0_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n0_blk2_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n1_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n1_blk2_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n1_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n1_blk2_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n2_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n2_blk2_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n2_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n2_blk2_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule

module nl_d1_n3_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff,
    output wire readout_valid,
    output wire signed [47:0] total_readout
);
    // Independent coefficient/readout streams; not a full SSM latency model.
    nl_n3_blk2_spe u_coeff (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
    nl_d1_readout #(.BLOCK_ID(2), .IS_SPE(1)) u_readout (
        .clk(clk), .rst(rst), .read_valid(read_valid), .channel(channel),
        .u_s8(u_s8), .c_readout(c_readout),
        .out_valid(readout_valid), .total_readout(total_readout)
    );
endmodule

module nl_clocked_d1_n3_blk2_spe (
    input wire clk, rst, in_valid,
    input wire signed [7:0] q_dt,
    input wire read_valid,
    input wire [5:0] channel,
    input wire [7:0] u_s8,
    input wire signed [47:0] c_readout,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff,
    output reg readout_valid,
    output reg signed [47:0] total_readout
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk), .CE(1'b1), .O(core_clk));
    reg rst_q, in_valid_q, read_valid_q;
    reg signed [7:0] q_dt_q;
    reg [5:0] channel_q;
    reg [7:0] u_s8_q;
    reg signed [47:0] c_readout_q;
    always @(posedge core_clk) begin
        rst_q <= rst;
        in_valid_q <= in_valid;
        read_valid_q <= read_valid;
        q_dt_q <= q_dt;
        channel_q <= channel;
        u_s8_q <= u_s8;
        c_readout_q <= c_readout;
    end
    wire v, rv;
    wire [7:0] addr;
    wire [418:0] coeff;
    wire signed [47:0] total;
    nl_d1_n3_blk2_spe u_dut (
        .clk(core_clk), .rst(rst_q), .in_valid(in_valid_q), .q_dt(q_dt_q),
        .read_valid(read_valid_q), .channel(channel_q), .u_s8(u_s8_q), .c_readout(c_readout_q),
        .out_valid(v), .out_address(addr), .out_coeff(coeff),
        .readout_valid(rv), .total_readout(total)
    );
    always @(posedge core_clk) begin
        out_valid <= v;
        out_address <= addr;
        out_coeff <= coeff;
        readout_valid <= rv;
        total_readout <= total;
    end
endmodule
