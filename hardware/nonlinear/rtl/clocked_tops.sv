`timescale 1ns/1ps

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk0_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk0_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk0_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk0_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk0_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk0_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk0_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk0_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk0_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk0_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk0_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk0_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk0_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk0_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk0_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk0_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk1_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk1_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk1_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk1_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk1_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk1_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk1_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk1_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk1_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk1_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk1_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk1_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk1_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk1_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk1_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk1_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk2_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk2_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk2_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk2_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk2_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk2_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk2_spa (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk2_spa u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n0_blk2_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n0_blk2_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n1_blk2_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n1_blk2_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n2_blk2_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n2_blk2_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule

// Clocked load shell: report u_dut resources separately from boundary/clock cost.
module nl_clocked_n3_blk2_spe (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output reg out_valid,
    output reg [7:0] out_address,
    output reg [418:0] out_coeff
);
    wire core_clk;
    BUFGCE #(.CE_TYPE("SYNC")) u_clk(.I(clk),.CE(1'b1),.O(core_clk));
    reg rst_q,in_valid_q;
    reg signed [7:0] q_dt_q;
    always @(posedge core_clk) begin
        rst_q<=rst;
        in_valid_q<=in_valid;
        q_dt_q<=q_dt;
    end
    wire v;
    wire [7:0] address;
    wire [418:0] coeff;
    nl_n3_blk2_spe u_dut(.clk(core_clk),.rst(rst_q),.in_valid(in_valid_q),.q_dt(q_dt_q),
                            .out_valid(v),.out_address(address),.out_coeff(coeff));
    always @(posedge core_clk) begin
        out_valid<=v;
        out_address<=address;
        out_coeff<=coeff;
    end
endmodule
