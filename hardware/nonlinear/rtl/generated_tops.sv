`timescale 1ns/1ps

module nl_n0_blk0_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd64197528), .SBSU_Q40(48'd1244279650), .DECAY_Q24(768'h00000fab339f00000ee661f200000e2ad4f100000ccc283300000b69bd5000000a08aff1000009b40e4b000008aff0d3000007a31afa000006a6f5cf000005cd3d2600000479435a0000040ccc0d000002e59a55000001e3f3b6000000eab8a1), .K_FRAC(24),
        .ROM_FILE("blk0_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk0_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd64197528), .SBSU_Q40(48'd1244279650), .DECAY_Q24(768'h00000fab339f00000ee661f200000e2ad4f100000ccc283300000b69bd5000000a08aff1000009b40e4b000008aff0d3000007a31afa000006a6f5cf000005cd3d2600000479435a0000040ccc0d000002e59a55000001e3f3b6000000eab8a1), .K_FRAC(24),
        .ROM_FILE("blk0_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk0_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd64197528), .SBSU_Q40(48'd1244279650), .DECAY_Q24(768'h00000fab339f00000ee661f200000e2ad4f100000ccc283300000b69bd5000000a08aff1000009b40e4b000008aff0d3000007a31afa000006a6f5cf000005cd3d2600000479435a0000040ccc0d000002e59a55000001e3f3b6000000eab8a1), .K_FRAC(24),
        .ROM_FILE("blk0_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n0_blk0_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd50962724), .SBSU_Q40(48'd497794115), .DECAY_Q24(768'h000010113c0400000e02238a00000de844cc00000cdfbe2600000b7e85af00000ac39acd00000a8234aa00000985b5b7000007d1f041000006b2924e000005c22515000004f11d5f000003eb259b00000307ce2d000001f4c22e00000100ba47), .K_FRAC(24),
        .ROM_FILE("blk0_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk0_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd50962724), .SBSU_Q40(48'd497794115), .DECAY_Q24(768'h000010113c0400000e02238a00000de844cc00000cdfbe2600000b7e85af00000ac39acd00000a8234aa00000985b5b7000007d1f041000006b2924e000005c22515000004f11d5f000003eb259b00000307ce2d000001f4c22e00000100ba47), .K_FRAC(24),
        .ROM_FILE("blk0_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk0_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd50962724), .SBSU_Q40(48'd497794115), .DECAY_Q24(768'h000010113c0400000e02238a00000de844cc00000cdfbe2600000b7e85af00000ac39acd00000a8234aa00000985b5b7000007d1f041000006b2924e000005c22515000004f11d5f000003eb259b00000307ce2d000001f4c22e00000100ba47), .K_FRAC(24),
        .ROM_FILE("blk0_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n0_blk1_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd65059028), .SBSU_Q40(48'd5850617722), .DECAY_Q24(768'h0000106f242d00000f285cba00000e8411c400000cf8241f00000c3e0b8a00000b08e7c900000a163fe1000008425dfc0000079ef0f1000006c2c17d000006068f21000004e6483b000004075294000002dff18f000001e2c6a0000000f9f920), .K_FRAC(23),
        .ROM_FILE("blk1_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk1_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd65059028), .SBSU_Q40(48'd5850617722), .DECAY_Q24(768'h0000106f242d00000f285cba00000e8411c400000cf8241f00000c3e0b8a00000b08e7c900000a163fe1000008425dfc0000079ef0f1000006c2c17d000006068f21000004e6483b000004075294000002dff18f000001e2c6a0000000f9f920), .K_FRAC(23),
        .ROM_FILE("blk1_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk1_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd65059028), .SBSU_Q40(48'd5850617722), .DECAY_Q24(768'h0000106f242d00000f285cba00000e8411c400000cf8241f00000c3e0b8a00000b08e7c900000a163fe1000008425dfc0000079ef0f1000006c2c17d000006068f21000004e6483b000004075294000002dff18f000001e2c6a0000000f9f920), .K_FRAC(23),
        .ROM_FILE("blk1_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n0_blk1_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd61719692), .SBSU_Q40(48'd1665361266), .DECAY_Q24(768'h00000f4e255700000f57cd5000000e36215b00000d31abe000000d09748c00000af356bf00000a41524e000009341739000007bd8a4e000006abc4cc0000068b83260000051a9aa40000040be8e000000314cc3200000202a202000000f9d1d6), .K_FRAC(24),
        .ROM_FILE("blk1_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk1_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd61719692), .SBSU_Q40(48'd1665361266), .DECAY_Q24(768'h00000f4e255700000f57cd5000000e36215b00000d31abe000000d09748c00000af356bf00000a41524e000009341739000007bd8a4e000006abc4cc0000068b83260000051a9aa40000040be8e000000314cc3200000202a202000000f9d1d6), .K_FRAC(24),
        .ROM_FILE("blk1_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk1_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd61719692), .SBSU_Q40(48'd1665361266), .DECAY_Q24(768'h00000f4e255700000f57cd5000000e36215b00000d31abe000000d09748c00000af356bf00000a41524e000009341739000007bd8a4e000006abc4cc0000068b83260000051a9aa40000040be8e000000314cc3200000202a202000000f9d1d6), .K_FRAC(24),
        .ROM_FILE("blk1_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n0_blk2_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd70843760), .SBSU_Q40(48'd15015494378), .DECAY_Q24(768'h00001127b66700000f99aef500000dbadcb200000d18281000000bf2126000000b20decb000009dc84fd000008c8ed100000081be187000006cbd3bf000006023f42000004d4163c000003b2893a000002f62a280000020782ca000000f45c25), .K_FRAC(22),
        .ROM_FILE("blk2_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk2_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd70843760), .SBSU_Q40(48'd15015494378), .DECAY_Q24(768'h00001127b66700000f99aef500000dbadcb200000d18281000000bf2126000000b20decb000009dc84fd000008c8ed100000081be187000006cbd3bf000006023f42000004d4163c000003b2893a000002f62a280000020782ca000000f45c25), .K_FRAC(22),
        .ROM_FILE("blk2_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk2_spa (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd70843760), .SBSU_Q40(48'd15015494378), .DECAY_Q24(768'h00001127b66700000f99aef500000dbadcb200000d18281000000bf2126000000b20decb000009dc84fd000008c8ed100000081be187000006cbd3bf000006023f42000004d4163c000003b2893a000002f62a280000020782ca000000f45c25), .K_FRAC(22),
        .ROM_FILE("blk2_spa_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n0_blk2_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(0), .SDT_Q30(32'd61817588), .SBSU_Q40(48'd5629172119), .DECAY_Q24(768'h00000f9ead8500000e47a88300000d40b2a100000c5cc70c00000becbb0e000009ec7aeb00000998d73a000008db225c0000083ee5af000006d1c24f000005abf36c000004ec938a000004202eea000002fb70a4000001ff8d0d000000ff3818), .K_FRAC(23),
        .ROM_FILE("blk2_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n1_blk2_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(1), .SDT_Q30(32'd61817588), .SBSU_Q40(48'd5629172119), .DECAY_Q24(768'h00000f9ead8500000e47a88300000d40b2a100000c5cc70c00000becbb0e000009ec7aeb00000998d73a000008db225c0000083ee5af000006d1c24f000005abf36c000004ec938a000004202eea000002fb70a4000001ff8d0d000000ff3818), .K_FRAC(23),
        .ROM_FILE("blk2_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule

module nl_n3_blk2_spe (
    input  wire clk, rst, in_valid,
    input  wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    nl_coeff #(
        .METHOD(3), .SDT_Q30(32'd61817588), .SBSU_Q40(48'd5629172119), .DECAY_Q24(768'h00000f9ead8500000e47a88300000d40b2a100000c5cc70c00000becbb0e000009ec7aeb00000998d73a000008db225c0000083ee5af000006d1c24f000005abf36c000004ec938a000004202eea000002fb70a4000001ff8d0d000000ff3818), .K_FRAC(23),
        .ROM_FILE("blk2_spe_n3.mem")
    ) u_core (
        .clk(clk), .rst(rst), .in_valid(in_valid), .q_dt(q_dt),
        .out_valid(out_valid), .out_address(out_address), .out_coeff(out_coeff)
    );
endmodule
