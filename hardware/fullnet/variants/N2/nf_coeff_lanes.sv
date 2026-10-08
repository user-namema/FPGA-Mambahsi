`timescale 1ns/1ps
// N2-only wrapper candidate. No method-dependent instance selection.
// Original nl_coeff_n2 and all 13 pipeline registers are left untouched.
// Use ONLY in an NF_METHOD=2 package.
module nf_coeff_lanes #(
    parameter integer METHOD=2, BLOCK_ID=0, IS_SPE=0, LANES=1
)(
    input wire clk, reset, in_valid,
    input wire [LANES*8-1:0] q_dt,
    output wire out_valid,
    output wire [418:0] data0, data1, data2, data3
);
    localparam integer CORE_INDEX = BLOCK_ID*2+IS_SPE;
    localparam [31:0] SDT_Q30 =
        CORE_INDEX==0 ? 32'd64197528 :
        CORE_INDEX==1 ? 32'd50962724 :
        CORE_INDEX==2 ? 32'd65059028 :
        CORE_INDEX==3 ? 32'd61719692 :
        CORE_INDEX==4 ? 32'd70843760 :
        32'd61817588;
    localparam [47:0] SBSU_Q40 =
        CORE_INDEX==0 ? 48'd1244279650 :
        CORE_INDEX==1 ? 48'd497794115 :
        CORE_INDEX==2 ? 48'd5850617722 :
        CORE_INDEX==3 ? 48'd1665361266 :
        CORE_INDEX==4 ? 48'd15015494378 :
        48'd5629172119;
    localparam [767:0] DECAY_Q24 =
        CORE_INDEX==0 ? 768'h00000fab339f00000ee661f200000e2ad4f100000ccc283300000b69bd5000000a08aff1000009b40e4b000008aff0d3000007a31afa000006a6f5cf000005cd3d2600000479435a0000040ccc0d000002e59a55000001e3f3b6000000eab8a1 :
        CORE_INDEX==1 ? 768'h000010113c0400000e02238a00000de844cc00000cdfbe2600000b7e85af00000ac39acd00000a8234aa00000985b5b7000007d1f041000006b2924e000005c22515000004f11d5f000003eb259b00000307ce2d000001f4c22e00000100ba47 :
        CORE_INDEX==2 ? 768'h0000106f242d00000f285cba00000e8411c400000cf8241f00000c3e0b8a00000b08e7c900000a163fe1000008425dfc0000079ef0f1000006c2c17d000006068f21000004e6483b000004075294000002dff18f000001e2c6a0000000f9f920 :
        CORE_INDEX==3 ? 768'h00000f4e255700000f57cd5000000e36215b00000d31abe000000d09748c00000af356bf00000a41524e000009341739000007bd8a4e000006abc4cc0000068b83260000051a9aa40000040be8e000000314cc3200000202a202000000f9d1d6 :
        CORE_INDEX==4 ? 768'h00001127b66700000f99aef500000dbadcb200000d18281000000bf2126000000b20decb000009dc84fd000008c8ed100000081be187000006cbd3bf000006023f42000004d4163c000003b2893a000002f62a280000020782ca000000f45c25 :
        768'h00000f9ead8500000e47a88300000d40b2a100000c5cc70c00000becbb0e000009ec7aeb00000998d73a000008db225c0000083ee5af000006d1c24f000005abf36c000004ec938a000004202eea000002fb70a4000001ff8d0d000000ff3818;
    localparam integer K_FRAC =
        CORE_INDEX==0 ? 24 :
        CORE_INDEX==1 ? 24 :
        CORE_INDEX==2 ? 23 :
        CORE_INDEX==3 ? 24 :
        CORE_INDEX==4 ? 22 :
        23;
    localparam [31:0] EXP_L =
        CORE_INDEX==0 ? 32'h3d903436 :
        CORE_INDEX==1 ? 32'h3d903436 :
        CORE_INDEX==2 ? 32'h3d903436 :
        CORE_INDEX==3 ? 32'h3d903436 :
        CORE_INDEX==4 ? 32'h3d903436 :
        32'h3d903436;
    localparam [31:0] EXP_B =
        CORE_INDEX==0 ? 32'h3d84fabc :
        CORE_INDEX==1 ? 32'h3d84fabc :
        CORE_INDEX==2 ? 32'h3d84fabc :
        CORE_INDEX==3 ? 32'h3d84fabc :
        CORE_INDEX==4 ? 32'h3d84fabc :
        32'h3d84fabc;
    localparam signed [31:0] LOG_L =
        CORE_INDEX==0 ? 32'h2f902724 :
        CORE_INDEX==1 ? 32'h2f902724 :
        CORE_INDEX==2 ? 32'h2f902724 :
        CORE_INDEX==3 ? 32'h2f902724 :
        CORE_INDEX==4 ? 32'h2f902724 :
        32'h2f902724;
    localparam signed [31:0] LOG_B =
        CORE_INDEX==0 ? 32'h00b16361 :
        CORE_INDEX==1 ? 32'h00b16361 :
        CORE_INDEX==2 ? 32'h00b16361 :
        CORE_INDEX==3 ? 32'h00b16361 :
        CORE_INDEX==4 ? 32'h00b16361 :
        32'h00b16361;
    localparam [31:0] STATE_L =
        CORE_INDEX==0 ? 32'h3f7ce978 :
        CORE_INDEX==1 ? 32'h3f7ce978 :
        CORE_INDEX==2 ? 32'h3f7ce978 :
        CORE_INDEX==3 ? 32'h3f7ce978 :
        CORE_INDEX==4 ? 32'h3f7ce978 :
        32'h3f7ce978;
    localparam [31:0] STATE_B =
        CORE_INDEX==0 ? 32'h3c96a4c3 :
        CORE_INDEX==1 ? 32'h3c96a4c3 :
        CORE_INDEX==2 ? 32'h3c96a4c3 :
        CORE_INDEX==3 ? 32'h3c96a4c3 :
        CORE_INDEX==4 ? 32'h3c96a4c3 :
        32'h3c96a4c3;
    wire [LANES*419-1:0] coeff;
    wire [LANES-1:0] valid;
    wire [1675:0] padded;
    assign padded = {{(4-LANES)*419{1'b0}}, coeff};
    assign {data3,data2,data1,data0} = padded;
    assign out_valid = valid[0];
    genvar lane;
    generate
        for (lane=0; lane<LANES; lane=lane+1) begin : G_LANE
            wire [418:0] local_coeff;
            nl_coeff_n2 #(
                .SDT_Q30(SDT_Q30),
                .SBSU_Q40(SBSU_Q40),
                .DECAY_Q24(DECAY_Q24),
                .K_FRAC(K_FRAC),
                .EXP_L(EXP_L),
                .EXP_B(EXP_B),
                .LOG_L(LOG_L),
                .LOG_B(LOG_B),
                .STATE_L(STATE_L),
                .STATE_B(STATE_B)
            ) u_coeff (
                .clk(clk), .rst(reset), .in_valid(in_valid),
                .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
                .out_address(), .out_coeff(local_coeff)
            );
            assign coeff[lane*419+:419] = local_coeff;
        end
    endgenerate
endmodule

