`timescale 1ns / 1ps
`default_nettype none

// One fused Block0-Spa selective-scan state cell.
//
// The exported v5 LUT contains Abar in UQ1.24 and K in UQ?.24.  B and U are
// signed INT8.  There is no rounding between K*B and (K*B)*U in the reference,
// therefore the associative rewrite below is bit exact:
//
//   (K * B) * U == K * (B * U)
//
// Two 3-cycle Multiplier Generator IPs form Abar*state from 16-bit state
// limbs.  A third 3-cycle IP forms K*(B*U); the small signed 8x8 B*U product
// is deliberately kept in LUT fabric.  Both terms are combined in Q48.
// Rounding/saturation remains outside this module so there is still exactly
// one Q48 -> Q24 conversion.
//
// The multiplier results are valid after three rising edges.  Limb
// recombination and the final Q48 add are two registered stages, therefore
// STATE_ACC_Q48 is valid exactly five rising edges after input sampling.
// STATE_FORWARD_Q48 exposes the same final add one edge earlier, before its
// output register.  Block0 Spe uses it only for an exact same-group
// recurrence bypass; it does not add another adder because both outputs share
// state_acc_comb.
module ssm_update_3dsp #(
    parameter integer K_FRACTION_BITS = 24
) (
    input  wire                 CLK,
    input  wire                 VALID,
    input  wire        [24:0]   ABAR_U25,
    input  wire signed [31:0]   STATE_S32,
    input  wire        [18:0]   K_U19,
    input  wire signed [7:0]    B_S8,
    input  wire signed [7:0]    U_S8,
    output reg  signed [64:0]   STATE_ACC_Q48,
    output wire signed [64:0]   STATE_FORWARD_Q48
);
    localparam integer KBU_TO_Q48_SHIFT = 48-K_FRACTION_BITS;

    wire signed [26:0] abar_dsp;
    wire signed [17:0] state_low_dsp;
    wire signed [17:0] state_high_dsp;
    assign abar_dsp = $signed({2'b00, ABAR_U25});
    assign state_low_dsp = $signed({2'b00, STATE_S32[15:0]});
    assign state_high_dsp
        = $signed({{2{STATE_S32[31]}}, STATE_S32[31:16]});

    // B*U stays an 8x8 LUT multiplication.  Its result and K are sampled by
    // the 20x16 multiplier IP on the same edge as the two A/state IPs.
    (* use_dsp = "no" *) wire signed [15:0] bu_product;
    assign bu_product = B_S8 * U_S8;

    wire signed [44:0] ah_low_product;
    wire signed [44:0] ah_high_product;
    wire signed [35:0] kbu_product;
    wire signed [19:0] k_dsp = $signed({1'b0, K_U19});

    ssm_abar_mult_27x18_3cyc u_abar_state_low (
        .CLK(CLK), .A(abar_dsp), .B(state_low_dsp), .P(ah_low_product));
    ssm_abar_mult_27x18_3cyc u_abar_state_high (
        .CLK(CLK), .A(abar_dsp), .B(state_high_dsp), .P(ah_high_product));
    ssm_kbu_mult_20x16_3cyc u_k_bu (
        .CLK(CLK), .A(k_dsp), .B(bu_product), .P(kbu_product));

    wire signed [64:0] ah_low_extended;
    wire signed [64:0] ah_high_extended;
    assign ah_low_extended = {{20{ah_low_product[44]}}, ah_low_product};
    assign ah_high_extended = {{20{ah_high_product[44]}}, ah_high_product};

    // Stage 4/5 after the three-cycle IPs.  These two physical add stages are
    // intentionally separated; they are not a combinational three-operand
    // tree.  Vivado may use the DSP post-adder for the low 48-bit portion and
    // needs only a short fabric extension for the signed Q48 guard bits.
    (* use_dsp = "yes" *) reg signed [64:0] ah_q;
    (* use_dsp = "yes" *) reg signed [64:0] kbu_q48_q;
    wire signed [64:0] kbu_extended;
    wire signed [64:0] kbu_q48;
    assign kbu_extended = {{29{kbu_product[35]}}, kbu_product};
    assign kbu_q48 = kbu_extended <<< KBU_TO_Q48_SHIFT;

    wire signed [64:0] state_acc_comb;
    assign state_acc_comb = ah_q + kbu_q48_q;
    assign STATE_FORWARD_Q48 = state_acc_comb;

    always @(posedge CLK) begin
        ah_q      <= ah_low_extended + (ah_high_extended <<< 16);
        kbu_q48_q <= kbu_q48;
        STATE_ACC_Q48 <= state_acc_comb;
    end

    wire unused_valid = VALID;
endmodule

`default_nettype wire
