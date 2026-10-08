`timescale 1ns/1ps
`default_nettype none

// One physical four-state endpoint for the Block1/2 fused core.  The local
// module owns its 128-bit state BRAM, four recurrence cells, four area-first
// five-cycle C multipliers and the local 4-to-1 reduction.  Consequently the
// parent no longer routes a 512-bit state write word or sixteen 40-bit C
// products across the whole lower/upper core.
module ssm_block1_spe_state_bank4 #(
    parameter integer BLOCK_ID = 1,
    parameter integer GROUP_COUNT = 8,
    parameter integer GROUP_ADDR_WIDTH = 3,
    parameter integer STATE_BASE = 0,
    parameter integer K_FRACTION_BITS = 24
) (
    input  wire                         clk,
    input  wire                         issue_valid,
    input  wire                         reset_state,
    input  wire [GROUP_ADDR_WIDTH-1:0]  read_group,
    input  wire                         state_write_valid,
    input  wire [GROUP_ADDR_WIDTH-1:0]  state_write_group,
    input  wire [99:0]                  abar_word,
    input  wire [18:0]                  k_word,
    input  wire [31:0]                  b_states,
    input  wire signed [7:0]            u_s8,
    input  wire [31:0]                  c_states,
    output wire signed [41:0]           c_sum_s42
);
    wire [127:0] state_bank_read;
    wire [127:0] state_bank_write;
    wire signed [64:0] state_accumulator [0:3];
    wire signed [31:0] state_next_comb [0:3];
    reg  signed [31:0] state_next_q [0:3];
    reg [31:0] c_states_q;
    wire signed [39:0] c_product [0:3];

    function signed [31:0] round_shift24_to_int32;
        input signed [64:0] value;
        reg round_up;
        reg signed [41:0] shifted_rounded;
        begin
            round_up = value[64]
                     ? (value[23] && (|value[22:0]))
                     : value[23];
            shifted_rounded = $signed({value[64], value[64:24]})
                            + $signed({41'd0, round_up});
            if (shifted_rounded[41:31] == {11{shifted_rounded[31]}})
                round_shift24_to_int32 = shifted_rounded[31:0];
            else if (shifted_rounded[41])
                round_shift24_to_int32 = 32'sh80000000;
            else
                round_shift24_to_int32 = 32'sh7fffffff;
        end
    endfunction

    generate
        if (BLOCK_ID == 1) begin : G_BLOCK1_STATE_BRAM
            // Four-state endpoints are exactly 128 bits wide.  Reuse the
            // existing 256x128 BRAM through a one-cycle adapter instead of
            // distributing the address/write-enable over 128 RAMD cells.
            ssm_block1_state_bram_128banked #(
                .WIDTH(128), .DEPTH(GROUP_COUNT),
                .ADDR_WIDTH(GROUP_ADDR_WIDTH)
            ) u_state_bank (
                .clk(clk), .wr_en(state_write_valid),
                .wr_addr(state_write_group), .wr_data(state_bank_write),
                .rd_en(1'b1), .rd_addr(read_group),
                .rd_data(state_bank_read));
        end else begin : G_BLOCK2_STATE_BRAM
            wire [7:0] read_group_256
                = {{(8-GROUP_ADDR_WIDTH){1'b0}}, read_group};
            wire [7:0] write_group_256
                = {{(8-GROUP_ADDR_WIDTH){1'b0}}, state_write_group};
            // One 128-bit bank per four states.  Two adjacent banks form the
            // requested 256-bit Block2 physical state endpoint; no 1024-bit
            // parent read/write bus crosses the SSM anymore.
            ssm_scratch_bram_256x128 u_state_bank (
                .clka(clk), .ena(state_write_valid),
                .wea({state_write_valid}), .addra(write_group_256),
                .dina(state_bank_write), .clkb(clk), .enb(1'b1),
                .addrb(read_group_256), .doutb(state_bank_read));
        end
    endgenerate

    genvar lane;
    generate
        for (lane=0; lane<4; lane=lane+1) begin : GEN_STATE
            wire signed [31:0] old_state = reset_state ? 32'sd0
                : $signed(state_bank_read[lane*32 +: 32]);
            ssm_update_3dsp #(.K_FRACTION_BITS(K_FRACTION_BITS))
            u_fused_update (
                .CLK(clk), .VALID(issue_valid),
                .ABAR_U25(abar_word[lane*25 +: 25]),
                .STATE_S32(old_state), .K_U19(k_word),
                .B_S8($signed(b_states[lane*8 +: 8])),
                .U_S8(u_s8),
                .STATE_ACC_Q48(state_accumulator[lane]));
            assign state_next_comb[lane]
                = round_shift24_to_int32(state_accumulator[lane]);
            assign state_bank_write[lane*32 +: 32] = state_next_q[lane];

            // Existing verified five-cycle area-priority Multiplier Generator.
            ssm_c_product_32x8_5cyc u_c_product (
                .clk(clk), .state_s32(state_next_q[lane]),
                .c_s8($signed(c_states_q[lane*8 +: 8])),
                .product_s40(c_product[lane]));
        end
    endgenerate

    // Payload is deliberately free-running; parent valid pipelines qualify
    // state writes and the returned partial sum.
    integer capture_index;
    always @(posedge clk) begin
        c_states_q <= c_states;
        for (capture_index=0; capture_index<4;
             capture_index=capture_index+1)
            state_next_q[capture_index]
                <= state_next_comb[capture_index];
    end

    reg signed [39:0] c_product_q [0:3];
    reg signed [40:0] c_level1 [0:1];
    reg signed [41:0] c_level2;
    integer reduce_index;
    always @(posedge clk) begin
        for (reduce_index=0; reduce_index<4; reduce_index=reduce_index+1)
            c_product_q[reduce_index] <= c_product[reduce_index];
        c_level1[0]
            <= $signed({c_product_q[0][39], c_product_q[0]})
             + $signed({c_product_q[1][39], c_product_q[1]});
        c_level1[1]
            <= $signed({c_product_q[2][39], c_product_q[2]})
             + $signed({c_product_q[3][39], c_product_q[3]});
        c_level2
            <= $signed({c_level1[0][40], c_level1[0]})
             + $signed({c_level1[1][40], c_level1[1]});
    end
    assign c_sum_s42 = c_level2;

`ifndef SYNTHESIS
    initial begin
        // Block2 Spa exports K in Q23; every other scaled branch uses Q24.
        // ssm_update_3dsp compensates this difference before producing the
        // common Q48 accumulator, so the state rounding below remains Q24.
        if ((STATE_BASE < 0) || (STATE_BASE > 12)
            || !((K_FRACTION_BITS == 22)
                 || (K_FRACTION_BITS == 23)
                 || (K_FRACTION_BITS == 24))) begin
            $error("Unsupported scaled four-state bank geometry");
            $finish;
        end
    end
`endif
endmodule

// Area-scaled fused selective-scan core for Block1 and Block2.
//
// Block1 keeps 16 state lanes and Block2 keeps 8 state lanes.  A request is
// one channel/state-chunk, so the useful work is:
//   Block1: 64/16 groups per Spa/Spe record
//   Block2: 128/32 groups per Spa/Spe record
// Each state lane evaluates the exact v5 recurrence in one fused pipeline:
//
//   h' = Abar*h + (K*(B*U) << (48-K_FRACTION_BITS))
//
// This removes the Bbar and Ah scratch memories and the B/A/X phase FSM while
// preserving the prior state-lane scaling and the exact C reduction order.
module ssm_scaled_fused_core #(
`ifdef MAMBA_D_PATH_D1
    parameter integer D_ENABLE = 1,
`else
    parameter integer D_ENABLE = 0,
`endif
    parameter integer D_CHANNEL_OFFSET = 0,
    parameter integer BLOCK_ID = 1,
    parameter integer IS_SPE = 0,
    parameter integer CHANNEL_COUNT = 64,
    parameter integer RECORD_COUNT = 64,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 u_valid,
    input  wire [9:0]           u_record_addr,
    input  wire [5:0]           u_channel_base,
    input  wire [63:0]          u_data,
    input  wire                 dt_valid,
    input  wire [9:0]           dt_record_addr,
    input  wire [5:0]           dt_channel_base,
    input  wire [63:0]          dt_data,
    input  wire                 b_valid,
    input  wire [9:0]           b_record_addr,
    input  wire [127:0]         b_data,
    input  wire                 c_valid,
    input  wire [9:0]           c_record_addr,
    input  wire [127:0]         c_data,
    input  wire                 cfg_lut_we,
    input  wire [7:0]           cfg_lut_addr,
    input  wire [418:0]         cfg_lut_data,
    input  wire                 frame_start,
    input  wire                 compute_start,
    output reg                  out_valid,
    output reg [9:0]            out_record_addr,
    output reg [5:0]            out_channel_base,
    output reg [191:0]          out_y_q24,
    output reg                  done,
    output wire                 busy
);
    localparam integer STATE_COUNT = 16;
    localparam integer ACTIVE_K_FRACTION_BITS = (D_ENABLE != 0)
        ? ((BLOCK_ID == 1) ? ((IS_SPE != 0) ? 24 : 23)
                           : ((IS_SPE != 0) ? 23 : 22))
        : K_FRACTION_BITS;
    localparam integer STATE_LANES = (BLOCK_ID == 1) ? 16 : 8;
    localparam integer STATE_BANKS = STATE_LANES / 4;
    localparam integer STATE_CHUNKS = STATE_COUNT / STATE_LANES;
    localparam integer GROUP_COUNT = CHANNEL_COUNT * STATE_CHUNKS;
    localparam integer GROUP_ADDR_WIDTH
        = (GROUP_COUNT <= 16) ? 4 : (GROUP_COUNT <= 64) ? 6 : 8;
    localparam integer RECORD_ADDR_WIDTH
        = (RECORD_COUNT <= 256) ? 8 : 10;
    localparam integer RECORD_INDEX_BITS = $clog2(RECORD_COUNT);
    localparam integer CHANNEL_BEATS = CHANNEL_COUNT / 8;
    localparam integer LAST_CHANNEL_BASE = CHANNEL_COUNT - 8;
    localparam [0:0] ST_IDLE = 1'b0;
    localparam [0:0] ST_RUN  = 1'b1;

`ifndef SYNTHESIS
    initial begin
        if (!(((BLOCK_ID == 1) || (BLOCK_ID == 2))
              && ((IS_SPE == 0) || (IS_SPE == 1))
              && ((CHANNEL_COUNT == 64) || (CHANNEL_COUNT == 16)
                  || ((BLOCK_ID == 1) && (IS_SPE == 1)
                      && (CHANNEL_COUNT == 8)))
              && ((RESET_PERIOD == 0) || (RESET_PERIOD == 4)))) begin
            $display("Scaled fused SSM instantiated with unsupported geometry");
            $finish;
        end
    end
`endif

    reg fsm_state;
    reg frame_active;
    // Absolute sequence is retained across local EOF.  The physical memories
    // are circular and use only the modulo-record address below.
    // 858 UP tiles require 219648 records in Block1 Spe. Absolute counters
    // must not wrap at 65536; physical addresses still use only low bits.
    reg [31:0] current_sequence;
    wire [RECORD_ADDR_WIDTH-1:0] current_record =
        {{(RECORD_ADDR_WIDTH-RECORD_INDEX_BITS){1'b0}},
          current_sequence[RECORD_INDEX_BITS-1:0]};
    reg [GROUP_ADDR_WIDTH-1:0] request_group;
    wire request_fire = (fsm_state == ST_RUN);

    // Monotonic producer-prefix watermarks replace per-record ready flops.
    reg [31:0] u_records_ready;
    reg [31:0] dt_records_ready;
    reg [31:0] b_records_ready;
    reg [31:0] c_records_ready;
    wire u_record_complete = u_valid
                           && (u_channel_base == LAST_CHANNEL_BASE);
    wire dt_record_complete = dt_valid
                            && (dt_channel_base == LAST_CHANNEL_BASE);
    wire [31:0] current_record_extended = current_sequence;
    wire [31:0] next_record_extended = current_sequence + 32'd1;
    wire current_record_ready
        = (u_records_ready  > current_record_extended)
       && (dt_records_ready > current_record_extended)
       && (b_records_ready  > current_record_extended)
       && (c_records_ready  > current_record_extended);
    wire next_record_ready
        = (u_records_ready  > next_record_extended)
       && (dt_records_ready > next_record_extended)
       && (b_records_ready  > next_record_extended)
       && (c_records_ready  > next_record_extended);
    wire record_zero_ready
        = (u_records_ready != 0) && (dt_records_ready != 0)
       && (b_records_ready != 0) && (c_records_ready != 0);

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            u_records_ready <= 0;
            dt_records_ready <= 0;
            b_records_ready <= 0;
            c_records_ready <= 0;
        end else begin
            if (u_record_complete)
                u_records_ready <= u_records_ready + 32'd1;
            if (dt_record_complete)
                dt_records_ready <= dt_records_ready + 32'd1;
            if (b_valid)
                b_records_ready <= b_records_ready + 32'd1;
            if (c_valid)
                c_records_ready <= c_records_ready + 32'd1;
        end
    end

    // Runtime B/C records and eight-channel U/DT beats.
    wire [127:0] b_record_word;
    wire [127:0] c_record_word;
    wire [10:0] u_write_addr
        = u_record_addr*CHANNEL_BEATS + (u_channel_base >> 3);
    wire [10:0] dt_write_addr
        = dt_record_addr*CHANNEL_BEATS + (dt_channel_base >> 3);
    wire [6:0] request_channel = request_group / STATE_CHUNKS;
    wire [10:0] runtime_input_read_addr
        = current_record*CHANNEL_BEATS + (request_channel >> 3);
    wire [63:0] u_read_word;
    wire [63:0] dt_read_word;

    ssm_bc_record_bram #(
        .DEPTH((RECORD_COUNT <= 256) ? 256 : 1024),
        .ADDR_WIDTH(RECORD_ADDR_WIDTH)
    ) u_b_record_bram (
        .clk(clk), .wr_en(b_valid),
        .wr_addr(b_record_addr[RECORD_ADDR_WIDTH-1:0]),
        .wr_data(b_data), .rd_en(request_fire),
        .rd_addr(current_record), .rd_data(b_record_word));
    ssm_bc_record_bram #(
        .DEPTH((RECORD_COUNT <= 256) ? 256 : 1024),
        .ADDR_WIDTH(RECORD_ADDR_WIDTH)
    ) u_c_record_bram (
        .clk(clk), .wr_en(c_valid),
        .wr_addr(c_record_addr[RECORD_ADDR_WIDTH-1:0]),
        .wr_data(c_data), .rd_en(request_fire),
        .rd_addr(current_record), .rd_data(c_record_word));
    ssm_udt_beat_bram u_u_bram (
        .clk(clk), .wr_en(u_valid), .wr_addr(u_write_addr),
        .wr_data(u_data), .rd_en(request_fire),
        .rd_addr(runtime_input_read_addr), .rd_data(u_read_word));
    ssm_udt_beat_bram u_dt_bram (
        .clk(clk), .wr_en(dt_valid), .wr_addr(dt_write_addr),
        .wr_data(dt_data), .rd_en(request_fire),
        .rd_addr(runtime_input_read_addr), .rd_data(dt_read_word));

    // R0 tags the runtime-memory request.  R1 holds its operands while the
    // one-read nonlinear ROM and state scratch complete their reads.
    reg raw_r0_valid;
    reg [RECORD_ADDR_WIDTH-1:0] raw_r0_record;
    reg [GROUP_ADDR_WIDTH-1:0] raw_r0_group;
    reg r1_valid;
    reg [RECORD_ADDR_WIDTH-1:0] r1_record;
    reg [GROUP_ADDR_WIDTH-1:0] r1_group;
    reg signed [7:0] r1_u_int8;
    reg [127:0] r1_b_record;
    reg [127:0] r1_c_record;
    (* keep = "true", max_fanout = 32 *) reg local_issue_valid;
    (* keep = "true", max_fanout = 32 *) reg local_reset_state;

    wire [6:0] raw_r0_channel = raw_r0_group / STATE_CHUNKS;
    wire [7:0] raw_selected_u
        = u_read_word[(raw_r0_channel % 8)*8 +: 8];
    wire [7:0] selected_dt
        = dt_read_word[(raw_r0_channel % 8)*8 +: 8];

    function [7:0] int8_lut_address;
        input [7:0] signed_code;
        begin
            int8_lut_address = {~signed_code[7], signed_code[6:0]};
        end
    endfunction


    // N0/N1/N2 online coefficient latency adapter. N3 remains the original
    // one-register D1 ROM. DEPTH is L_online - L_original, not L_online.
    localparam integer NF_DELAY = (`NF_METHOD == 0) ? 58 :
                                  (`NF_METHOD == 1) ? 11 :
                                  (`NF_METHOD == 2) ? 12 : 0;
    wire r0_valid, nf_pending;
    wire [RECORD_ADDR_WIDTH-1:0] r0_record;
    wire [GROUP_ADDR_WIDTH-1:0] r0_group;
    wire [7:0] selected_u;
    wire [127:0] nf_b_record_word, nf_c_record_word;
    nf_align #(.WIDTH(RECORD_ADDR_WIDTH+GROUP_ADDR_WIDTH+8+256), .DEPTH(NF_DELAY)) u_nf_align (
        .clk(clk), .reset(!rst_n || frame_start),
        .in_valid(raw_r0_valid), .out_valid(r0_valid), .pending(nf_pending),
        .in_data({raw_r0_record, raw_r0_group, raw_selected_u, b_record_word, c_record_word}),
        .out_data({r0_record, r0_group, selected_u, nf_b_record_word, nf_c_record_word})
    );
    wire [6:0] r0_channel = r0_group / STATE_CHUNKS;

    wire [418:0] lut_word;
    generate if (D_ENABLE != 0) begin : G_D1_NONLINEAR
      if (`NF_METHOD == 3) begin : G_NATIVE_ROM
        mamba_d1_nonlinear_rom_1r #(.BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE))
        u_rom (.clk(clk), .en(1'b1),
            .addr(int8_lut_address(selected_dt)), .data(lut_word));

      end else begin : G_ONLINE
        wire nf_coeff_valid;
        nf_coeff_lanes #(.METHOD(`NF_METHOD), .BLOCK_ID(BLOCK_ID),
            .IS_SPE(IS_SPE), .LANES(1)) u_online (
            .clk(clk), .reset(!rst_n || frame_start),
            .in_valid(raw_r0_valid), .q_dt(selected_dt),
            .out_valid(nf_coeff_valid), .data0(lut_word));
`ifndef SYNTHESIS
        always @(posedge clk) if (rst_n && !frame_start &&
            (nf_coeff_valid !== r1_valid))
            $fatal(1, "NL coefficient/metadata valid misalignment B%0d Spe%0d", BLOCK_ID, IS_SPE);
`endif
      end
    end else begin : G_LEGACY_NONLINEAR
    mamba_ssm_lut_rom_1r #(.BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE))
    u_nonlinear_lut_rom (
        // The returned ROM word is consumed only with r1_valid.  A permanent
        // read enable removes r0_valid from the distributed BRAM enable tree.
        .clk(clk), .en(1'b1),
        .addr(int8_lut_address(selected_dt)), .data(lut_word));
    end endgenerate

    // Seventeen edges from selected_u (R0->R1) through Y_L4.  Delay only
    // a compact LUT address; the registered product is generated at the end.
    wire signed [47:0] d_readout;
    generate if (D_ENABLE != 0) begin : G_D_READOUT
        wire [5:0] d_channel = r0_channel + D_CHANNEL_OFFSET;
        mamba_d_path_lut #(
            .BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE), .LATENCY(17)
        ) u_d_lut (.clk(clk), .channel(d_channel), .u_s8(selected_u),
            .d_s48(d_readout));
    end else begin : G_NO_D_READOUT
        assign d_readout = 48'sd0;
    end endgenerate

    // Only state survives between groups.  Bbar and Ah scratch memories are
    // eliminated by the fused state-cell pipeline.
    wire [511:0] state_group_read;
    wire [511:0] state_group_write;
    wire state_group_write_en;
    wire [GROUP_ADDR_WIDTH-1:0] state_group_write_addr;
    // State storage is owned by the four-state arithmetic banks below.  Keep
    // the old aggregate ports unloaded so neither Block1 Spa nor Block2
    // creates a 512/1024-bit cross-core route.
    assign state_group_read = 512'd0;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            raw_r0_valid <= 1'b0;
            r1_valid <= 1'b0;
            local_issue_valid <= 1'b0;
        end else begin
            raw_r0_valid <= request_fire;
            r1_valid <= r0_valid;
            local_issue_valid <= r0_valid;
        end

        // Payload and metadata run continuously.  r0/r1/local_issue_valid are
        // the only validity contract, so invalid-cycle contents are don't-care.
        // Keeping these registers out of request_fire/frame_start branches
        // prevents those controls from becoming wide, distributed CEs.
        raw_r0_record <= current_record;
        raw_r0_group <= request_group;
        r1_record <= r0_record;
        r1_group <= r0_group;
        r1_u_int8 <= $signed(selected_u);
        r1_b_record <= nf_b_record_word;
        r1_c_record <= nf_c_record_word;
        local_reset_state <= (RESET_PERIOD == 4)
            ? (r0_record[1:0] == 0) : (r0_record == 0);
    end

    // Metadata matching the fixed five-edge ssm_update_3dsp result.
    reg arith_valid_d0;
    reg arith_valid_d1;
    reg arith_valid_d2;
    reg arith_valid_d3;
    reg arith_valid_d4;
    reg [RECORD_ADDR_WIDTH-1:0] arith_record_d0;
    reg [RECORD_ADDR_WIDTH-1:0] arith_record_d1;
    reg [RECORD_ADDR_WIDTH-1:0] arith_record_d2;
    reg [RECORD_ADDR_WIDTH-1:0] arith_record_d3;
    reg [RECORD_ADDR_WIDTH-1:0] arith_record_d4;
    reg [GROUP_ADDR_WIDTH-1:0] arith_group_d0;
    reg [GROUP_ADDR_WIDTH-1:0] arith_group_d1;
    reg [GROUP_ADDR_WIDTH-1:0] arith_group_d2;
    reg [GROUP_ADDR_WIDTH-1:0] arith_group_d3;
    reg [GROUP_ADDR_WIDTH-1:0] arith_group_d4;
    reg [127:0] arith_c_record_d0;
    reg [127:0] arith_c_record_d1;
    reg [127:0] arith_c_record_d2;
    reg [127:0] arith_c_record_d3;
    reg [127:0] arith_c_record_d4;

    wire [2:0] issue_state_chunk = r1_group % STATE_CHUNKS;
    wire [4:0] issue_state_base = issue_state_chunk * STATE_LANES;
    // Local registered control endpoint for this state bank.  In the Block1
    // Spe dual wrapper each half now owns its valid/reset driver instead of
    // sharing a frame-wide high-fanout control tree.
    wire [18:0] lut_k = lut_word[418:400];
    wire signed [64:0] state_accumulator [0:STATE_LANES-1];
    wire signed [31:0] state_next_comb [0:STATE_LANES-1];
    reg  signed [31:0] state_next_q [0:STATE_LANES-1];
    wire signed [39:0] c_product [0:15];
    reg state_round_valid;
    reg [RECORD_ADDR_WIDTH-1:0] state_round_record;
    reg [GROUP_ADDR_WIDTH-1:0] state_round_group;
    reg [127:0] state_round_c_record;
    // state_next_q and the bank-local C payload are captured on the same
    // edge.  Before that edge, arith_group_d4 is the tag of the returning
    // 3-DSP result, whereas state_round_group still contains the preceding
    // group's tag.  The distinction is invisible in Block1 (one state
    // chunk), but swaps Block2's lower/upper eight C coefficients.
    wire [2:0] arith_state_chunk_d4
        = arith_group_d4 % STATE_CHUNKS;
    wire [4:0] arith_state_base_d4
        = arith_state_chunk_d4 * STATE_LANES;
    wire signed [41:0] block1_spe_bank_c_sum [0:3];
    integer state_capture_index;

    function signed [31:0] round_shift_to_int32;
        input signed [64:0] value;
        input integer shift_value;
        reg signed [65:0] magnitude;
        reg signed [65:0] rounded;
        reg signed [65:0] signed_rounded;
        begin
            if (value < 0)
                magnitude = -$signed({value[64], value});
            else
                magnitude = $signed({1'b0, value});
            if (shift_value > 0)
                rounded = (magnitude + (66'sd1 <<< (shift_value-1)))
                        >>> shift_value;
            else
                rounded = magnitude <<< (-shift_value);
            signed_rounded = (value < 0) ? -rounded : rounded;
            if (signed_rounded[65:31] == {35{signed_rounded[31]}})
                round_shift_to_int32 = signed_rounded[31:0];
            else if (signed_rounded[65])
                round_shift_to_int32 = 32'sh80000000;
            else
                round_shift_to_int32 = 32'sh7fffffff;
        end
    endfunction

    genvar lane;
    generate
        if ((BLOCK_ID == 1) || (BLOCK_ID == 2)) begin : G_SCALED_BANKED
            genvar state_bank;
            assign state_group_write = 512'd0;
            for (state_bank=0; state_bank<STATE_BANKS;
                 state_bank=state_bank+1)
                begin : G_STATE_BANK
                (* keep_hierarchy = "yes" *)
                ssm_block1_spe_state_bank4 #(
                    .BLOCK_ID(BLOCK_ID),
                    .GROUP_COUNT(GROUP_COUNT),
                    .GROUP_ADDR_WIDTH(GROUP_ADDR_WIDTH),
                    .STATE_BASE(state_bank*4),
                    .K_FRACTION_BITS(ACTIVE_K_FRACTION_BITS)
                ) u_local_bank (
                    .clk(clk), .issue_valid(local_issue_valid),
                    .reset_state(local_reset_state),
                    .read_group(r0_group),
                    // state_next_q is captured from the 3-DSP result on the
                    // arith_valid_d4 edge.  Write it one clock later, using
                    // the matching rounded-state metadata.  The old d4
                    // qualification wrote the previous group's payload and
                    // was hidden for record 0 by the all-zero initial state.
                    .state_write_valid(state_round_valid),
                    .state_write_group(state_round_group),
                    .abar_word(lut_word[
                        (issue_state_base+state_bank*4)*25 +: 100]),
                    .k_word(lut_k),
                    .b_states(r1_b_record[
                        (issue_state_base+state_bank*4)*8 +: 32]),
                    .u_s8(r1_u_int8),
                    .c_states(arith_c_record_d4[
                        (arith_state_base_d4+state_bank*4)*8 +: 32]),
                    .c_sum_s42(block1_spe_bank_c_sum[state_bank]));
            end
            if (STATE_BANKS == 2) begin : G_UNUSED_BANK_SUMS
                assign block1_spe_bank_c_sum[2] = 42'sd0;
                assign block1_spe_bank_c_sum[3] = 42'sd0;
            end
            for (lane=0; lane<STATE_LANES; lane=lane+1)
                begin : G_UNUSED_STATE_ARITH
                assign c_product[lane] = 40'sd0;
                assign state_accumulator[lane] = 65'sd0;
                assign state_next_comb[lane] = 32'sd0;
            end
            if (STATE_LANES == 8) begin : G_UNUSED_C_UPPER
                for (lane=8; lane<16; lane=lane+1)
                    begin : G_ZERO_C
                    assign c_product[lane] = 40'sd0;
                end
            end
        end else begin : G_CENTRAL_UPDATE
            assign block1_spe_bank_c_sum[0] = 42'sd0;
            assign block1_spe_bank_c_sum[1] = 42'sd0;
            assign block1_spe_bank_c_sum[2] = 42'sd0;
            assign block1_spe_bank_c_sum[3] = 42'sd0;
            for (lane = 0; lane < STATE_LANES; lane = lane + 1) begin : G_UPDATE
                wire [4:0] state_index = issue_state_base + lane;
                wire signed [31:0] old_state
                    = local_reset_state ? 32'sd0
                    : $signed(state_group_read[lane*32 +: 32]);
                wire signed [7:0] lane_b
                    = $signed(r1_b_record[state_index*8 +: 8]);

                ssm_update_3dsp #(.K_FRACTION_BITS(ACTIVE_K_FRACTION_BITS))
                u_fused_update (
                    .CLK(clk), .VALID(local_issue_valid),
                    .ABAR_U25(lut_word[state_index*25 +: 25]),
                    .STATE_S32(old_state), .K_U19(lut_k),
                    .B_S8(lane_b), .U_S8(r1_u_int8),
                    .STATE_ACC_Q48(state_accumulator[lane]));

                assign state_next_comb[lane]
                    = round_shift_to_int32(state_accumulator[lane], 24);
                assign state_group_write[lane*32 +: 32]
                    = state_next_q[lane];

                (* use_dsp = "no" *)
                wire signed [39:0] c_product_internal
                    = state_next_q[lane]
                    * $signed(state_round_c_record[
                        (state_round_group % STATE_CHUNKS)*STATE_LANES*8
                        + lane*8 +: 8]);
                assign c_product[lane] = c_product_internal;
            end
            if (STATE_LANES == 8) begin : G_PAD_STATE8
                assign state_group_write[511:256] = 256'd0;
                for (lane = 8; lane < 16; lane = lane + 1)
                    begin : G_ZERO_C
                    assign c_product[lane] = 40'sd0;
                end
            end
        end
    endgenerate

    assign state_group_write_en = state_round_valid;
    assign state_group_write_addr = state_round_group;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            arith_valid_d0 <= 1'b0;
            arith_valid_d1 <= 1'b0;
            arith_valid_d2 <= 1'b0;
            arith_valid_d3 <= 1'b0;
            arith_valid_d4 <= 1'b0;
            state_round_valid <= 1'b0;
        end else begin
            arith_valid_d0 <= local_issue_valid;
            arith_valid_d1 <= arith_valid_d0;
            arith_valid_d2 <= arith_valid_d1;
            arith_valid_d3 <= arith_valid_d2;
            arith_valid_d4 <= arith_valid_d3;
            state_round_valid <= arith_valid_d4;
        end


        // The validity pipeline above is frame-cleared; all matching metadata
        // and arithmetic payload advance every clock.  State RAM writes remain
        // qualified by state_round_valid, so invalid values never become
        // architectural state while the high-fanout valid/frame_start nets no
        // longer drive CE pins on the wide payload banks or C multipliers.
        arith_record_d0 <= r1_record;
        arith_group_d0 <= r1_group;
        arith_c_record_d0 <= r1_c_record;
        arith_record_d1 <= arith_record_d0;
        arith_group_d1 <= arith_group_d0;
        arith_c_record_d1 <= arith_c_record_d0;
        arith_record_d2 <= arith_record_d1;
        arith_group_d2 <= arith_group_d1;
        arith_c_record_d2 <= arith_c_record_d1;
        arith_record_d3 <= arith_record_d2;
        arith_group_d3 <= arith_group_d2;
        arith_c_record_d3 <= arith_c_record_d2;
        arith_record_d4 <= arith_record_d3;
        arith_group_d4 <= arith_group_d3;
        arith_c_record_d4 <= arith_c_record_d3;
        state_round_record <= arith_record_d4;
        state_round_group <= arith_group_d4;
        state_round_c_record <= arith_c_record_d4;
        for (state_capture_index = 0;
             state_capture_index < STATE_LANES;
             state_capture_index = state_capture_index + 1)
            state_next_q[state_capture_index]
                <= state_next_comb[state_capture_index];
    end

    // Sixteen-input reduction tree.  Block2's unused upper eight products are
    // zero, so ordering and final Q24 width match the software reference.
    reg signed [39:0] c_product_pipe [0:15];
    reg signed [40:0] y_level1 [0:7];
    reg signed [41:0] y_level2 [0:3];
    reg signed [42:0] y_level3 [0:1];
    reg signed [43:0] y_level4;
    reg y_product_valid;
    reg y_valid_l1;
    reg y_valid_l2;
    reg y_valid_l3;
    reg y_valid_l4;
    reg c_dsp_valid_d0;
    reg c_dsp_valid_d1;
    reg c_dsp_valid_d2;
    reg c_dsp_valid_d3;
    reg c_dsp_valid_d4;
    reg [RECORD_ADDR_WIDTH-1:0] c_dsp_record_d0;
    reg [RECORD_ADDR_WIDTH-1:0] c_dsp_record_d1;
    reg [RECORD_ADDR_WIDTH-1:0] c_dsp_record_d2;
    reg [RECORD_ADDR_WIDTH-1:0] c_dsp_record_d3;
    reg [RECORD_ADDR_WIDTH-1:0] c_dsp_record_d4;
    reg [GROUP_ADDR_WIDTH-1:0] c_dsp_group_d0;
    reg [GROUP_ADDR_WIDTH-1:0] c_dsp_group_d1;
    reg [GROUP_ADDR_WIDTH-1:0] c_dsp_group_d2;
    reg [GROUP_ADDR_WIDTH-1:0] c_dsp_group_d3;
    reg [GROUP_ADDR_WIDTH-1:0] c_dsp_group_d4;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_d0;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l1;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l2;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l3;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l4;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_d0;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l1;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l2;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l3;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l4;
    reg signed [42:0] block1_spe_bank_level1 [0:1];
    reg signed [43:0] block1_spe_bank_level2;
    integer reduce_index;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            y_product_valid <= 1'b0;
            y_valid_l1 <= 1'b0;
            y_valid_l2 <= 1'b0;
            y_valid_l3 <= 1'b0;
            y_valid_l4 <= 1'b0;
            c_dsp_valid_d0 <= 1'b0;
            c_dsp_valid_d1 <= 1'b0;
            c_dsp_valid_d2 <= 1'b0;
            c_dsp_valid_d3 <= 1'b0;
            c_dsp_valid_d4 <= 1'b0;
        end else begin
            c_dsp_valid_d0 <= state_round_valid;
            c_dsp_valid_d1 <= c_dsp_valid_d0;
            c_dsp_valid_d2 <= c_dsp_valid_d1;
            c_dsp_valid_d3 <= c_dsp_valid_d2;
            c_dsp_valid_d4 <= c_dsp_valid_d3;
            // Every scaled branch now uses the local five-cycle C-product
            // DSP inside its four-state banks.
            y_product_valid <= c_dsp_valid_d4;
            y_valid_l1 <= y_product_valid;
            y_valid_l2 <= y_valid_l1;
            y_valid_l3 <= y_valid_l2;
            y_valid_l4 <= y_valid_l3;
        end

        // Five metadata clocks mirror the bank-local C-product DSP.
        c_dsp_record_d0 <= state_round_record;
        c_dsp_record_d1 <= c_dsp_record_d0;
        c_dsp_record_d2 <= c_dsp_record_d1;
        c_dsp_record_d3 <= c_dsp_record_d2;
        c_dsp_record_d4 <= c_dsp_record_d3;
        c_dsp_group_d0 <= state_round_group;
        c_dsp_group_d1 <= c_dsp_group_d0;
        c_dsp_group_d2 <= c_dsp_group_d1;
        c_dsp_group_d3 <= c_dsp_group_d2;
        c_dsp_group_d4 <= c_dsp_group_d3;

        // C products and reduction metadata are free-running payload.  Only
        // the valid pipeline defines which cycle is meaningful.
        y_record_d0 <= c_dsp_record_d4;
        y_group_d0 <= c_dsp_group_d4;
        for (reduce_index = 0; reduce_index < 16;
             reduce_index = reduce_index + 1)
            c_product_pipe[reduce_index] <= c_product[reduce_index];

        y_record_l1 <= y_record_d0;
        y_group_l1 <= y_group_d0;
        for (reduce_index = 0; reduce_index < 8;
             reduce_index = reduce_index + 1)
            y_level1[reduce_index]
                <= $signed({c_product_pipe[reduce_index*2][39],
                            c_product_pipe[reduce_index*2]})
                 + $signed({c_product_pipe[reduce_index*2+1][39],
                            c_product_pipe[reduce_index*2+1]});

        y_record_l2 <= y_record_l1;
        y_group_l2 <= y_group_l1;
        for (reduce_index = 0; reduce_index < 4;
             reduce_index = reduce_index + 1)
            y_level2[reduce_index]
                <= $signed({y_level1[reduce_index*2][40],
                            y_level1[reduce_index*2]})
                 + $signed({y_level1[reduce_index*2+1][40],
                            y_level1[reduce_index*2+1]});

        y_record_l3 <= y_record_l2;
        y_group_l3 <= y_group_l2;
        y_level3[0]
            <= $signed({y_level2[0][41], y_level2[0]})
             + $signed({y_level2[1][41], y_level2[1]});
        y_level3[1]
            <= $signed({y_level2[2][41], y_level2[2]})
             + $signed({y_level2[3][41], y_level2[3]});

        y_record_l4 <= y_record_l3;
        y_group_l4 <= y_group_l3;
        y_level4 <= $signed({y_level3[0][42], y_level3[0]})
                  + $signed({y_level3[1][42], y_level3[1]});

        // The four physical state banks perform product capture and their
        // local 4-to-1 reductions.  These two parent stages combine only four
        // narrow partial sums, preserving the original sixteen-term order.
        block1_spe_bank_level1[0]
            <= $signed({block1_spe_bank_c_sum[0][41],
                        block1_spe_bank_c_sum[0]})
             + $signed({block1_spe_bank_c_sum[1][41],
                        block1_spe_bank_c_sum[1]});
        block1_spe_bank_level1[1]
            <= $signed({block1_spe_bank_c_sum[2][41],
                        block1_spe_bank_c_sum[2]})
             + $signed({block1_spe_bank_c_sum[3][41],
                        block1_spe_bank_c_sum[3]});
        block1_spe_bank_level2
            <= $signed({block1_spe_bank_level1[0][42],
                        block1_spe_bank_level1[0]})
             + $signed({block1_spe_bank_level1[1][42],
                        block1_spe_bank_level1[1]});
    end

    wire [6:0] output_channel = y_group_l4 / STATE_CHUNKS;
    wire [2:0] output_state_chunk = y_group_l4 % STATE_CHUNKS;
    wire signed [43:0] selected_y_level4 = block1_spe_bank_level2;
    wire signed [47:0] partial_y
        = {{4{selected_y_level4[43]}}, selected_y_level4};
    reg signed [47:0] state_y_accumulator;
    reg [143:0] output_pack;
    wire signed [47:0] complete_y
        = (output_state_chunk == 0)
        ? (partial_y + d_readout) : (state_y_accumulator + partial_y);

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            out_valid <= 1'b0;
            done <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            done <= 1'b0;
            if (y_valid_l4) begin
                if (output_state_chunk == 0)
                    // D belongs to the channel: seed with C_low+D, then
                    // add only C_high at chunk1. No extra cycle/adder chain.
                    state_y_accumulator <= partial_y + d_readout;
                else
                    state_y_accumulator
                        <= state_y_accumulator + partial_y;

                if (output_state_chunk == STATE_CHUNKS-1) begin
                    if (output_channel[1:0] == 3) begin
                        out_valid <= 1'b1;
                        out_record_addr <= y_record_l4;
                        out_channel_base
                            <= {output_channel[5:2], 2'b00};
                        out_y_q24 <= {complete_y, output_pack};
                        if ((y_record_l4 == RECORD_COUNT-1)
                            && (output_channel == CHANNEL_COUNT-1))
                            done <= 1'b1;
                    end else begin
                        output_pack[output_channel[1:0]*48 +: 48]
                            <= complete_y;
                    end
                end
            end
        end
    end

    assign busy = frame_active || (fsm_state != ST_IDLE)
               || raw_r0_valid || nf_pending || r0_valid || r1_valid
               || arith_valid_d0 || arith_valid_d1 || arith_valid_d2
               || arith_valid_d3 || arith_valid_d4
               || state_round_valid
               || c_dsp_valid_d0 || c_dsp_valid_d1 || c_dsp_valid_d2
               || c_dsp_valid_d3 || c_dsp_valid_d4
               || y_product_valid || y_valid_l1 || y_valid_l2
               || y_valid_l3 || y_valid_l4 || out_valid;

    // One useful group every clock; ready records cross the boundary without
    // phase drain or PRE bubbles.
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            fsm_state <= ST_IDLE;
            frame_active <= rst_n && frame_start;
            current_sequence <= 0;
            request_group <= 0;
        end else begin
            if (compute_start && !frame_active && (fsm_state == ST_IDLE)) begin
                frame_active <= 1'b1;
                current_sequence <= 0;
                request_group <= 0;
                if (record_zero_ready)
                    fsm_state <= ST_RUN;
            end else begin
                case (fsm_state)
                    ST_IDLE: begin
                        request_group <= 0;
                        if (frame_active && current_record_ready)
                            fsm_state <= ST_RUN;
                    end
                    ST_RUN: begin
                        if (request_group == GROUP_COUNT-1) begin
                            request_group <= 0;
                            current_sequence <= current_sequence + 32'd1;
                            if (!next_record_ready)
                                fsm_state <= ST_IDLE;
                        end else begin
                            request_group <= request_group + 1'b1;
                        end
                    end
                    default: fsm_state <= ST_IDLE;
                endcase
            end
        end
    end

`ifndef SYNTHESIS
    integer sim_issue_groups;
    integer sim_continuous_record_transitions;
    integer sim_state_reset_records;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            sim_issue_groups <= 0;
            sim_continuous_record_transitions <= 0;
            sim_state_reset_records <= 0;
        end else if (request_fire) begin
            sim_issue_groups <= sim_issue_groups + 1;
            if ((request_group == 0)
                && (((RESET_PERIOD == 4)
                     && (current_record[1:0] == 0))
                    || ((RESET_PERIOD != 4) && (current_record == 0))))
                sim_state_reset_records <= sim_state_reset_records + 1;
            if ((request_group == GROUP_COUNT-1)
                && (current_record != RECORD_COUNT-1)
                && next_record_ready)
                sim_continuous_record_transitions
                    <= sim_continuous_record_transitions + 1;
        end

        if (request_fire
            && ((u_valid
                 && (u_record_addr[RECORD_ADDR_WIDTH-1:0]
                     == current_record))
                || (dt_valid
                    && (dt_record_addr[RECORD_ADDR_WIDTH-1:0]
                        == current_record))
                || (b_valid
                    && (b_record_addr[RECORD_ADDR_WIDTH-1:0]
                        == current_record))
                || (c_valid
                    && (c_record_addr[RECORD_ADDR_WIDTH-1:0]
                        == current_record))))
            $error("Scaled fused SSM read/write collision on active record");
        if (frame_active && cfg_lut_we)
            $error("Scaled fused SSM LUT write during active frame");
    end
`endif

    wire unused_cfg = cfg_lut_we ^ cfg_lut_addr[0] ^ cfg_lut_data[0];
endmodule

// Block1 Spe throughput specialization.  Spectral channels are independent
// state machines sharing the same B/C record, so split channels 0..7 and
// 8..15 across two exact fused cores.  Both halves run one channel/clock.
// Input beats reach the low and high halves on adjacent clocks, so outputs
// are normally staggered by one clock.  A one-entry buffer also handles the
// legal same-clock case without dropping either four-channel result.
module ssm_block1_spe_dual_fused_core #(
    parameter integer RECORD_COUNT = 256,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 4
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 u_valid,
    input  wire [9:0]           u_record_addr,
    input  wire [5:0]           u_channel_base,
    input  wire [63:0]          u_data,
    input  wire                 dt_valid,
    input  wire [9:0]           dt_record_addr,
    input  wire [5:0]           dt_channel_base,
    input  wire [63:0]          dt_data,
    input  wire                 b_valid,
    input  wire [9:0]           b_record_addr,
    input  wire [127:0]         b_data,
    input  wire                 c_valid,
    input  wire [9:0]           c_record_addr,
    input  wire [127:0]         c_data,
    input  wire                 cfg_lut_we,
    input  wire [7:0]           cfg_lut_addr,
    input  wire [418:0]         cfg_lut_data,
    input  wire                 frame_start,
    input  wire                 compute_start,
    output reg                  out_valid,
    output reg [9:0]            out_record_addr,
    output reg [5:0]            out_channel_base,
    output reg [191:0]          out_y_q24,
    output reg                  done,
    output wire                 busy
);
    wire lower_u_valid = u_valid && (u_channel_base == 0);
    wire upper_u_valid = u_valid && (u_channel_base == 8);
    wire lower_dt_valid = dt_valid && (dt_channel_base == 0);
    wire upper_dt_valid = dt_valid && (dt_channel_base == 8);

    wire lower_out_valid;
    wire [9:0] lower_out_record;
    wire [5:0] lower_out_channel;
    wire [191:0] lower_out_y;
    wire lower_done;
    wire lower_busy;
    wire upper_out_valid;
    wire [9:0] upper_out_record;
    wire [5:0] upper_out_channel;
    wire [191:0] upper_out_y;
    wire upper_done;
    wire upper_busy;
    wire lower_frame_start_local;
    wire upper_frame_start_local;
    ssm_local_control_buffer u_lower_frame_start_buffer (
        .control_in(frame_start), .control_out(lower_frame_start_local));
    ssm_local_control_buffer u_upper_frame_start_buffer (
        .control_in(frame_start), .control_out(upper_frame_start_local));

    ssm_scaled_fused_core #(
        .BLOCK_ID(1), .IS_SPE(1), .CHANNEL_COUNT(8),
        .RECORD_COUNT(RECORD_COUNT),
        .K_FRACTION_BITS(K_FRACTION_BITS),
        .RESET_PERIOD(RESET_PERIOD)
    ) u_lower (
        .clk(clk), .rst_n(rst_n),
        .u_valid(lower_u_valid), .u_record_addr(u_record_addr),
        .u_channel_base(6'd0), .u_data(u_data),
        .dt_valid(lower_dt_valid), .dt_record_addr(dt_record_addr),
        .dt_channel_base(6'd0), .dt_data(dt_data),
        .b_valid(b_valid), .b_record_addr(b_record_addr), .b_data(b_data),
        .c_valid(c_valid), .c_record_addr(c_record_addr), .c_data(c_data),
        .cfg_lut_we(cfg_lut_we), .cfg_lut_addr(cfg_lut_addr),
        .cfg_lut_data(cfg_lut_data), .frame_start(lower_frame_start_local),
        .compute_start(compute_start),
        .out_valid(lower_out_valid), .out_record_addr(lower_out_record),
        .out_channel_base(lower_out_channel), .out_y_q24(lower_out_y),
        .done(lower_done), .busy(lower_busy));

    ssm_scaled_fused_core #(
        .BLOCK_ID(1), .IS_SPE(1), .CHANNEL_COUNT(8),
        .RECORD_COUNT(RECORD_COUNT),
        .K_FRACTION_BITS(K_FRACTION_BITS),
        .RESET_PERIOD(RESET_PERIOD),
        .D_CHANNEL_OFFSET(8)
    ) u_upper (
        .clk(clk), .rst_n(rst_n),
        .u_valid(upper_u_valid), .u_record_addr(u_record_addr),
        .u_channel_base(6'd0), .u_data(u_data),
        .dt_valid(upper_dt_valid), .dt_record_addr(dt_record_addr),
        .dt_channel_base(6'd0), .dt_data(dt_data),
        .b_valid(b_valid), .b_record_addr(b_record_addr), .b_data(b_data),
        .c_valid(c_valid), .c_record_addr(c_record_addr), .c_data(c_data),
        .cfg_lut_we(cfg_lut_we), .cfg_lut_addr(cfg_lut_addr),
        .cfg_lut_data(cfg_lut_data), .frame_start(upper_frame_start_local),
        .compute_start(compute_start),
        .out_valid(upper_out_valid), .out_record_addr(upper_out_record),
        .out_channel_base(upper_out_channel), .out_y_q24(upper_out_y),
        .done(upper_done), .busy(upper_busy));

    reg upper_pending;
    reg [9:0] upper_pending_record;
    reg [5:0] upper_pending_channel;
    reg [191:0] upper_pending_y;
    reg upper_pending_done;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            out_valid <= 1'b0;
            done <= 1'b0;
            upper_pending <= 1'b0;
        end else begin
            out_valid <= 1'b0;
            done <= 1'b0;
            if (upper_pending) begin
                out_valid <= 1'b1;
                out_record_addr <= upper_pending_record;
                out_channel_base <= upper_pending_channel + 6'd8;
                out_y_q24 <= upper_pending_y;
                done <= upper_pending_done;
                upper_pending <= 1'b0;
            end else if (lower_out_valid) begin
                out_valid <= 1'b1;
                out_record_addr <= lower_out_record;
                out_channel_base <= lower_out_channel;
                out_y_q24 <= lower_out_y;
                if (upper_out_valid) begin
                    upper_pending <= 1'b1;
                    upper_pending_record <= upper_out_record;
                    upper_pending_channel <= upper_out_channel;
                    upper_pending_y <= upper_out_y;
                    upper_pending_done <= upper_done;
                end
            end else if (upper_out_valid) begin
                out_valid <= 1'b1;
                out_record_addr <= upper_out_record;
                out_channel_base <= upper_out_channel + 6'd8;
                out_y_q24 <= upper_out_y;
                done <= upper_done;
            end
        end
    end

    assign busy = lower_busy || upper_busy || upper_pending || out_valid;

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !frame_start) begin
            if (upper_pending && (lower_out_valid || upper_out_valid))
                $error("Block1 Spe dual SSM output serializer overflow");
            if (lower_out_valid && upper_out_valid
                && ((lower_out_record != upper_out_record)
                    || (lower_out_channel != upper_out_channel)))
                $error("Block1 Spe dual SSM half metadata mismatch");
        end
    end
`endif
endmodule

`default_nettype wire
