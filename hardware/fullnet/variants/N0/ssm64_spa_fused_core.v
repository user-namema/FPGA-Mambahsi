`timescale 1ns / 1ps
`default_nettype none

// Signed 32x8 C projection used by both Block0 Spa and Spe.  Synthesis
// instantiates one area-optimised, five-cycle Multiplier Generator IP.  RTL
// simulation uses the cycle-accurate model below, so functional tests do not
// depend on the generated XCI being present in sim_1.
module ssm_c_product_32x8_5cyc (
    input  wire                 clk,
    input  wire signed [31:0]   state_s32,
    input  wire signed [7:0]    c_s8,
    output wire signed [39:0]   product_s40
);
`ifdef SYNTHESIS
    ssm_c_mult_32x8_5cyc_area u_ip (
        .CLK(clk), .A(state_s32), .B(c_s8), .P(product_s40));
`else
    reg signed [39:0] product_d0;
    reg signed [39:0] product_d1;
    reg signed [39:0] product_d2;
    reg signed [39:0] product_d3;
    reg signed [39:0] product_d4;
    always @(posedge clk) begin
        product_d0 <= state_s32 * c_s8;
        product_d1 <= product_d0;
        product_d2 <= product_d1;
        product_d3 <= product_d2;
        product_d4 <= product_d3;
    end
    assign product_s40 = product_d4;
`endif
endmodule

// Half of one Block0 channel bank.  Eight state cells, their 256-bit state
// memory, C products and the local 8-to-1 reduction remain inside one physical
// hierarchy.  reset_state_request is captured into eight preserved local
// endpoints on the same edge on which the parent captures issue_valid, so the
// arithmetic latency is unchanged while no reset net drives a 512-bit mux.
module ssm_block0_fused_state_subbank8 #(
    parameter integer GROUP_COUNT = 4,
    parameter integer GROUP_ADDR_WIDTH = 2,
    parameter integer RECORD_ADDR_WIDTH = 8,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 0,
    parameter integer STATE_BASE = 0,
    parameter integer USE_DSP_C_PRODUCT = 1
) (
    input  wire                         clk,
    input  wire                         issue_valid,
    input  wire                         reset_state_request,
    input  wire [RECORD_ADDR_WIDTH-1:0] issue_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  issue_group,
    input  wire                         forward_d3_valid,
    input  wire [RECORD_ADDR_WIDTH-1:0] forward_d3_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  forward_d3_group,
    input  wire                         forward_d4_valid,
    input  wire [RECORD_ADDR_WIDTH-1:0] forward_d4_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  forward_d4_group,
    input  wire [GROUP_ADDR_WIDTH-1:0]  read_group,
    input  wire                         state_write_valid,
    input  wire [GROUP_ADDR_WIDTH-1:0]  state_write_group,
    input  wire [199:0]                 abar_word,
    input  wire [18:0]                  k_word,
    input  wire [63:0]                  b_states,
    input  wire signed [7:0]            u_s8,
    input  wire [63:0]                  c_states,
    output wire signed [42:0]           c_sum_s43
);
    localparam integer STATE_COUNT = 8;

    wire [255:0] state_bank_read;
    wire [255:0] state_bank_write;
    wire signed [64:0] state_accumulator [0:STATE_COUNT-1];
    wire signed [64:0] state_forward_accumulator [0:STATE_COUNT-1];
    wire signed [31:0] state_next_comb [0:STATE_COUNT-1];
    wire signed [31:0] state_forward_next [0:STATE_COUNT-1];
    reg  signed [31:0] state_next_q [0:STATE_COUNT-1];
    reg [63:0] c_states_q;
    wire signed [39:0] c_product [0:STATE_COUNT-1];
    (* keep = "true", max_fanout = 32 *) reg [7:0] reset_state_lane_q;
    // Local U endpoints replace the former parent R1 broadcast.  Two copies
    // per eight-state subbank cap each byte endpoint at four recurrence cells.
    (* keep = "true", max_fanout = 8 *) reg signed [7:0] u_s8_local_lo;
    (* keep = "true", max_fanout = 8 *) reg signed [7:0] u_s8_local_hi;

    // Recompute the two recurrence-forwarding decisions inside each physical
    // channel bank.  Previously one pair of parent comparators drove 2048 LUT
    // inputs across all four banks.  These local endpoints cap the fanout and
    // allow synthesis to replicate only within the owning arithmetic cluster.
    (* keep = "true", max_fanout = 128 *) wire forward_d3_hit_local
        = (RESET_PERIOD == 4) && issue_valid && (issue_record[1:0] != 0)
       && forward_d3_valid && (forward_d3_group == issue_group)
       && (forward_d3_record + 1'b1 == issue_record);
    (* keep = "true", max_fanout = 128 *) wire forward_d4_hit_local
        = (RESET_PERIOD == 4) && issue_valid && (issue_record[1:0] != 0)
       && forward_d4_valid && (forward_d4_group == issue_group)
       && (forward_d4_record + 1'b1 == issue_record);

    // K_FRACTION_BITS is fixed at 24 for the deployed Block0 core.  Avoid the
    // old generic abs/add/variable-shift/reapply-sign network: arithmetic
    // right shift gives floor(value/2^24), then one narrow increment implements
    // round-to-nearest with ties away from zero.  This is bit-exact with the
    // former function, including negative half-way cases and saturation.
    function signed [31:0] round_shift24_to_int32;
        input signed [64:0] value;
        reg round_up;
        reg signed [41:0] shifted_rounded;
        begin
            // Positive values increment on remainder >= 0x800000.  For a
            // negative floor quotient, increment toward zero only when the
            // remainder is strictly greater than half; an exact half stays
            // at the more-negative quotient (tie away from zero).
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

    ssm_block0_state_channel_bank #(
        .WIDTH(256), .DEPTH(GROUP_COUNT),
        .ADDR_WIDTH(GROUP_ADDR_WIDTH)
    ) u_state_bank (
        .clk(clk),
        .wr_req(state_write_valid),
        .wr_addr(state_write_group),
        .wr_data(state_bank_write),
        .rd_en(1'b1),
        .rd_addr(read_group),
        .rd_data(state_bank_read));

    genvar state_lane;
    generate
        for (state_lane = 0; state_lane < STATE_COUNT;
             state_lane = state_lane + 1) begin : GEN_STATE_CELL
            wire signed [31:0] old_state
                = reset_state_lane_q[state_lane] ? 32'sd0
                : forward_d3_hit_local ? state_forward_next[state_lane]
                : forward_d4_hit_local ? state_next_comb[state_lane]
                : $signed(state_bank_read[state_lane*32 +: 32]);
            wire signed [7:0] lane_b
                = $signed(b_states[state_lane*8 +: 8]);

            ssm_update_3dsp #(.K_FRACTION_BITS(K_FRACTION_BITS))
            u_fused_update (
                .CLK(clk), .VALID(issue_valid),
                .ABAR_U25(abar_word[state_lane*25 +: 25]),
                .STATE_S32(old_state), .K_U19(k_word),
                .B_S8(lane_b),
                .U_S8((state_lane < 4) ? u_s8_local_lo
                                       : u_s8_local_hi),
                .STATE_ACC_Q48(state_accumulator[state_lane]),
                .STATE_FORWARD_Q48(
                    state_forward_accumulator[state_lane]));

            assign state_next_comb[state_lane]
                = round_shift24_to_int32(state_accumulator[state_lane]);
            assign state_forward_next[state_lane]
                = round_shift24_to_int32(
                    state_forward_accumulator[state_lane]);
            assign state_bank_write[state_lane*32 +: 32]
                = state_next_q[state_lane];

            if (USE_DSP_C_PRODUCT != 0) begin : GEN_DSP_C_PRODUCT
                ssm_c_product_32x8_5cyc u_c_product (
                    .clk(clk),
                    .state_s32(state_next_q[state_lane]),
                    .c_s8($signed(c_states_q[state_lane*8 +: 8])),
                    .product_s40(c_product[state_lane]));
            end else begin : GEN_LUT_C_PRODUCT
                (* use_dsp = "no" *)
                wire signed [39:0] c_product_lut
                    = state_next_q[state_lane]
                    * $signed(c_states_q[state_lane*8 +: 8]);
                assign c_product[state_lane] = c_product_lut;
            end
        end
    endgenerate

    // Payload registers intentionally have no reset and no clock enable.
    // state_write_valid and the parent valid pipeline determine whether the
    // continuously changing payload is architecturally visible.
    integer capture_index;
    always @(posedge clk) begin
        reset_state_lane_q <= {8{reset_state_request}};
        u_s8_local_lo <= u_s8;
        u_s8_local_hi <= u_s8;
        c_states_q <= c_states;
        for (capture_index = 0; capture_index < STATE_COUNT;
             capture_index = capture_index + 1)
            state_next_q[capture_index]
                <= state_next_comb[capture_index];
    end

    // Four registered stages cover product capture and an 8-input reduction.
    // The channel-bank wrapper adds the fifth stage when combining its two
    // state halves, exactly matching the former 16-input reduction latency.
    reg signed [39:0] c_product_q [0:7];
    reg signed [40:0] c_level1 [0:3];
    reg signed [41:0] c_level2 [0:1];
    reg signed [42:0] c_level3;
    integer reduce_index;
    always @(posedge clk) begin
        for (reduce_index = 0; reduce_index < 8;
             reduce_index = reduce_index + 1)
            c_product_q[reduce_index] <= c_product[reduce_index];
        for (reduce_index = 0; reduce_index < 4;
             reduce_index = reduce_index + 1)
            c_level1[reduce_index]
                <= $signed({c_product_q[reduce_index*2][39],
                            c_product_q[reduce_index*2]})
                 + $signed({c_product_q[reduce_index*2+1][39],
                            c_product_q[reduce_index*2+1]});
        for (reduce_index = 0; reduce_index < 2;
             reduce_index = reduce_index + 1)
            c_level2[reduce_index]
                <= $signed({c_level1[reduce_index*2][40],
                            c_level1[reduce_index*2]})
                 + $signed({c_level1[reduce_index*2+1][40],
                            c_level1[reduce_index*2+1]});
        c_level3
            <= $signed({c_level2[0][41], c_level2[0]})
             + $signed({c_level2[1][41], c_level2[1]});
    end
    assign c_sum_s43 = c_level3;
endmodule

// One physical four-channel endpoint is divided again along the state axis.
// The two 8-state subbanks own independent state memories and arithmetic
// clusters; only their narrow C partial sums meet at this registered boundary.
module ssm_block0_fused_channel_bank #(
    parameter integer GROUP_COUNT = 4,
    parameter integer GROUP_ADDR_WIDTH = 2,
    parameter integer RECORD_ADDR_WIDTH = 8,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 0,
    parameter integer USE_DSP_C_PRODUCT = 1
) (
    input  wire                         clk,
    input  wire                         issue_valid,
    input  wire                         reset_state_request,
    input  wire [RECORD_ADDR_WIDTH-1:0] issue_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  issue_group,
    input  wire                         forward_d3_valid,
    input  wire [RECORD_ADDR_WIDTH-1:0] forward_d3_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  forward_d3_group,
    input  wire                         forward_d4_valid,
    input  wire [RECORD_ADDR_WIDTH-1:0] forward_d4_record,
    input  wire [GROUP_ADDR_WIDTH-1:0]  forward_d4_group,
    input  wire [GROUP_ADDR_WIDTH-1:0]  read_group,
    input  wire [GROUP_ADDR_WIDTH-1:0]  state_write_group,
    input  wire [399:0]                 abar_word,
    input  wire [18:0]                  k_word,
    input  wire [127:0]                 b_record,
    input  wire signed [7:0]            u_s8,
    input  wire [127:0]                 c_record,
    output wire signed [43:0]           c_sum_s44
);
    wire signed [42:0] c_half_sum [0:1];
    // One B snapshot is captured at the physical channel-bank boundary.
    // The parent now supplies the synchronous BRAM word directly, so this
    // register has exactly the same R0->R1 latency as the former global
    // r1_b_record while eliminating its four-bank broadcast.
    reg [127:0] b_record_local;
    always @(posedge clk)
        b_record_local <= b_record;
    genvar state_half;
    generate
        for (state_half=0; state_half<2; state_half=state_half+1)
            begin : GEN_STATE_SUBBANK
            (* keep_hierarchy = "yes" *)
            ssm_block0_fused_state_subbank8 #(
                .GROUP_COUNT(GROUP_COUNT),
                .GROUP_ADDR_WIDTH(GROUP_ADDR_WIDTH),
                .RECORD_ADDR_WIDTH(RECORD_ADDR_WIDTH),
                .K_FRACTION_BITS(K_FRACTION_BITS),
                .RESET_PERIOD(RESET_PERIOD),
                .STATE_BASE(state_half*8),
                .USE_DSP_C_PRODUCT(USE_DSP_C_PRODUCT)
            ) u_state_half (
                .clk(clk), .issue_valid(issue_valid),
                .reset_state_request(reset_state_request),
                .issue_record(issue_record), .issue_group(issue_group),
                .forward_d3_valid(forward_d3_valid),
                .forward_d3_record(forward_d3_record),
                .forward_d3_group(forward_d3_group),
                .forward_d4_valid(forward_d4_valid),
                .forward_d4_record(forward_d4_record),
                .forward_d4_group(forward_d4_group),
                .read_group(read_group),
                .state_write_valid(forward_d4_valid),
                .state_write_group(state_write_group),
                .abar_word(abar_word[state_half*200 +: 200]),
                .k_word(k_word),
                .b_states(b_record_local[state_half*64 +: 64]),
                .u_s8(u_s8),
                .c_states(c_record[state_half*64 +: 64]),
                .c_sum_s43(c_half_sum[state_half])
            );
        end
    endgenerate

    reg signed [43:0] c_sum_q;
    always @(posedge clk)
        c_sum_q <= $signed({c_half_sum[0][42],c_half_sum[0]})
                 + $signed({c_half_sum[1][42],c_half_sum[1]});
    assign c_sum_s44 = c_sum_q;
endmodule

// Block0 Spa/Spe fused selective-scan core.
//
// One request covers four channels x sixteen states.  Unlike the legacy
// B -> A -> X schedule, this core launches all three mathematical products
// for a state cell together:
//
//   h' = Abar*h + K*(B*U)
//
// Abar and K are read from the same DT-addressed v5 ROM word.  U and B remain
// signed INT8.  Every ready record therefore needs only GROUP_COUNT issue
// clocks (16 for Spa or 4 for Spe), while a record ID carried through the
// pipeline preserves recurrence order across continuous record boundaries.
// Spe uses a four-record reset period: records 0/4/8/... ignore the previous
// state, while the intervening records consume the just-written LUTRAM state.
module ssm64_parallel_core_block0_spa_fused #(
`ifdef MAMBA_D_PATH_D1
    parameter integer D_ENABLE = 1,
`else
    parameter integer D_ENABLE = 0,
`endif
    parameter integer BLOCK_ID = 0,
    parameter integer IS_SPE = 0,
    parameter integer CHANNEL_COUNT = 64,
    parameter integer RECORD_COUNT = 256,
    parameter integer GROUP_COUNT = 16,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 0,
    parameter integer STREAM_INPUT = 0,
    parameter integer CONTINUOUS_TILES = 0
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
    output wire                 busy,
    output wire                 input_record_ready,
    output wire                 input_record_consumed
);
    localparam integer STATE_COUNT = 16;
    localparam integer RECORD_ADDR_WIDTH
        = (RECORD_COUNT <= 256) ? 8 : 10;
    localparam integer GROUP_ADDR_WIDTH
        = (GROUP_COUNT <= 4) ? 2 : 4;
    localparam integer CHANNEL_BEATS = CHANNEL_COUNT / 8;
    localparam integer LAST_CHANNEL_BASE = CHANNEL_COUNT - 8;
    localparam [0:0] ST_IDLE = 1'b0;
    localparam [0:0] ST_RUN  = 1'b1;

    // Only the two Block0 geometries are legal.  This catches accidental use
    // by the area-scaled Block1/2 wrappers without adding synthesis hardware.
`ifndef SYNTHESIS
    initial begin
        if ((BLOCK_ID != 0)
            || !(((IS_SPE == 0) && (CHANNEL_COUNT == 64)
                   && (RECORD_COUNT == 256) && (GROUP_COUNT == 16)
                   && (RESET_PERIOD == 0))
               || ((IS_SPE != 0) && (CHANNEL_COUNT == 16)
                   && (RECORD_COUNT == 1024) && (GROUP_COUNT == 4)
                   && (RESET_PERIOD == 4)))) begin
            $display("Block0 fused core instantiated with unsupported geometry");
            $finish;
        end
    end
`endif

    reg fsm_state;
    reg frame_active;
    reg [RECORD_ADDR_WIDTH-1:0] current_record;
    reg [GROUP_ADDR_WIDTH-1:0] request_group;
    wire request_fire = (fsm_state == ST_RUN);

    // Ordered producer-prefix watermarks: a record is consumed only after all
    // U/DT/B/C components have arrived.  This also prevents BRAM read/write
    // collisions while capture and recurrence overlap.
    reg [31:0] u_records_ready;
    reg [31:0] dt_records_ready;
    reg [31:0] b_records_ready;
    reg [31:0] c_records_ready;
    // Readiness is an absolute sequence; BRAM addresses and state-reset
    // boundaries remain tile-local.  Old tile watermarks must never make
    // record zero of the next tile appear ready.
    reg [31:0] current_sequence;
    wire [31:0] next_sequence = current_sequence + 1'b1;
    // Hardware readiness depends only on the number of complete records not
    // yet consumed.  Keep the existing wide sequence counters for simulation
    // diagnostics, while the datapath uses these bounded ring occupancies.
    localparam integer READY_COUNT_WIDTH = $clog2(RECORD_COUNT + 1);
    reg [READY_COUNT_WIDTH-1:0] u_pending_records;
    reg [READY_COUNT_WIDTH-1:0] dt_pending_records;
    reg [READY_COUNT_WIDTH-1:0] b_pending_records;
    reg [READY_COUNT_WIDTH-1:0] c_pending_records;
    wire u_record_complete = u_valid
                           && (u_channel_base == LAST_CHANNEL_BASE);
    wire dt_record_complete = dt_valid
                            && (dt_channel_base == LAST_CHANNEL_BASE);
    wire [RECORD_ADDR_WIDTH:0] current_record_extended
        = {1'b0, current_record};
    wire current_is_last=(current_record==RECORD_COUNT-1);
    wire [RECORD_ADDR_WIDTH-1:0] next_record_index
        =current_is_last?{RECORD_ADDR_WIDTH{1'b0}}:
         (current_record+1'b1);
    wire [RECORD_ADDR_WIDTH:0] next_record_extended
        ={1'b0,next_record_index};
    // A producer completion and the last request of the preceding record can
    // occur on the same edge.  The old watermark-only comparison could not see
    // that edge until the following clock, forcing one ST_IDLE bubble even
    // though the just-written BRAM word is available before group 0 reads it.
    // Include the monotonic producer's current completion in the visibility
    // test; record/address equality prevents an unrelated write qualifying it.
    wire u_current_complete_now=u_record_complete
        &&(u_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire dt_current_complete_now=dt_record_complete
        &&(dt_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire b_current_complete_now=b_valid
        &&(b_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire c_current_complete_now=c_valid
        &&(c_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire u_next_complete_now=u_record_complete
        &&(u_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire dt_next_complete_now=dt_record_complete
        &&(dt_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire b_next_complete_now=b_valid
        &&(b_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire c_next_complete_now=c_valid
        &&(c_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire legacy_current_record_ready
        = ((u_pending_records  != 0)||u_current_complete_now)
       && ((dt_pending_records != 0)||dt_current_complete_now)
       && ((b_pending_records  != 0)||b_current_complete_now)
       && ((c_pending_records  != 0)||c_current_complete_now);
    wire legacy_next_record_ready
        = ((u_pending_records  > 1)||u_next_complete_now)
       && ((dt_pending_records > 1)||dt_next_complete_now)
       && ((b_pending_records  > 1)||b_next_complete_now)
       && ((c_pending_records  > 1)||c_next_complete_now);
    wire legacy_record_zero_ready
        = (u_pending_records != 0) && (dt_pending_records != 0)
       && (b_pending_records != 0) && (c_pending_records != 0);

    // STREAM_INPUT is enabled for the credit-protected Spa branch. Four modulo-addressed
    // record slots replace the full-frame U/DT/B/C memories.  Payload is not
    // cleared when a slot is released; tags and ready masks define validity.
    reg [511:0] stream_u_slot [0:3];
    reg [511:0] stream_dt_slot [0:3];
    reg [127:0] stream_b_slot [0:3];
    reg [127:0] stream_c_slot [0:3];
    reg [9:0] stream_slot_tag [0:3];
    reg [7:0] stream_u_mask [0:3];
    reg [7:0] stream_dt_mask [0:3];
    reg [3:0] stream_b_ready;
    reg [3:0] stream_c_ready;
    reg [3:0] stream_slot_active;
    localparam [7:0] STREAM_EXPECTED_MASK =
        (CHANNEL_BEATS == 8) ? 8'hff :
        (CHANNEL_BEATS == 2) ? 8'h03 : 8'h01;
    wire [1:0] stream_current_slot = current_record[1:0];
    wire [1:0] stream_next_slot = next_record_index[1:0];
    wire stream_u_targets_current=u_valid
        &&(u_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire stream_dt_targets_current=dt_valid
        &&(dt_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire stream_b_targets_current=b_valid
        &&(b_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire stream_c_targets_current=c_valid
        &&(c_record_addr[RECORD_ADDR_WIDTH-1:0]==current_record);
    wire stream_u_targets_next=u_valid
        &&(u_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire stream_dt_targets_next=dt_valid
        &&(dt_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire stream_b_targets_next=b_valid
        &&(b_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire stream_c_targets_next=c_valid
        &&(c_record_addr[RECORD_ADDR_WIDTH-1:0]==next_record_index);
    wire[7:0] stream_current_u_mask_visible
        =stream_u_mask[stream_current_slot]
        |(stream_u_targets_current?(8'b1<<(u_channel_base>>3)):8'b0);
    wire[7:0] stream_current_dt_mask_visible
        =stream_dt_mask[stream_current_slot]
        |(stream_dt_targets_current?(8'b1<<(dt_channel_base>>3)):8'b0);
    wire[7:0] stream_next_u_mask_visible
        =stream_u_mask[stream_next_slot]
        |(stream_u_targets_next?(8'b1<<(u_channel_base>>3)):8'b0);
    wire[7:0] stream_next_dt_mask_visible
        =stream_dt_mask[stream_next_slot]
        |(stream_dt_targets_next?(8'b1<<(dt_channel_base>>3)):8'b0);
    wire stream_current_slot_visible
        =(stream_slot_active[stream_current_slot]
          &&(stream_slot_tag[stream_current_slot]==current_record_extended))
        ||(stream_u_targets_current&&(u_channel_base==0));
    wire stream_next_slot_visible
        =(stream_slot_active[stream_next_slot]
          &&(stream_slot_tag[stream_next_slot]==next_record_extended))
        ||(stream_u_targets_next&&(u_channel_base==0));
    wire stream_current_record_ready
        =stream_current_slot_visible
       &&(stream_current_u_mask_visible==STREAM_EXPECTED_MASK)
       &&(stream_current_dt_mask_visible==STREAM_EXPECTED_MASK)
       &&(stream_b_ready[stream_current_slot]||stream_b_targets_current)
       &&(stream_c_ready[stream_current_slot]||stream_c_targets_current);
    wire stream_next_record_ready
        =stream_next_slot_visible
       &&(stream_next_u_mask_visible==STREAM_EXPECTED_MASK)
       &&(stream_next_dt_mask_visible==STREAM_EXPECTED_MASK)
       &&(stream_b_ready[stream_next_slot]||stream_b_targets_next)
       &&(stream_c_ready[stream_next_slot]||stream_c_targets_next);
    wire [2:0] stream_active_count = stream_slot_active[0]
        + stream_slot_active[1] + stream_slot_active[2]
        + stream_slot_active[3];
    wire stream_release = (STREAM_INPUT != 0) && request_fire
        && (request_group == GROUP_COUNT-1);
    // Physical-space indication for diagnostics.  Admission is governed by
    // the launch-time credit counter in mamba_block_rom_top.  The slot being
    // released by the last group can be replaced in this same cycle; the
    // valid/tag checks below keep the old and new records unambiguous.
    assign input_record_ready = (STREAM_INPUT == 0)
        ? 1'b1 : ((stream_active_count <= 3'd3) || stream_release);
    wire current_record_ready = (STREAM_INPUT != 0)
        ? stream_current_record_ready : legacy_current_record_ready;
    wire next_record_ready = (STREAM_INPUT != 0)
        ? stream_next_record_ready : legacy_next_record_ready;
    wire record_zero_ready = (STREAM_INPUT != 0)
        ? stream_current_record_ready : legacy_record_zero_ready;
    assign input_record_consumed = stream_release;
    wire legacy_record_consume = (STREAM_INPUT == 0) && request_fire
        && (request_group == GROUP_COUNT-1);

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            stream_u_mask[0] <= 0; stream_u_mask[1] <= 0;
            stream_u_mask[2] <= 0; stream_u_mask[3] <= 0;
            stream_dt_mask[0] <= 0; stream_dt_mask[1] <= 0;
            stream_dt_mask[2] <= 0; stream_dt_mask[3] <= 0;
            stream_b_ready <= 0;
            stream_c_ready <= 0;
            stream_slot_active <= 0;
        end else if (STREAM_INPUT != 0) begin
            if (stream_release) begin
                stream_slot_active[stream_current_slot] <= 1'b0;
                stream_u_mask[stream_current_slot] <= 0;
                stream_dt_mask[stream_current_slot] <= 0;
                stream_b_ready[stream_current_slot] <= 1'b0;
                stream_c_ready[stream_current_slot] <= 1'b0;
            end
            if (u_valid) begin
                if (u_channel_base == 0) begin
                    stream_slot_active[u_record_addr[1:0]] <= 1'b1;
                    stream_slot_tag[u_record_addr[1:0]] <= u_record_addr;
                    stream_u_mask[u_record_addr[1:0]] <= 8'h01;
                    stream_dt_mask[u_record_addr[1:0]] <= 0;
                    stream_b_ready[u_record_addr[1:0]] <= 1'b0;
                    stream_c_ready[u_record_addr[1:0]] <= 1'b0;
                end
                stream_u_slot[u_record_addr[1:0]][u_channel_base*8 +: 64]
                    <= u_data;
                if (u_channel_base != 0)
                    stream_u_mask[u_record_addr[1:0]][u_channel_base >> 3]
                        <= 1'b1;
            end
            if (dt_valid) begin
                stream_dt_slot[dt_record_addr[1:0]][dt_channel_base*8 +: 64]
                    <= dt_data;
                stream_dt_mask[dt_record_addr[1:0]][dt_channel_base >> 3]
                    <= 1'b1;
            end
            if (b_valid) begin
                stream_b_slot[b_record_addr[1:0]] <= b_data;
                stream_b_ready[b_record_addr[1:0]] <= 1'b1;
            end
            if (c_valid) begin
                stream_c_slot[c_record_addr[1:0]] <= c_data;
                stream_c_ready[c_record_addr[1:0]] <= 1'b1;
            end
        end
    end

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            u_pending_records <= 0;
            dt_pending_records <= 0;
            b_pending_records <= 0;
            c_pending_records <= 0;
            u_records_ready <= 0;
            dt_records_ready <= 0;
            b_records_ready <= 0;
            c_records_ready <= 0;
        end else begin
            case ({u_record_complete,legacy_record_consume})
                2'b10: if (u_pending_records < RECORD_COUNT)
                    u_pending_records <= u_pending_records + 1'b1;
                2'b01: if (u_pending_records != 0)
                    u_pending_records <= u_pending_records - 1'b1;
                default: u_pending_records <= u_pending_records;
            endcase
            case ({dt_record_complete,legacy_record_consume})
                2'b10: if (dt_pending_records < RECORD_COUNT)
                    dt_pending_records <= dt_pending_records + 1'b1;
                2'b01: if (dt_pending_records != 0)
                    dt_pending_records <= dt_pending_records - 1'b1;
                default: dt_pending_records <= dt_pending_records;
            endcase
            case ({b_valid,legacy_record_consume})
                2'b10: if (b_pending_records < RECORD_COUNT)
                    b_pending_records <= b_pending_records + 1'b1;
                2'b01: if (b_pending_records != 0)
                    b_pending_records <= b_pending_records - 1'b1;
                default: b_pending_records <= b_pending_records;
            endcase
            case ({c_valid,legacy_record_consume})
                2'b10: if (c_pending_records < RECORD_COUNT)
                    c_pending_records <= c_pending_records + 1'b1;
                2'b01: if (c_pending_records != 0)
                    c_pending_records <= c_pending_records - 1'b1;
                default: c_pending_records <= c_pending_records;
            endcase
            if (u_record_complete && ((CONTINUOUS_TILES != 0) || (u_records_ready < RECORD_COUNT)))
                u_records_ready <= u_records_ready + 1'b1;
            if (dt_record_complete && ((CONTINUOUS_TILES != 0) || (dt_records_ready < RECORD_COUNT)))
                dt_records_ready <= dt_records_ready + 1'b1;
            if (b_valid && ((CONTINUOUS_TILES != 0) || (b_records_ready < RECORD_COUNT)))
                b_records_ready <= b_records_ready + 1'b1;
            if (c_valid && ((CONTINUOUS_TILES != 0) || (c_records_ready < RECORD_COUNT)))
                c_records_ready <= c_records_ready + 1'b1;
        end
    end

    // Native record memories and eight-channel U/DT beat memories.
    wire [127:0] b_record_word;
    wire [127:0] c_record_word;
    wire [10:0] u_write_addr
        = u_record_addr*CHANNEL_BEATS + (u_channel_base >> 3);
    wire [10:0] dt_write_addr
        = dt_record_addr*CHANNEL_BEATS + (dt_channel_base >> 3);
    wire [10:0] runtime_input_read_addr
        = current_record*CHANNEL_BEATS + (request_group >> 1);
    wire [63:0] u_read_word;
    wire [63:0] dt_read_word;

    generate
        if (STREAM_INPUT != 0) begin : G_STREAM_RECORD_INPUT
            reg [127:0] b_read_q;
            reg [127:0] c_read_q;
            reg [63:0] u_read_q;
            reg [63:0] dt_read_q;
            always @(posedge clk) begin
                if (request_fire) begin
                    b_read_q <= stream_b_slot[stream_current_slot];
                    c_read_q <= stream_c_slot[stream_current_slot];
                    u_read_q <= stream_u_slot[stream_current_slot]
                        [(request_group >> 1)*64 +: 64];
                    dt_read_q <= stream_dt_slot[stream_current_slot]
                        [(request_group >> 1)*64 +: 64];
                end
            end
            assign b_record_word = b_read_q;
            assign c_record_word = c_read_q;
            assign u_read_word = u_read_q;
            assign dt_read_word = dt_read_q;
        end else begin : G_FRAME_RECORD_INPUT
            ssm_bc_record_bram #(
                .DEPTH(RECORD_COUNT), .ADDR_WIDTH(RECORD_ADDR_WIDTH)
            ) u_b_record_bram (
                .clk(clk), .wr_en(b_valid),
                .wr_addr(b_record_addr[RECORD_ADDR_WIDTH-1:0]),
                .wr_data(b_data), .rd_en(request_fire),
                .rd_addr(current_record), .rd_data(b_record_word));
            ssm_bc_record_bram #(
                .DEPTH(RECORD_COUNT), .ADDR_WIDTH(RECORD_ADDR_WIDTH)
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
        end
    endgenerate

    // R0 tags the synchronous U/DT/B/C request.  At the following edge those
    // words address the nonlinear ROM and the state scratch in parallel.
    reg raw_r0_valid;
    reg [RECORD_ADDR_WIDTH-1:0] raw_r0_record;
    reg [GROUP_ADDR_WIDTH-1:0] raw_r0_group;
    reg r1_valid;
    reg [RECORD_ADDR_WIDTH-1:0] r1_record;
    reg [GROUP_ADDR_WIDTH-1:0] r1_group;
    reg [127:0] r1_c_record;

    wire [31:0] raw_selected_u_group
        = raw_r0_group[0] ? u_read_word[63:32] : u_read_word[31:0];
    wire [31:0] selected_dt_group
        = raw_r0_group[0] ? dt_read_word[63:32] : dt_read_word[31:0];

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
    wire [31:0] selected_u_group;
    wire [127:0] nf_b_record_word, nf_c_record_word;
    nf_align #(.WIDTH(RECORD_ADDR_WIDTH+GROUP_ADDR_WIDTH+32+256), .DEPTH(NF_DELAY)) u_nf_align (
        .clk(clk), .reset(!rst_n || frame_start),
        .in_valid(raw_r0_valid), .out_valid(r0_valid), .pending(nf_pending),
        .in_data({raw_r0_record, raw_r0_group, raw_selected_u_group, b_record_word, c_record_word}),
        .out_data({r0_record, r0_group, selected_u_group, nf_b_record_word, nf_c_record_word})
    );

    wire [418:0] lane_lut_word0;
    wire [418:0] lane_lut_word1;
    wire [418:0] lane_lut_word2;
    wire [418:0] lane_lut_word3;
    generate if (D_ENABLE != 0) begin : G_D1_NONLINEAR
      if (`NF_METHOD == 3) begin : G_NATIVE_ROM
        mamba_d1_nonlinear_rom_4r #(.BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE))
        u_rom (.clk(clk), .en(1'b1),
            .addr0(int8_lut_address(selected_dt_group[7:0])),
            .addr1(int8_lut_address(selected_dt_group[15:8])),
            .addr2(int8_lut_address(selected_dt_group[23:16])),
            .addr3(int8_lut_address(selected_dt_group[31:24])),
            .data0(lane_lut_word0), .data1(lane_lut_word1),
            .data2(lane_lut_word2), .data3(lane_lut_word3));

      end else begin : G_ONLINE
        wire nf_coeff_valid;
        nf_coeff_lanes #(.METHOD(`NF_METHOD), .BLOCK_ID(BLOCK_ID),
            .IS_SPE(IS_SPE), .LANES(4)) u_online (
            .clk(clk), .reset(!rst_n || frame_start),
            .in_valid(raw_r0_valid), .q_dt(selected_dt_group),
            .out_valid(nf_coeff_valid), .data0(lane_lut_word0), .data1(lane_lut_word1), .data2(lane_lut_word2), .data3(lane_lut_word3));
`ifndef SYNTHESIS
        always @(posedge clk) if (rst_n && !frame_start &&
            (nf_coeff_valid !== r1_valid))
            $fatal(1, "NL coefficient/metadata valid misalignment B%0d Spe%0d", BLOCK_ID, IS_SPE);
`endif
      end
    end else begin : G_LEGACY_NONLINEAR
    mamba_ssm_lut_rom_4r #(.BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE))
    u_nonlinear_lut_bram (
        // Keep the fixed nonlinear ROM read ports enabled.  r1_valid already
        // qualifies the returned word, and removing r0_valid from the BRAM
        // EN pins avoids a long high-fanout control route.
        .clk(clk), .en(1'b1),
        .addr0(int8_lut_address(selected_dt_group[7:0])),
        .addr1(int8_lut_address(selected_dt_group[15:8])),
        .addr2(int8_lut_address(selected_dt_group[23:16])),
        .addr3(int8_lut_address(selected_dt_group[31:24])),
        .data0(lane_lut_word0), .data1(lane_lut_word1),
        .data2(lane_lut_word2), .data3(lane_lut_word3));

    end endgenerate
    // The arithmetic samples ROM/state outputs when r1_valid is asserted.
    // Its fixed five-edge latency is mirrored by the metadata below.
    // Preserve a physically separate valid pipeline for each four-channel
    // bank.  The previous scalar arith_valid_d4 drove state write, forwarding
    // and C-product control throughout all four banks (641 routed loads).
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_arith_valid_d0;
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_arith_valid_d1;
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_arith_valid_d2;
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_arith_valid_d3;
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_arith_valid_d4;
    wire arith_valid_d0=bank_arith_valid_d0[0];
    wire arith_valid_d1=bank_arith_valid_d1[0];
    wire arith_valid_d2=bank_arith_valid_d2[0];
    wire arith_valid_d3=bank_arith_valid_d3[0];
    wire arith_valid_d4=bank_arith_valid_d4[0];
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

    // Duplicate the issue endpoint per four-channel bank.  Reset is derived
    // directly from R0 and captured into one register per state inside each
    // 8-state physical subbank.  This retains the old R0->R1 alignment while
    // eliminating the former 512-load reset endpoint.
    (* keep = "true", max_fanout = 32 *) reg [3:0] bank_issue_valid;
    wire reset_state_request = (RESET_PERIOD == 4)
        ? (r0_record[1:0] == 0) : (r0_record == 0);
    reg state_round_valid;
    reg [RECORD_ADDR_WIDTH-1:0] state_round_record;
    reg [GROUP_ADDR_WIDTH-1:0] state_round_group;

    // With four groups per Spe record, record r+1 reaches the arithmetic
    // input on the same edge that record r's registered result is produced.
    // arith_d3 identifies that exact predecessor and selects the shared
    // pre-register final sum.  arith_d4 covers a one-cycle producer stall;
    // after two or more stalls the LUTRAM write is already visible.
`ifndef SYNTHESIS
    wire spe_forward_d3_hit
        = (RESET_PERIOD == 4) && r1_valid && (r1_record[1:0] != 0)
       && arith_valid_d3
       && (arith_group_d3 == r1_group)
       && (arith_record_d3 + 1'b1 == r1_record);
    wire spe_forward_d4_hit
        = (RESET_PERIOD == 4) && r1_valid && (r1_record[1:0] != 0)
       && arith_valid_d4
       && (arith_group_d4 == r1_group)
       && (arith_record_d4 + 1'b1 == r1_record);
`endif

    // The complete state/C/update/readout cluster is physically local to
    // each four-channel endpoint.  Only four 44-bit partial sums return to
    // this parent; there is no longer a 512-bit state write bus or 64-wide C
    // product tree crossing the bank boundaries.
    wire signed [43:0] bank_c_sum [0:3];
    genvar channel_bank;
    generate
        for (channel_bank = 0; channel_bank < 4;
             channel_bank = channel_bank + 1) begin : GEN_CHANNEL_BANK
            localparam integer CHANNEL_LANE = channel_bank;
            wire [418:0] bank_lut_word
                = (CHANNEL_LANE == 0) ? lane_lut_word0
                : (CHANNEL_LANE == 1) ? lane_lut_word1
                : (CHANNEL_LANE == 2) ? lane_lut_word2
                                      : lane_lut_word3;
            (* keep_hierarchy = "yes" *)
            ssm_block0_fused_channel_bank #(
                .GROUP_COUNT(GROUP_COUNT),
                .GROUP_ADDR_WIDTH(GROUP_ADDR_WIDTH),
                .RECORD_ADDR_WIDTH(RECORD_ADDR_WIDTH),
                .K_FRACTION_BITS(K_FRACTION_BITS),
                .RESET_PERIOD(RESET_PERIOD),
                // Both branches use the same five-cycle, area-prioritised
                // 32x8 Multiplier Generator.  This removes the 64-wide Spa
                // LUT multiplier field and keeps one verified latency model.
                .USE_DSP_C_PRODUCT(1)
            ) u_local_bank (
                .clk(clk),
                .issue_valid(bank_issue_valid[CHANNEL_LANE]),
                .reset_state_request(reset_state_request),
                .issue_record(r1_record),
                .issue_group(r1_group),
                .forward_d3_valid(bank_arith_valid_d3[CHANNEL_LANE]),
                .forward_d3_record(arith_record_d3),
                .forward_d3_group(arith_group_d3),
                .forward_d4_valid(bank_arith_valid_d4[CHANNEL_LANE]),
                .forward_d4_record(arith_record_d4),
                .forward_d4_group(arith_group_d4),
                .read_group(r0_group),
                .state_write_group(arith_group_d4),
                .abar_word(bank_lut_word[399:0]),
                .k_word(bank_lut_word[418:400]),
                .b_record(nf_b_record_word),
                .u_s8($signed(selected_u_group[CHANNEL_LANE*8 +: 8])),
                .c_record(arith_c_record_d4),
                .c_sum_s44(bank_c_sum[CHANNEL_LANE])
            );
        end
    endgenerate

    // D uses the SAME U snapshot as the recurrence.  Its narrow address
    // pipeline follows R1, arithmetic[0:4], state_round, C-DSP[0:4],
    // product and reduction[1:4]: 17 edges from selected_u_group to Y_L4.
    // Lookup is late, so we delay 9/10-bit addresses, not wide products.
    wire signed [47:0] d_readout [0:3];
    genvar d_lane;
    generate for (d_lane=0; d_lane<4; d_lane=d_lane+1) begin : G_D_READOUT
        if (D_ENABLE != 0) begin : G_ENABLED
            wire [5:0] d_channel = r0_group*4 + d_lane;
            mamba_d_path_lut #(
                .BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE), .LATENCY(17)
            ) u_d_lut (.clk(clk), .channel(d_channel),
                .u_s8(selected_u_group[d_lane*8 +: 8]),
                .d_s48(d_readout[d_lane]));
        end else begin : G_DISABLED
            assign d_readout[d_lane] = 48'sd0;
        end
    end endgenerate

    // Five additional valid/metadata clocks cover the Spa/Spe 32x8
    // Multiplier Generator pipeline.  The bank-local product register and
    // four reduction levels retain the proven output ordering.
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
    wire c_product_capture_valid = c_dsp_valid_d4;
    wire [RECORD_ADDR_WIDTH-1:0] c_product_capture_record
        = c_dsp_record_d4;
    wire [GROUP_ADDR_WIDTH-1:0] c_product_capture_group
        = c_dsp_group_d4;

    reg y_product_valid;
    reg [RECORD_ADDR_WIDTH-1:0] y_product_record;
    reg [GROUP_ADDR_WIDTH-1:0] y_product_group;
    reg y_valid_l1;
    reg y_valid_l2;
    reg y_valid_l3;
    reg y_valid_l4;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l1;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l2;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l3;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l4;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l1;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l2;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l3;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l4;

    assign busy = frame_active || (fsm_state != ST_IDLE)
               || raw_r0_valid || nf_pending || r0_valid || r1_valid
               || arith_valid_d0 || arith_valid_d1 || arith_valid_d2
               || arith_valid_d3 || arith_valid_d4
               || state_round_valid
               || c_dsp_valid_d0 || c_dsp_valid_d1 || c_dsp_valid_d2
               || c_dsp_valid_d3 || c_dsp_valid_d4
               || y_product_valid || y_valid_l1 || y_valid_l2
               || y_valid_l3 || y_valid_l4 || out_valid;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            fsm_state <= ST_IDLE;
            frame_active <= rst_n && frame_start;
            current_record <= 0;
            current_sequence <= 0;
            request_group <= 0;
            raw_r0_valid <= 1'b0;
            r1_valid <= 1'b0;
            bank_issue_valid <= 4'b0000;
            bank_arith_valid_d0 <= 4'b0000;
            bank_arith_valid_d1 <= 4'b0000;
            bank_arith_valid_d2 <= 4'b0000;
            bank_arith_valid_d3 <= 4'b0000;
            bank_arith_valid_d4 <= 4'b0000;
            state_round_valid <= 1'b0;
            c_dsp_valid_d0 <= 1'b0;
            c_dsp_valid_d1 <= 1'b0;
            c_dsp_valid_d2 <= 1'b0;
            c_dsp_valid_d3 <= 1'b0;
            c_dsp_valid_d4 <= 1'b0;
            y_product_valid <= 1'b0;
            y_valid_l1 <= 1'b0;
            y_valid_l2 <= 1'b0;
            y_valid_l3 <= 1'b0;
            y_valid_l4 <= 1'b0;
            out_valid <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            out_valid <= y_valid_l4;

            // Runtime-memory request metadata.
            raw_r0_valid <= request_fire;
            raw_r0_record <= current_record;
            raw_r0_group <= request_group;
            r1_valid <= r0_valid;
            bank_issue_valid <= {4{r0_valid}};
            // Runtime words and tags are payload, not control.  Keep all of
            // them free-running so r0_valid is not promoted into a wide CE
            // tree; bank_issue_valid alone qualifies the bank arithmetic.
            r1_record <= r0_record;
            r1_group <= r0_group;
            r1_c_record <= nf_c_record_word;

            // Fused arithmetic metadata follows bank-local issue valid.
            bank_arith_valid_d0 <= bank_issue_valid;
            bank_arith_valid_d1 <= bank_arith_valid_d0;
            bank_arith_valid_d2 <= bank_arith_valid_d1;
            bank_arith_valid_d3 <= bank_arith_valid_d2;
            bank_arith_valid_d4 <= bank_arith_valid_d3;
            // Metadata and C payload shift every clock.  Their valid bits
            // alone qualify use, eliminating five levels of payload CE.
            arith_record_d0 <= r1_record;
            arith_record_d1 <= arith_record_d0;
            arith_record_d2 <= arith_record_d1;
            arith_record_d3 <= arith_record_d2;
            arith_record_d4 <= arith_record_d3;
            arith_group_d0 <= r1_group;
            arith_group_d1 <= arith_group_d0;
            arith_group_d2 <= arith_group_d1;
            arith_group_d3 <= arith_group_d2;
            arith_group_d4 <= arith_group_d3;
            arith_c_record_d0 <= r1_c_record;
            arith_c_record_d1 <= arith_c_record_d0;
            arith_c_record_d2 <= arith_c_record_d1;
            arith_c_record_d3 <= arith_c_record_d2;
            arith_c_record_d4 <= arith_c_record_d3;

            // Register round/saturate before the C multiply.  This cuts the
            // former Q48 -> CARRY chain -> DSP path while state forwarding
            // continues to use the exact pre-register value above.
            state_round_valid <= arith_valid_d4;
            state_round_record <= arith_record_d4;
            state_round_group <= arith_group_d4;

            c_dsp_valid_d0 <= state_round_valid;
            c_dsp_valid_d1 <= c_dsp_valid_d0;
            c_dsp_valid_d2 <= c_dsp_valid_d1;
            c_dsp_valid_d3 <= c_dsp_valid_d2;
            c_dsp_valid_d4 <= c_dsp_valid_d3;
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

            y_product_valid <= c_product_capture_valid;
            y_valid_l1 <= y_product_valid;
            y_valid_l2 <= y_valid_l1;
            y_valid_l3 <= y_valid_l2;
            y_valid_l4 <= y_valid_l3;
            y_product_record <= c_product_capture_record;
            y_product_group <= c_product_capture_group;
            y_record_l1 <= y_product_record;
            y_record_l2 <= y_record_l1;
            y_record_l3 <= y_record_l2;
            y_record_l4 <= y_record_l3;
            y_group_l1 <= y_product_group;
            y_group_l2 <= y_group_l1;
            y_group_l3 <= y_group_l2;
            y_group_l4 <= y_group_l3;
            if (y_valid_l4) begin
                out_record_addr <= y_record_l4;
                out_channel_base <= y_group_l4*4;
                out_y_q24[47:0]
                    <= $signed({{4{bank_c_sum[0][43]}}, bank_c_sum[0]}) + d_readout[0];
                out_y_q24[95:48]
                    <= $signed({{4{bank_c_sum[1][43]}}, bank_c_sum[1]}) + d_readout[1];
                out_y_q24[143:96]
                    <= $signed({{4{bank_c_sum[2][43]}}, bank_c_sum[2]}) + d_readout[2];
                out_y_q24[191:144]
                    <= $signed({{4{bank_c_sum[3][43]}}, bank_c_sum[3]}) + d_readout[3];
                if ((y_record_l4 == RECORD_COUNT-1)
                    && (y_group_l4 == GROUP_COUNT-1))
                    done <= 1'b1;
            end

            // One request every clock.  Ready next records cross the boundary
            // without PRE or drain cycles; otherwise the request side pauses
            // while already-issued groups continue through the datapath.
            if (compute_start && !frame_active && (fsm_state == ST_IDLE)) begin
                frame_active <= 1'b1;
                current_record <= 0;
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
                            current_sequence <= next_sequence;
                            if (current_is_last) begin
                                if((STREAM_INPUT!=0)||(CONTINUOUS_TILES!=0))begin
                                    // Tile EOF and the following tile SOF are
                                    // adjacent records in steady state.  Wrap
                                    // the logical record while retaining the
                                    // active context; record zero performs the
                                    // architectural state reset.
                                    current_record<=0;
                                    if(!next_record_ready)
                                        fsm_state<=ST_IDLE;
                                end else begin
                                    fsm_state <= ST_IDLE;
                                    frame_active <= 1'b0;
                                end
                            end else begin
                                current_record <= next_record_index;
                                if (!next_record_ready)
                                    fsm_state <= ST_IDLE;
                            end
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
    always @(posedge clk) begin
        if(rst_n && !frame_start) begin
            if((STREAM_INPUT!=0) && u_valid && (u_channel_base==0)
                && stream_slot_active[u_record_addr[1:0]]
                && !(stream_release && (stream_current_slot==u_record_addr[1:0])))
                $fatal(1,"SSM FIFO overwrite: spe=%0d incoming=%0d slot_tag=%0d current=%0d",
                    IS_SPE,u_record_addr,stream_slot_tag[u_record_addr[1:0]],current_record);
            if((STREAM_INPUT==0) && (CONTINUOUS_TILES!=0))begin
                if(u_valid && (u_channel_base==0)
                    && ((u_records_ready-current_sequence)>=RECORD_COUNT))
                    $fatal(1,"SSM U ring overrun: producer=%0d consumer=%0d",u_records_ready,current_sequence);
                if(dt_valid && (dt_channel_base==0)
                    && ((dt_records_ready-current_sequence)>=RECORD_COUNT))
                    $fatal(1,"SSM DT ring overrun: producer=%0d consumer=%0d",dt_records_ready,current_sequence);
                if(b_valid && ((b_records_ready-current_sequence)>=RECORD_COUNT))
                    $fatal(1,"SSM B ring overrun: producer=%0d consumer=%0d",b_records_ready,current_sequence);
                if(c_valid && ((c_records_ready-current_sequence)>=RECORD_COUNT))
                    $fatal(1,"SSM C ring overrun: producer=%0d consumer=%0d",c_records_ready,current_sequence);
            end
        end
    end
    integer sim_continuous_record_transitions;
    integer sim_state_reset_records;
    integer sim_state_forward_d3_hits;
    integer sim_state_forward_d4_hits;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            sim_issue_groups <= 0;
            sim_continuous_record_transitions <= 0;
            sim_state_reset_records <= 0;
            sim_state_forward_d3_hits <= 0;
            sim_state_forward_d4_hits <= 0;
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

        if (spe_forward_d3_hit)
            sim_state_forward_d3_hits <= sim_state_forward_d3_hits + 1;
        if (spe_forward_d4_hit)
            sim_state_forward_d4_hits <= sim_state_forward_d4_hits + 1;

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
            $error("Fused Block0 SSM read/write collision on active record");
        if (frame_active && cfg_lut_we)
            $error("Fused Block0 SSM LUT write during active frame");
    end
`endif

    wire unused_cfg = cfg_lut_we ^ cfg_lut_addr[0] ^ cfg_lut_data[0];
endmodule

`default_nettype wire
