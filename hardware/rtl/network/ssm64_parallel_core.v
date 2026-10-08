`timescale 1ns / 1ps
`default_nettype none

// Same-cycle physical endpoint for high-fanout SSM control pulses.  RTL
// simulation remains a transparent connection.  Synthesis preserves only
// this single LUT1 buffer (not the consuming hierarchy), giving placement a
// real local driver without adding a clock of latency.
module ssm_local_control_buffer (
    input  wire control_in,
    output wire control_out
);
`ifdef SYNTHESIS
    (* dont_touch = "yes" *)
    LUT1 #(.INIT(2'b10)) u_control_lut (
        .I0(control_in), .O(control_out));
`else
    assign control_out = control_in;
`endif
endmodule

// -----------------------------------------------------------------------------
// Vendor Block Memory Generator adapter for one packed B/C record memory.
//
// Both synthesis and simulation instantiate the generated IP explicitly.  This
// is deliberate: relying on a tool-specific SYNTHESIS preprocessor macro left
// the behavioral array active in synthesis and mapped thousands of bits back
// into LUT fabric.  Generate all IP simulation/output products before XSim.
// -----------------------------------------------------------------------------
module ssm_bc_record_bram #(
    parameter integer DEPTH = 256,
    parameter integer ADDR_WIDTH = 8
) (
    input  wire                  clk,
    input  wire                  wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [127:0]          wr_data,
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [127:0]          rd_data
);
    generate
        if (DEPTH == 256 && ADDR_WIDTH == 8) begin : GEN_BRAM_256
            ssm_bc_bram_256x128 u_bram (
                .clka  (clk),
                .ena   (wr_en),
                .wea   ({wr_en}),
                .addra (wr_addr),
                .dina  (wr_data),
                .clkb  (clk),
                .enb   (rd_en),
                .addrb (rd_addr),
                .doutb (rd_data)
            );
        end else if (DEPTH == 1024 && ADDR_WIDTH == 10) begin : GEN_BRAM_1024
            ssm_bc_bram_1024x128 u_bram (
                .clka  (clk),
                .ena   (wr_en),
                .wea   ({wr_en}),
                .addra (wr_addr),
                .dina  (wr_data),
                .clkb  (clk),
                .enb   (rd_en),
                .addrb (rd_addr),
                .doutb (rd_data)
            );
        end else begin : GEN_UNSUPPORTED_BRAM_SHAPE
            // This branch is elaborated only for an unsupported parameter set.
            // Keep it self-contained so the Sources view does not show an
            // artificial unresolved-module question mark.
            initial $error("Unsupported ssm_bc_record_bram shape");
            assign rd_data = 128'd0;
        end
    endgenerate
endmodule

// U/DT are captured eight adjacent channels per beat.  Both Spa
// (256 records x 8 beats) and Spe (1024 records x 2 beats) therefore use the
// same 2048x64 simple-dual-port Block Memory Generator.
module ssm_udt_beat_bram (
    input  wire        clk,
    input  wire        wr_en,
    input  wire [10:0] wr_addr,
    input  wire [63:0] wr_data,
    input  wire        rd_en,
    input  wire [10:0] rd_addr,
    output wire [63:0] rd_data
);
    ssm_udt_bram_2048x64 u_bram (
        .clka  (clk),
        .ena   (wr_en),
        .wea   ({wr_en}),
        .addra (wr_addr),
        .dina  (wr_data),
        .clkb  (clk),
        .enb   (rd_en),
        .addrb (rd_addr),
        .doutb (rd_data)
    );
endmodule

// Four-read-port SSM nonlinear LUT built from two replicated true-dual-port
// Block Memory Generators.  All three phase-specific tables share one word:
//   [399:0]   Abar[16 states] (16 x UQ1.24, 25 bits each)
//   [418:400] K unsigned U19 (Q format is selected per SSM core)
//
// Only one field is consumed in each mutually-exclusive B/A/X phase.  The
// read address is therefore DT for B/A and U for X; four channels need four
// addresses, supplied by the two ports of each replicated copy.  Configuration
// writes a complete 419-bit word and is broadcast to both copies.
module ssm_lut_4r_bram (
    input  wire         clk,
    input  wire         cfg_we,
    input  wire [7:0]   cfg_addr,
    input  wire [418:0] cfg_data,
    input  wire         rd_en,
    input  wire [7:0]   rd_addr0,
    input  wire [7:0]   rd_addr1,
    input  wire [7:0]   rd_addr2,
    input  wire [7:0]   rd_addr3,
    output wire [418:0] rd_data0,
    output wire [418:0] rd_data1,
    output wire [418:0] rd_data2,
    output wire [418:0] rd_data3
);
    ssm_lut_bram_256x419 u_copy01 (
        .clka  (clk),
        .ena   (cfg_we | rd_en),
        .wea   ({cfg_we}),
        .addra (cfg_we ? cfg_addr : rd_addr0),
        .dina  (cfg_data),
        .douta (rd_data0),
        .clkb  (clk),
        .enb   (rd_en),
        .web   (1'b0),
        .addrb (rd_addr1),
        .dinb  (419'd0),
        .doutb (rd_data1)
    );

    ssm_lut_bram_256x419 u_copy23 (
        .clka  (clk),
        .ena   (cfg_we | rd_en),
        .wea   ({cfg_we}),
        .addra (cfg_we ? cfg_addr : rd_addr2),
        .dina  (cfg_data),
        .douta (rd_data2),
        .clkb  (clk),
        .enb   (rd_en),
        .web   (1'b0),
        .addrb (rd_addr3),
        .dinb  (419'd0),
        .doutb (rd_data3)
    );
endmodule

// Runtime scratch memories are shallow but extremely wide because one issue
// beat consumes all 4 channels x 16 states.  Explicit BMG instances trade the
// former asynchronous 64-read register muxes for banked BRAM primitives while
// preserving one group per clock.  Port A writes a completed group; port B
// synchronously prefetches the next group.
module ssm_scratch_2048_bram (
    input  wire          clk,
    input  wire          wr_en,
    input  wire [3:0]    wr_addr,
    input  wire [2047:0] wr_data,
    input  wire          rd_en,
    input  wire [3:0]    rd_addr,
    output wire [2047:0] rd_data
);
    ssm_scratch_bram_16x2048 u_bram (
        .clka(clk),
        .ena(wr_en),
        .wea({wr_en}),
        .addra(wr_addr),
        .dina(wr_data),
        .clkb(clk),
        .enb(rd_en),
        .addrb(rd_addr),
        .doutb(rd_data)
    );
endmodule

module ssm_scratch_4096_bram (
    input  wire          clk,
    input  wire          wr_en,
    input  wire [3:0]    wr_addr,
    input  wire [4095:0] wr_data,
    input  wire          rd_en,
    input  wire [3:0]    rd_addr,
    output wire [4095:0] rd_data
);
    ssm_scratch_bram_16x4096 u_bram (
        .clka(clk),
        .ena(wr_en),
        .wea({wr_en}),
        .addra(wr_addr),
        .dina(wr_data),
        .clkb(clk),
        .enb(rd_en),
        .addrb(rd_addr),
        .doutb(rd_data)
    );
endmodule

// Block0 has an unusually wide but very shallow recurrence scratchpad.  A
// Block RAM implementation wastes almost every physical row and therefore
// consumes dozens of RAMB36s merely to obtain 2048/4096 bits of read width.
// Infer exact-depth distributed RAM instead.  Registering only the read
// address preserves the original one-clock read contract without adding a
// WIDTH-bit output register.  One synchronous write plus one asynchronous
// read address is the native simple-dual-port LUTRAM pattern.
module ssm_block0_scratch_lutram #(
    parameter integer WIDTH = 2048,
    parameter integer DEPTH = 16,
    parameter integer ADDR_WIDTH = 4
) (
    input  wire                  clk,
    input  wire                  wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0]      wr_data,
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [WIDTH-1:0]      rd_data
);
    (* ram_style = "distributed" *) reg [WIDTH-1:0] memory [0:DEPTH-1];
    reg [ADDR_WIDTH-1:0] rd_addr_q;

    always @(posedge clk) begin
        if (wr_en)
            memory[wr_addr] <= wr_data;
        // The downstream valid pipeline qualifies rd_data, so an idle-cycle
        // address is harmless.  Updating every clock removes rd_en from the
        // CE pins of this very wide LUTRAM read-address register bank.
        rd_addr_q <= rd_addr;
    end

    assign rd_data = memory[rd_addr_q];
endmodule

// Block0 fused state-bank slice with physically local address/control.
//
// Keep the address/control endpoint local, but do not freeze the complete
// memory hierarchy.  The first local-bank revision used sixteen 32-bit
// slices per 512-bit channel bank and DONT_TOUCH on every slice.  That fixed
// the global address net, but duplicated too many address/one-hot registers
// and prevented normal LUTRAM packing.  The active configuration below uses
// eight 64-bit slices: address fanout remains local to 64 RAM bits while the
// register/control overhead is halved and Vivado may pack adjacent slices.
//
// wr_req/wr_addr are sampled one clock before the memory write.  wr_data is
// consumed on the following edge, exactly matching the old parent-level
// state_write_valid/state_write_group pipeline, so recurrence latency and
// same-group forwarding are unchanged.
module ssm_block0_scratch_lutram_slice #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 16,
    parameter integer ADDR_WIDTH = 4
) (
    input  wire                  clk,
    input  wire                  wr_req,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0]      wr_data,
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [WIDTH-1:0]      rd_data
);
    (* ram_style = "distributed" *)
    reg [WIDTH-1:0] memory [0:DEPTH-1];

    (* keep = "true", max_fanout = 64 *)
    reg [ADDR_WIDTH-1:0] local_wr_addr_q;
    (* keep = "true", max_fanout = 64 *)
    reg [DEPTH-1:0] local_wr_select_q;
    (* keep = "true", max_fanout = 64 *)
    reg [ADDR_WIDTH-1:0] local_rd_addr_q;

    // The selected one-hot bit is the local RAM write enable.  Keeping both
    // the encoded address and one-hot select local prevents either control
    // from becoming a bank-spanning high-fanout net.
    wire local_write_fire
        = local_wr_select_q[local_wr_addr_q];

    always @(posedge clk) begin
        if (wr_req) begin
            local_wr_addr_q <= wr_addr;
            local_wr_select_q
                <= {{(DEPTH-1){1'b0}}, 1'b1} << wr_addr;
        end else begin
            local_wr_select_q <= {DEPTH{1'b0}};
        end

        if (local_write_fire)
            memory[local_wr_addr_q] <= wr_data;

        // rd_data is consumed only when the matching valid metadata arrives.
        // Always clocking the local address removes the former global
        // rd_en/r0_valid network from every slice-register CE pin without
        // changing the one-clock LUTRAM read contract.
        local_rd_addr_q <= rd_addr;
    end

    assign rd_data = memory[local_rd_addr_q];
endmodule

// One 512-bit channel bank is physically divided into eight 64-bit slices.
// Every slice owns its write/read address registers and one-hot write select;
// the global group metadata consequently terminates at 64 small register
// clusters across the four channel banks instead of at thousands of RAMD32
// address pins.
module ssm_block0_state_channel_bank #(
    parameter integer WIDTH = 512,
    parameter integer DEPTH = 16,
    parameter integer ADDR_WIDTH = 4,
    parameter integer SLICE_WIDTH = 64
) (
    input  wire                  clk,
    input  wire                  wr_req,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0]      wr_data,
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [WIDTH-1:0]      rd_data
);
    localparam integer SLICE_COUNT = WIDTH / SLICE_WIDTH;

`ifndef SYNTHESIS
    initial begin
        if ((WIDTH % SLICE_WIDTH) != 0) begin
            $display("Block0 state bank width must divide into slices");
            $finish;
        end
    end
`endif

    genvar slice_index;
    generate
        for (slice_index = 0; slice_index < SLICE_COUNT;
             slice_index = slice_index + 1) begin : GEN_LOCAL_SLICE
            ssm_block0_scratch_lutram_slice #(
                .WIDTH(SLICE_WIDTH),
                .DEPTH(DEPTH),
                .ADDR_WIDTH(ADDR_WIDTH)
            ) u_slice (
                .clk(clk),
                .wr_req(wr_req),
                .wr_addr(wr_addr),
                .wr_data(wr_data[
                    slice_index*SLICE_WIDTH +: SLICE_WIDTH]),
                .rd_en(rd_en),
                .rd_addr(rd_addr),
                .rd_data(rd_data[
                    slice_index*SLICE_WIDTH +: SLICE_WIDTH])
            );
        end
    endgenerate
endmodule

// -----------------------------------------------------------------------------
// 64-state-cell parallel selective-scan core
//
// One issue group contains four channels and all sixteen states:
//     4 channels x 16 states = 64 state cells.
//
// A single bank of 64 exact two-DSP multipliers is reused in three phases:
//     PHASE_B : unsigned K[dt] x signed B[state]
//     PHASE_A : unsigned Abar[dt,state] x signed h[channel,state]
//     PHASE_X : signed KB[channel,state] x signed U_int8[channel]
//
// ssm_mult_2dsp preserves the former three-clock result contract.  It uses
// two native 27x18 products for Abar*state and one of those products for the
// narrow K*B / KB*U phases, avoiding the generic four-DSP 33x32 mapping.
// -----------------------------------------------------------------------------
module ssm64_parallel_core_block0 #(
    parameter integer BLOCK_ID = 0,
    parameter integer IS_SPE = 0,
    parameter integer CHANNEL_COUNT = 64,
    parameter integer RECORD_COUNT = 256,
    parameter integer GROUP_COUNT = 16,
    parameter integer K_FRACTION_BITS = 24,
    parameter integer RESET_PERIOD = 0
) (
    input  wire                 clk,
    input  wire                 rst_n,

    // Runtime input capture.  record_addr is pixel for Spa and
    // pixel*4+token for Spe.  Eight adjacent channels arrive per beat.
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

    // Complete nonlinear LUT word configuration.  Packing at the interface
    // prevents a 419-bit read-modify-write shadow array from being synthesized
    // in LUTRAM merely to support 25-bit partial Abar writes.
    input  wire                 cfg_lut_we,
    input  wire [7:0]           cfg_lut_addr,
    input  wire [418:0]         cfg_lut_data,

    // frame_start arms a new frame before its first runtime input.  The core
    // then starts each record as soon as its U/DT/B/C completion watermarks
    // have all reached that record.  compute_start is retained as a legacy
    // arm pulse for the existing load-all-then-compute testbenches; it does
    // not clear capture watermarks.
    input  wire                 frame_start,
    input  wire                 compute_start,

    // Four Q24 readout values per beat, record-major/channel-major.
    output reg                  out_valid,
    output reg [9:0]            out_record_addr,
    output reg [5:0]            out_channel_base,
    output reg [191:0]          out_y_q24,
    output reg                  done,
    output wire                 busy
);
    localparam integer STATE_COUNT = 16;
    localparam integer RECORD_ADDR_WIDTH = (RECORD_COUNT <= 256) ? 8 : 10;
    localparam integer GROUP_ADDR_WIDTH = (GROUP_COUNT <= 4) ? 2 : 4;
    localparam integer CHANNEL_BEATS = CHANNEL_COUNT / 8;
    localparam integer LAST_CHANNEL_BASE = CHANNEL_COUNT - 8;
    // Only block0 Spe has four groups per record.  Its per-record PRE/WAIT
    // overhead is a large fraction of the twelve useful B/A/X issue clocks,
    // so retain an active-record cache and prefetch the following record while
    // the current B phase is issuing.  Spa keeps the proven legacy schedule.
    localparam integer CONTINUOUS_SPE
        = ((IS_SPE != 0) && (CHANNEL_COUNT == 16) && (GROUP_COUNT == 4));
    // Spa has sixteen groups per phase.  Its U/DT read port is idle for the
    // final two X requests, so use those clocks to fetch B0 of record r+1.
    // This removes WAIT_X/PRE_B_* between ready records without caching the
    // complete 512-bit Spa operands in registers.
    localparam integer CONTINUOUS_SPA
        = ((IS_SPE == 0) && (CHANNEL_COUNT == 64) && (GROUP_COUNT == 16));

    // Binary encoding is smaller than one-hot for this seven-state FSM.
    // PRE_B_* supplies the first record/phase operands.  Thereafter the final
    // two issue groups of PHASE_B/PHASE_A prefetch group zero for the following
    // phase, allowing B -> A -> X to issue continuously without bubbles.  The
    // three-cycle multiplier drain is tracked by phase/group metadata.
    localparam [2:0] ST_IDLE        = 3'd0;
    localparam [2:0] ST_PRE_B_INPUT = 3'd1;
    localparam [2:0] ST_PRE_B_LUT   = 3'd2;
    localparam [2:0] ST_ISSUE_B     = 3'd3;
    localparam [2:0] ST_ISSUE_A     = 3'd4;
    localparam [2:0] ST_ISSUE_X     = 3'd5;
    localparam [2:0] ST_WAIT_X      = 3'd6;

    localparam [1:0] PHASE_B = 2'd0;
    localparam [1:0] PHASE_A = 2'd1;
    localparam [1:0] PHASE_X = 2'd2;

    reg [2:0] fsm_state;
    reg [GROUP_ADDR_WIDTH-1:0] issue_group;

    // B and C arrive as complete sixteen-state records.  Store that native
    // 128-bit record instead of flattening it into byte arrays with 64
    // concurrent asynchronous reads.  The latter forces RAM replication and
    // large LUT mux trees; the synchronous record memories infer Block RAM.
    wire [127:0] b_record_word;
    wire [127:0] c_record_word;
    reg [RECORD_ADDR_WIDTH-1:0] current_record;
    reg frame_active;
    wire runtime_record_read_en;
    wire [RECORD_ADDR_WIDTH-1:0] runtime_record_read_addr;

    // Two-record operand cache used only by the continuous Spe schedule.
    // U/DT need two 64-bit BRAM reads; B/C are captured alongside them.  The
    // active values remain stable while the same ports fetch the next record.
    reg [127:0] active_u_record;
    reg [127:0] active_dt_record;
    reg [127:0] active_b_record;
    reg [127:0] active_c_record;
    reg [127:0] next_u_record;
    reg [127:0] next_dt_record;
    reg [127:0] next_b_record;
    reg [127:0] next_c_record;
    reg next_cache_valid;
    reg next_prefetch_started;
    reg cache_read_en;
    reg [RECORD_ADDR_WIDTH-1:0] cache_read_record;
    reg cache_read_beat;
    reg cache_read_valid_d;
    reg [RECORD_ADDR_WIDTH-1:0] cache_read_record_d;
    reg cache_read_beat_d;
    reg spa_prefetch_valid;
    reg [127:0] current_c_record;
    wire continuous_switch;
    wire spa_prefetch_request;
    wire spa_read_next_record;

    // Each producer is record-major and monotonic.  These four prefix
    // watermarks are therefore equivalent to a ready bit for every record in
    // the completed prefix, without synthesizing/resetting thousands of flag
    // flops.  U and DT become complete on their final 8-channel beat; B and C
    // already arrive as complete 16-state records.
    reg [RECORD_ADDR_WIDTH:0] u_records_ready;
    reg [RECORD_ADDR_WIDTH:0] dt_records_ready;
    reg [RECORD_ADDR_WIDTH:0] b_records_ready;
    reg [RECORD_ADDR_WIDTH:0] c_records_ready;
    wire u_record_complete;
    wire dt_record_complete;
    wire current_record_ready;
    wire next_record_ready;
    wire record_zero_ready;
    wire [RECORD_ADDR_WIDTH:0] current_record_extended;
    wire [RECORD_ADDR_WIDTH:0] next_record_extended;

    assign u_record_complete = u_valid
        && (u_channel_base == LAST_CHANNEL_BASE);
    assign dt_record_complete = dt_valid
        && (dt_channel_base == LAST_CHANNEL_BASE);
    assign current_record_extended = {1'b0, current_record};
    assign next_record_extended = current_record_extended + 1'b1;
    assign current_record_ready
        = (u_records_ready  > current_record_extended)
       && (dt_records_ready > current_record_extended)
       && (b_records_ready  > current_record_extended)
       && (c_records_ready  > current_record_extended);
    assign next_record_ready
        = (u_records_ready  > next_record_extended)
       && (dt_records_ready > next_record_extended)
       && (b_records_ready  > next_record_extended)
       && (c_records_ready  > next_record_extended);
    assign record_zero_ready
        = (u_records_ready  != 0)
       && (dt_records_ready != 0)
       && (b_records_ready  != 0)
       && (c_records_ready  != 0);

    // The next record is fetched during B0/B1.  By the final X group all four
    // operands are cached, allowing X(last,r) -> B0(r+1) with no PRE/WAIT
    // bubble.  If the producer has not completed the next record in time, the
    // original WAIT_X/PRE path remains the safe fallback.
    assign spa_prefetch_request
        = (CONTINUOUS_SPA != 0)
       && (fsm_state == ST_ISSUE_X)
       && (issue_group == GROUP_COUNT-2)
       && (current_record != RECORD_COUNT-1)
       && next_record_ready;
    assign spa_read_next_record
        = (CONTINUOUS_SPA != 0)
       && (fsm_state == ST_ISSUE_X)
       && ((issue_group == GROUP_COUNT-2)
           || (issue_group == GROUP_COUNT-1))
       && (spa_prefetch_request || spa_prefetch_valid);
    assign continuous_switch
        = (fsm_state == ST_ISSUE_X)
       && (issue_group == GROUP_COUNT-1)
       && (current_record != RECORD_COUNT-1)
       && (((CONTINUOUS_SPE != 0) && next_cache_valid)
           || ((CONTINUOUS_SPA != 0) && spa_prefetch_valid));

    always @* begin
        cache_read_en = 1'b0;
        cache_read_record = current_record;
        cache_read_beat = 1'b0;
        if (CONTINUOUS_SPE != 0) begin
            case (fsm_state)
                ST_PRE_B_INPUT: begin
                    cache_read_en = 1'b1;
                    cache_read_record = current_record;
                    cache_read_beat = 1'b0;
                end
                ST_PRE_B_LUT: begin
                    cache_read_en = 1'b1;
                    cache_read_record = current_record;
                    cache_read_beat = 1'b1;
                end
                ST_ISSUE_B: begin
                    if ((issue_group == 0)
                        && (current_record != RECORD_COUNT-1)
                        && next_record_ready) begin
                        cache_read_en = 1'b1;
                        cache_read_record = current_record + 1'b1;
                        cache_read_beat = 1'b0;
                    end else if ((issue_group == 1)
                                 && next_prefetch_started) begin
                        cache_read_en = 1'b1;
                        cache_read_record = current_record + 1'b1;
                        cache_read_beat = 1'b1;
                    end
                end
                default: begin
                    cache_read_en = 1'b0;
                    cache_read_record = current_record;
                    cache_read_beat = 1'b0;
                end
            endcase
        end
    end

    assign runtime_record_read_en
        = (CONTINUOUS_SPE != 0) ? cache_read_en : (fsm_state != ST_IDLE);
    assign runtime_record_read_addr
        = (CONTINUOUS_SPE != 0) ? cache_read_record
        : spa_read_next_record ? current_record + 1'b1
        : current_record;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            spa_prefetch_valid <= 1'b0;
            current_c_record <= 128'd0;
        end else begin
            if (continuous_switch || (fsm_state == ST_WAIT_X)
                || (fsm_state == ST_IDLE))
                spa_prefetch_valid <= 1'b0;
            else if (spa_prefetch_request)
                spa_prefetch_valid <= 1'b1;

            // PRE_B_INPUT has already issued the synchronous B/C read when
            // PRE_B_LUT samples it.  At a continuous X(last) edge the BRAM
            // output is record r+1, while the old hold still tags X(last,r).
            if ((CONTINUOUS_SPA != 0)
                && ((fsm_state == ST_PRE_B_LUT) || continuous_switch))
                current_c_record <= c_record_word;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            u_records_ready <= {(RECORD_ADDR_WIDTH+1){1'b0}};
            dt_records_ready <= {(RECORD_ADDR_WIDTH+1){1'b0}};
            b_records_ready <= {(RECORD_ADDR_WIDTH+1){1'b0}};
            c_records_ready <= {(RECORD_ADDR_WIDTH+1){1'b0}};
        end else begin
            if (u_record_complete
                && (u_records_ready < RECORD_COUNT))
                u_records_ready <= u_records_ready + 1'b1;
            if (dt_record_complete
                && (dt_records_ready < RECORD_COUNT))
                dt_records_ready <= dt_records_ready + 1'b1;
            if (b_valid && (b_records_ready < RECORD_COUNT))
                b_records_ready <= b_records_ready + 1'b1;
            if (c_valid && (c_records_ready < RECORD_COUNT))
                c_records_ready <= c_records_ready + 1'b1;
        end
    end

`ifndef SYNTHESIS
    // Watermarks are valid only while each component stream is ordered and
    // gap-free.  Fail loudly in simulation if an upstream refactor violates
    // that contract instead of silently consuming an incomplete record.
    reg [5:0] sim_expected_u_channel_base;
    reg [5:0] sim_expected_dt_channel_base;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            sim_expected_u_channel_base <= 6'd0;
            sim_expected_dt_channel_base <= 6'd0;
        end else begin
            if (u_valid
                && (u_record_addr[RECORD_ADDR_WIDTH-1:0]
                    != u_records_ready[RECORD_ADDR_WIDTH-1:0]))
                $error("SSM U records are not monotonic/gap-free");
            if (dt_valid
                && (dt_record_addr[RECORD_ADDR_WIDTH-1:0]
                    != dt_records_ready[RECORD_ADDR_WIDTH-1:0]))
                $error("SSM DT records are not monotonic/gap-free");
            if (u_valid && (u_channel_base != sim_expected_u_channel_base))
                $error("SSM U channel beats are not 0,8,...,LAST");
            if (dt_valid && (dt_channel_base != sim_expected_dt_channel_base))
                $error("SSM DT channel beats are not 0,8,...,LAST");
            if (u_valid) begin
                if (u_record_complete)
                    sim_expected_u_channel_base <= 6'd0;
                else
                    sim_expected_u_channel_base
                        <= sim_expected_u_channel_base + 6'd8;
            end
            if (dt_valid) begin
                if (dt_record_complete)
                    sim_expected_dt_channel_base <= 6'd0;
                else
                    sim_expected_dt_channel_base
                        <= sim_expected_dt_channel_base + 6'd8;
            end
            if (b_valid
                && (b_record_addr[RECORD_ADDR_WIDTH-1:0]
                    != b_records_ready[RECORD_ADDR_WIDTH-1:0]))
                $error("SSM B records are not monotonic/gap-free");
            if (c_valid
                && (c_record_addr[RECORD_ADDR_WIDTH-1:0]
                    != c_records_ready[RECORD_ADDR_WIDTH-1:0]))
                $error("SSM C records are not monotonic/gap-free");
            if (runtime_record_read_en
                && ((u_valid
                     && (u_record_addr[RECORD_ADDR_WIDTH-1:0]
                         == runtime_record_read_addr))
                    || (dt_valid
                        && (dt_record_addr[RECORD_ADDR_WIDTH-1:0]
                            == runtime_record_read_addr))
                    || (b_valid
                        && (b_record_addr[RECORD_ADDR_WIDTH-1:0]
                            == runtime_record_read_addr))
                    || (c_valid
                        && (c_record_addr[RECORD_ADDR_WIDTH-1:0]
                            == runtime_record_read_addr))))
                $error("SSM attempted to write the record being computed");
            if (frame_active && cfg_lut_we)
                $error("SSM nonlinear LUT write during an active frame");
        end
    end
`endif
    // Capture and compute may overlap, but a record is never read until all of
    // its four input components have completed.  The first PRE_* state gives
    // the simple-dual-port BRAMs a full synchronous-read setup cycle.
    ssm_bc_record_bram #(
        .DEPTH((RECORD_COUNT <= 256) ? 256 : 1024),
        .ADDR_WIDTH(RECORD_ADDR_WIDTH)
    ) u_b_record_bram (
        .clk(clk),
        .wr_en(b_valid),
        .wr_addr(b_record_addr[RECORD_ADDR_WIDTH-1:0]),
        .wr_data(b_data),
        .rd_en(runtime_record_read_en),
        .rd_addr(runtime_record_read_addr),
        .rd_data(b_record_word)
    );

    ssm_bc_record_bram #(
        .DEPTH((RECORD_COUNT <= 256) ? 256 : 1024),
        .ADDR_WIDTH(RECORD_ADDR_WIDTH)
    ) u_c_record_bram (
        .clk(clk),
        .wr_en(c_valid),
        .wr_addr(c_record_addr[RECORD_ADDR_WIDTH-1:0]),
        .wr_data(c_data),
        .rd_en(runtime_record_read_en),
        .rd_addr(runtime_record_read_addr),
        .rd_data(c_record_word)
    );

    wire [418:0] lane_lut_word0;
    wire [418:0] lane_lut_word1;
    wire [418:0] lane_lut_word2;
    wire [418:0] lane_lut_word3;
    reg [7:0] lut_read_addr0;
    reg [7:0] lut_read_addr1;
    reg [7:0] lut_read_addr2;
    reg [7:0] lut_read_addr3;
    reg lut_read_en;

    wire [2047:0] state_group_read;
    wire [2047:0] bbar_group_read;
    wire [4095:0] ah_group_read;
    wire [2047:0] state_group_write;
    wire [2047:0] bbar_group_write;
    wire [4095:0] ah_group_write;
    wire state_group_write_en;
    wire bbar_group_write_en;
    wire ah_group_write_en;
    reg state_group_read_en;
    reg bbar_group_read_en;
    wire ah_group_read_en;
    wire [GROUP_ADDR_WIDTH-1:0] scratch_read_addr;
    wire [GROUP_ADDR_WIDTH-1:0] ah_scratch_read_addr;
    wire [GROUP_ADDR_WIDTH-1:0] scratch_write_addr;

    reg mult_valid_d0;
    reg mult_valid_d1;
    reg mult_valid_d2;
    reg [1:0] mult_phase_d0;
    reg [1:0] mult_phase_d1;
    reg [1:0] mult_phase_d2;
    reg [GROUP_ADDR_WIDTH-1:0] mult_group_d0;
    reg [GROUP_ADDR_WIDTH-1:0] mult_group_d1;
    reg [GROUP_ADDR_WIDTH-1:0] mult_group_d2;
    reg [RECORD_ADDR_WIDTH-1:0] mult_record_d0;
    reg [RECORD_ADDR_WIDTH-1:0] mult_record_d1;
    reg [RECORD_ADDR_WIDTH-1:0] mult_record_d2;
    reg [127:0] mult_c_record_d0;
    reg [127:0] mult_c_record_d1;
    reg [127:0] mult_c_record_d2;
    reg y_product_valid;
    reg [RECORD_ADDR_WIDTH-1:0] y_product_record;
    reg [GROUP_ADDR_WIDTH-1:0] y_product_group;
    reg y_valid_l1;
    reg y_valid_l2;
    reg y_valid_l3;
    reg y_valid_l4;
    assign busy = frame_active || (fsm_state != ST_IDLE) || y_product_valid
               || y_valid_l1 || y_valid_l2 || y_valid_l3 || y_valid_l4
               || out_valid;

    wire issue_fire;
    reg [1:0] issue_phase;
    assign issue_fire = (fsm_state == ST_ISSUE_B)
                     || (fsm_state == ST_ISSUE_A)
                     || (fsm_state == ST_ISSUE_X);
    always @* begin
        case (fsm_state)
            ST_ISSUE_A: issue_phase = PHASE_A;
            ST_ISSUE_X: issue_phase = PHASE_X;
            default:    issue_phase = PHASE_B;
        endcase
    end

    // Packed runtime memories feed a second, synchronous nonlinear-LUT BRAM.
    // PRE_B_INPUT fetches U/DT group zero; PRE_B_LUT uses that word to fetch
    // nonlinear-LUT group zero.  During ISSUE group g, U/DT fetches g+2 while
    // the nonlinear LUT fetches g+1.  At g=last-1 the U/DT request wraps to the
    // following phase's group0; at g=last that word drives the group0 LUT and
    // scratch read.  Both synchronous-memory latencies are therefore hidden.
    wire [10:0] u_write_addr;
    wire [10:0] dt_write_addr;
    reg [GROUP_ADDR_WIDTH-1:0] input_request_group;
    reg [GROUP_ADDR_WIDTH-1:0] lut_source_group;
    reg [1:0] lut_source_phase;
    wire [10:0] input_read_addr;
    wire [63:0] u_read_word;
    wire [63:0] dt_read_word;
    assign u_write_addr = u_record_addr*CHANNEL_BEATS
                        + (u_channel_base >> 3);
    assign dt_write_addr = dt_record_addr*CHANNEL_BEATS
                         + (dt_channel_base >> 3);
    assign input_read_addr
        = (CONTINUOUS_SPE != 0)
        ? (cache_read_record*CHANNEL_BEATS + cache_read_beat)
        : ((spa_read_next_record ? current_record + 1'b1 : current_record)
           *CHANNEL_BEATS + (input_request_group >> 1));
    assign scratch_read_addr = lut_source_group;
    assign scratch_write_addr = mult_group_d2;
    // Bbar is an operand of the PHASE_X multiplier and must be prefetched for
    // the issue edge.  Ah is different: it is added only when that multiplier
    // result returns three clocks later.  Read Ah using the d1 metadata so its
    // one-clock scratch output is aligned with mult_p/d2, without registering a
    // 4096-bit word or reducing the one-group-per-cycle throughput.
    assign ah_group_read_en
        = mult_valid_d1 && (mult_phase_d1 == PHASE_X);
    assign ah_scratch_read_addr = mult_group_d1;

    always @* begin
        state_group_read_en = 1'b0;
        bbar_group_read_en = 1'b0;
        case (fsm_state)
            // During the final B issue, lut_source_group has already switched
            // to A/group0.  Read state/group0 in parallel with that LUT read.
            ST_ISSUE_B: state_group_read_en
                = (issue_group == GROUP_COUNT-1);
            // The final A issue similarly prefetches Bbar/group0 for X.
            ST_ISSUE_A: begin
                state_group_read_en = (issue_group != GROUP_COUNT-1);
                bbar_group_read_en = (issue_group == GROUP_COUNT-1);
            end
            ST_ISSUE_X: bbar_group_read_en
                = (issue_group != GROUP_COUNT-1);
            default: begin
                state_group_read_en = 1'b0;
                bbar_group_read_en = 1'b0;
            end
        endcase
    end

    ssm_block0_scratch_lutram #(
        .WIDTH(2048),
        .DEPTH(GROUP_COUNT),
        .ADDR_WIDTH(GROUP_ADDR_WIDTH)
    ) u_state_group_lutram (
        .clk(clk),
        .wr_en(state_group_write_en),
        .wr_addr(scratch_write_addr),
        .wr_data(state_group_write),
        .rd_en(state_group_read_en),
        .rd_addr(scratch_read_addr),
        .rd_data(state_group_read)
    );

    ssm_block0_scratch_lutram #(
        .WIDTH(2048),
        .DEPTH(GROUP_COUNT),
        .ADDR_WIDTH(GROUP_ADDR_WIDTH)
    ) u_bbar_group_lutram (
        .clk(clk),
        .wr_en(bbar_group_write_en),
        .wr_addr(scratch_write_addr),
        .wr_data(bbar_group_write),
        .rd_en(bbar_group_read_en),
        .rd_addr(scratch_read_addr),
        .rd_data(bbar_group_read)
    );

    ssm_block0_scratch_lutram #(
        .WIDTH(4096),
        .DEPTH(GROUP_COUNT),
        .ADDR_WIDTH(GROUP_ADDR_WIDTH)
    ) u_ah_group_lutram (
        .clk(clk),
        .wr_en(ah_group_write_en),
        .wr_addr(scratch_write_addr),
        .wr_data(ah_group_write),
        .rd_en(ah_group_read_en),
        .rd_addr(ah_scratch_read_addr),
        .rd_data(ah_group_read)
    );

    always @* begin
        input_request_group = {GROUP_ADDR_WIDTH{1'b0}};
        case (fsm_state)
            ST_PRE_B_INPUT: input_request_group
                = {GROUP_ADDR_WIDTH{1'b0}};

            ST_PRE_B_LUT: input_request_group
                = {{(GROUP_ADDR_WIDTH-1){1'b0}}, 1'b1};

            ST_ISSUE_B,
            ST_ISSUE_A,
            ST_ISSUE_X: begin
                if ((issue_group != GROUP_COUNT-1)
                    && (issue_group != GROUP_COUNT-2))
                    input_request_group = issue_group + 2'd2;
            end

            default: input_request_group
                = {GROUP_ADDR_WIDTH{1'b0}};
        endcase
    end

    always @* begin
        lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
        lut_source_phase = PHASE_B;
        case (fsm_state)
            ST_PRE_B_LUT: begin
                lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
                lut_source_phase = PHASE_B;
            end
            ST_ISSUE_B: begin
                if (issue_group == GROUP_COUNT-1) begin
                    // input_request_group returned to group0 one cycle ago;
                    // use that DT beat to fetch A/group0 now.
                    lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
                    lut_source_phase = PHASE_A;
                end else begin
                    lut_source_group = issue_group + 1'b1;
                    lut_source_phase = PHASE_B;
                end
            end
            ST_ISSUE_A: begin
                if (issue_group == GROUP_COUNT-1) begin
                    // Fetch X/group0 from U while the final A operation enters
                    // the multiplier pipeline.
                    lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
                    lut_source_phase = PHASE_X;
                end else begin
                    lut_source_group = issue_group + 1'b1;
                    lut_source_phase = PHASE_A;
                end
            end
            ST_ISSUE_X: begin
                if (continuous_switch) begin
                    // X(last,r) simultaneously fetches B/group0 for r+1.
                    lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
                    lut_source_phase = PHASE_B;
                end else begin
                    lut_source_group = issue_group + 1'b1;
                    lut_source_phase = PHASE_X;
                end
            end
            default: begin
                lut_source_group = {GROUP_ADDR_WIDTH{1'b0}};
                lut_source_phase = PHASE_B;
            end
        endcase
    end

    ssm_udt_beat_bram u_u_bram (
        .clk(clk),
        .wr_en(u_valid),
        .wr_addr(u_write_addr),
        .wr_data(u_data),
        .rd_en(runtime_record_read_en),
        .rd_addr(input_read_addr),
        .rd_data(u_read_word)
    );

    ssm_udt_beat_bram u_dt_bram (
        .clk(clk),
        .wr_en(dt_valid),
        .wr_addr(dt_write_addr),
        .wr_data(dt_data),
        .rd_en(runtime_record_read_en),
        .rd_addr(input_read_addr),
        .rd_data(dt_read_word)
    );

    // Delay the request metadata by the BRAM's one-clock read latency, then
    // fill either the active record or the look-ahead record.  Promotion is
    // atomic at X(last), so a returning X result can still use the old C copy.
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            active_u_record <= 128'd0;
            active_dt_record <= 128'd0;
            active_b_record <= 128'd0;
            active_c_record <= 128'd0;
            next_u_record <= 128'd0;
            next_dt_record <= 128'd0;
            next_b_record <= 128'd0;
            next_c_record <= 128'd0;
            next_cache_valid <= 1'b0;
            next_prefetch_started <= 1'b0;
            cache_read_valid_d <= 1'b0;
            cache_read_record_d <= {RECORD_ADDR_WIDTH{1'b0}};
            cache_read_beat_d <= 1'b0;
        end else begin
            cache_read_valid_d <= cache_read_en;
            if (cache_read_en) begin
                cache_read_record_d <= cache_read_record;
                cache_read_beat_d <= cache_read_beat;
            end

            if ((CONTINUOUS_SPE != 0)
                && (fsm_state == ST_ISSUE_B)
                && (issue_group == 0)
                && cache_read_en)
                next_prefetch_started <= 1'b1;

            if ((CONTINUOUS_SPE != 0) && cache_read_valid_d) begin
                if (cache_read_record_d == current_record) begin
                    case (cache_read_beat_d)
                        1'b0: begin
                            active_u_record[63:0] <= u_read_word;
                            active_dt_record[63:0] <= dt_read_word;
                        end
                        default: begin
                            active_u_record[127:64] <= u_read_word;
                            active_dt_record[127:64] <= dt_read_word;
                        end
                    endcase
                    active_b_record <= b_record_word;
                    active_c_record <= c_record_word;
                end else begin
                    case (cache_read_beat_d)
                        1'b0: begin
                            next_u_record[63:0] <= u_read_word;
                            next_dt_record[63:0] <= dt_read_word;
                        end
                        default: begin
                            next_u_record[127:64] <= u_read_word;
                            next_dt_record[127:64] <= dt_read_word;
                            next_cache_valid <= 1'b1;
                        end
                    endcase
                    next_b_record <= b_record_word;
                    next_c_record <= c_record_word;
                end
            end

            if (continuous_switch) begin
                active_u_record <= next_u_record;
                active_dt_record <= next_dt_record;
                active_b_record <= next_b_record;
                active_c_record <= next_c_record;
                next_cache_valid <= 1'b0;
                next_prefetch_started <= 1'b0;
            end
        end
    end

    // Convert signed INT8 code to LUT address code+128 without an adder.
    function [7:0] int8_lut_address;
        input [7:0] signed_code;
        begin
            int8_lut_address = {~signed_code[7], signed_code[6:0]};
        end
    endfunction

    reg [399:0] lane_abar_word [0:3];
    reg [18:0] lane_k [0:3];
    reg [31:0] selected_u_group;
    reg [31:0] selected_dt_group;
    // The nonlinear ROM and this register see the same request edge.  Hence
    // lut_u_group_q is aligned with lane_lut_word* when that word is consumed.
    // Keeping U as INT8 is the v5 contract and avoids a 32-bit U-Q24 ROM field.
    reg [31:0] lut_u_group_q;

    always @(posedge clk) begin
        if (!rst_n || frame_start)
            lut_u_group_q <= 32'd0;
        else if (lut_read_en)
            lut_u_group_q <= selected_u_group;
    end

    always @* begin
        if ((CONTINUOUS_SPE != 0) && continuous_switch) begin
            selected_u_group = next_u_record[31:0];
            selected_dt_group = next_dt_record[31:0];
        end else if ((CONTINUOUS_SPE != 0)
                     && (fsm_state != ST_PRE_B_LUT)) begin
            selected_u_group
                = active_u_record[lut_source_group*32 +: 32];
            selected_dt_group
                = active_dt_record[lut_source_group*32 +: 32];
        end else if (lut_source_group[0]) begin
            selected_u_group = u_read_word[63:32];
            selected_dt_group = dt_read_word[63:32];
        end else begin
            selected_u_group = u_read_word[31:0];
            selected_dt_group = dt_read_word[31:0];
        end

        lut_read_en = 1'b0;
        case (fsm_state)
            ST_PRE_B_LUT: lut_read_en = 1'b1;
            // The last B/A issue performs the following phase's group0 read.
            ST_ISSUE_B,
            ST_ISSUE_A: lut_read_en = 1'b1;
            ST_ISSUE_X: lut_read_en
                = (issue_group != GROUP_COUNT-1) || continuous_switch;
            default: lut_read_en = 1'b0;
        endcase

        case (lut_source_phase)
            PHASE_X: begin
                lut_read_addr0 = int8_lut_address(selected_u_group[7:0]);
                lut_read_addr1 = int8_lut_address(selected_u_group[15:8]);
                lut_read_addr2 = int8_lut_address(selected_u_group[23:16]);
                lut_read_addr3 = int8_lut_address(selected_u_group[31:24]);
            end
            default: begin
                lut_read_addr0 = int8_lut_address(selected_dt_group[7:0]);
                lut_read_addr1 = int8_lut_address(selected_dt_group[15:8]);
                lut_read_addr2 = int8_lut_address(selected_dt_group[23:16]);
                lut_read_addr3 = int8_lut_address(selected_dt_group[31:24]);
            end
        endcase

        lane_abar_word[0] = lane_lut_word0[399:0];
        lane_abar_word[1] = lane_lut_word1[399:0];
        lane_abar_word[2] = lane_lut_word2[399:0];
        lane_abar_word[3] = lane_lut_word3[399:0];
        lane_k[0] = lane_lut_word0[418:400];
        lane_k[1] = lane_lut_word1[418:400];
        lane_k[2] = lane_lut_word2[418:400];
        lane_k[3] = lane_lut_word3[418:400];
    end

    // Deployment LUT is immutable after QAT/export.  Two replicated TDP ROMs
    // provide four channel addresses without a configuration shadow array.
    mamba_ssm_lut_rom_4r #(
        .BLOCK_ID(BLOCK_ID),
        .IS_SPE(IS_SPE)
    ) u_nonlinear_lut_bram (
        .clk(clk),
        .en(lut_read_en),
        .addr0(lut_read_addr0),
        .addr1(lut_read_addr1),
        .addr2(lut_read_addr2),
        .addr3(lut_read_addr3),
        .data0(lane_lut_word0),
        .data1(lane_lut_word1),
        .data2(lane_lut_word2),
        .data3(lane_lut_word3)
    );

    reg signed [32:0] mult_a [0:63];
    reg signed [31:0] mult_b [0:63];
    integer operand_channel_lane;
    integer operand_state_lane;
    integer operand_flat_lane;
    reg reset_state_for_record;

    always @* begin
        // Stale BRAM contents are ignored at the first record of every frame,
        // so no multi-cycle scratch-memory clear pass is required.
        reset_state_for_record
            = (current_record == {RECORD_ADDR_WIDTH{1'b0}});
        case (RESET_PERIOD)
            4: reset_state_for_record = (current_record[1:0] == 2'd0);
            default: reset_state_for_record
                = (current_record == {RECORD_ADDR_WIDTH{1'b0}});
        endcase

        for (operand_channel_lane = 0; operand_channel_lane < 4;
             operand_channel_lane = operand_channel_lane + 1) begin
            for (operand_state_lane = 0;
                 operand_state_lane < STATE_COUNT;
                 operand_state_lane = operand_state_lane + 1) begin
                operand_flat_lane
                    = operand_channel_lane*STATE_COUNT + operand_state_lane;

                case (issue_phase)
                    PHASE_A: begin
                        mult_a[operand_flat_lane]
                            = {8'd0,
                               lane_abar_word[operand_channel_lane]
                                   [operand_state_lane*25 +: 25]};
                        case (reset_state_for_record)
                            1'b1: mult_b[operand_flat_lane] = 32'sd0;
                            default:
                                mult_b[operand_flat_lane]
                                    = state_group_read[
                                        operand_flat_lane*32 +: 32];
                        endcase
                    end
                    PHASE_X: begin
                        mult_a[operand_flat_lane]
                            = {{1{bbar_group_read[
                                     operand_flat_lane*32+31]}},
                               bbar_group_read[
                                   operand_flat_lane*32 +: 32]};
                        mult_b[operand_flat_lane]
                            = {{24{lut_u_group_q[
                                  operand_channel_lane*8+7]}},
                               lut_u_group_q[
                                  operand_channel_lane*8 +: 8]};
                    end
                    default: begin
                        mult_a[operand_flat_lane]
                            = {14'd0, lane_k[operand_channel_lane]};
                        if (CONTINUOUS_SPE != 0)
                            mult_b[operand_flat_lane]
                                = {{24{active_b_record[
                                      operand_state_lane*8+7]}},
                                   active_b_record[
                                      operand_state_lane*8 +: 8]};
                        else
                            mult_b[operand_flat_lane]
                                = {{24{b_record_word[
                                      operand_state_lane*8+7]}},
                                   b_record_word[
                                      operand_state_lane*8 +: 8]};
                    end
                endcase
            end
        end
    end

    wire signed [64:0] mult_p [0:63];
    genvar multiplier_lane;
    generate
        for (multiplier_lane = 0; multiplier_lane < 64;
             multiplier_lane = multiplier_lane + 1) begin : GEN_SSM_MULTIPLIER
            ssm_mult_2dsp u_multiplier (
                .CLK(clk),
                .VALID(issue_fire),
                .WIDE_MODE(issue_phase == PHASE_A),
                .A(mult_a[multiplier_lane]),
                .B(mult_b[multiplier_lane]),
                .P(mult_p[multiplier_lane])
            );
        end
    endgenerate

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

            if (value < 0)
                signed_rounded = -rounded;
            else
                signed_rounded = rounded;

            // A valid signed-32 result is exactly a sign extension of bit 31.
            // This replaces two wide magnitude comparators with one reduction
            // comparison and a sign-selected saturation constant.
            if (signed_rounded[65:31]
                == {35{signed_rounded[31]}}) begin
                round_shift_to_int32 = signed_rounded[31:0];
            end else begin
                case (signed_rounded[65])
                    1'b1: round_shift_to_int32 = 32'sh80000000;
                    default: round_shift_to_int32 = 32'sh7fffffff;
                endcase
            end
        end
    endfunction

    localparam integer KBU_TO_Q48_SHIFT = 48-K_FRACTION_BITS;
    (* use_dsp = "yes" *) wire signed [64:0] state_accumulator [0:63];
    wire signed [64:0] kbu_q48 [0:63];
    wire signed [31:0] kb_next [0:63];
    wire signed [31:0] state_next [0:63];
    wire signed [39:0] c_product [0:63];
    (* use_dsp = "yes" *) reg signed [39:0] c_product_pipe [0:63];
    genvar result_lane;
    generate
        for (result_lane = 0; result_lane < 64;
             result_lane = result_lane + 1) begin : GEN_STATE_RESULT
            localparam integer RESULT_STATE = result_lane % 16;
            assign kbu_q48[result_lane]
                = $signed(mult_p[result_lane]) <<< KBU_TO_Q48_SHIFT;
            assign state_accumulator[result_lane]
                = $signed({ah_group_read[result_lane*64+63],
                           ah_group_read[result_lane*64 +: 64]})
                + kbu_q48[result_lane];
            // K19*B8 is guaranteed by the exporter to fit signed 27 bits.
            // Store the exact raw KB product; no intermediate rounding occurs.
            assign kb_next[result_lane] = mult_p[result_lane][31:0];
            assign state_next[result_lane]
                = round_shift_to_int32(state_accumulator[result_lane], 24);
            assign bbar_group_write[result_lane*32 +: 32]
                = kb_next[result_lane];
            assign ah_group_write[result_lane*64 +: 64]
                = mult_p[result_lane][63:0];
            assign state_group_write[result_lane*32 +: 32]
                = state_next[result_lane];

            // Keep all 64 C products in parallel, but map the registered
            // 32x8 multiply into the available DSP48 fabric instead of LUTs.
            (* use_dsp = "yes" *)
            wire signed [39:0] c_product_internal;
            assign c_product_internal
                = state_next[result_lane]
                * $signed(mult_c_record_d2[RESULT_STATE*8 +: 8]);
            assign c_product[result_lane] = c_product_internal;
        end
    endgenerate

    assign bbar_group_write_en
        = mult_valid_d2 && (mult_phase_d2 == PHASE_B);
    assign ah_group_write_en
        = mult_valid_d2 && (mult_phase_d2 == PHASE_A);
    assign state_group_write_en
        = mult_valid_d2 && (mult_phase_d2 == PHASE_X);

    // Four-stage readout tree: sixteen h*C terms -> one Q24 value per channel.
    (* use_dsp = "yes" *) reg signed [40:0] y_level1 [0:31];
    (* use_dsp = "yes" *) reg signed [41:0] y_level2 [0:15];
    (* use_dsp = "yes" *) reg signed [42:0] y_level3 [0:7];
    (* use_dsp = "yes" *) reg signed [43:0] y_level4 [0:3];
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l1;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l2;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l3;
    reg [RECORD_ADDR_WIDTH-1:0] y_record_l4;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l1;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l2;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l3;
    reg [GROUP_ADDR_WIDTH-1:0] y_group_l4;
    integer tree_index;

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            fsm_state <= ST_IDLE;
            current_record <= 10'd0;
            frame_active <= rst_n && frame_start;
            issue_group <= 6'd0;
            mult_valid_d0 <= 1'b0;
            mult_valid_d1 <= 1'b0;
            mult_valid_d2 <= 1'b0;
            mult_phase_d0 <= PHASE_B;
            mult_phase_d1 <= PHASE_B;
            mult_phase_d2 <= PHASE_B;
            mult_group_d0 <= 6'd0;
            mult_group_d1 <= 6'd0;
            mult_group_d2 <= 6'd0;
            mult_record_d0 <= {RECORD_ADDR_WIDTH{1'b0}};
            mult_record_d1 <= {RECORD_ADDR_WIDTH{1'b0}};
            mult_record_d2 <= {RECORD_ADDR_WIDTH{1'b0}};
            mult_c_record_d0 <= 128'd0;
            mult_c_record_d1 <= 128'd0;
            mult_c_record_d2 <= 128'd0;
            y_product_valid <= 1'b0;
            y_product_record <= {RECORD_ADDR_WIDTH{1'b0}};
            y_product_group <= {GROUP_ADDR_WIDTH{1'b0}};
            y_valid_l1 <= 1'b0;
            y_valid_l2 <= 1'b0;
            y_valid_l3 <= 1'b0;
            y_valid_l4 <= 1'b0;
            out_valid <= 1'b0;
            out_record_addr <= 10'd0;
            out_channel_base <= 6'd0;
            out_y_q24 <= 192'd0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            out_valid <= y_valid_l4;

            mult_valid_d0 <= issue_fire;
            mult_valid_d1 <= mult_valid_d0;
            mult_valid_d2 <= mult_valid_d1;
            if (issue_fire) begin
                mult_phase_d0 <= issue_phase;
                mult_group_d0 <= issue_group;
                mult_record_d0 <= current_record;
                if (CONTINUOUS_SPE != 0)
                    mult_c_record_d0 <= active_c_record;
                else if (CONTINUOUS_SPA != 0)
                    mult_c_record_d0 <= current_c_record;
                else
                    mult_c_record_d0 <= c_record_word;
            end
            if (mult_valid_d0) begin
                mult_phase_d1 <= mult_phase_d0;
                mult_group_d1 <= mult_group_d0;
                mult_record_d1 <= mult_record_d0;
                mult_c_record_d1 <= mult_c_record_d0;
            end
            if (mult_valid_d1) begin
                mult_phase_d2 <= mult_phase_d1;
                mult_group_d2 <= mult_group_d1;
                mult_record_d2 <= mult_record_d1;
                mult_c_record_d2 <= mult_c_record_d1;
            end

            y_product_valid
                <= mult_valid_d2 && (mult_phase_d2 == PHASE_X);
            y_valid_l1 <= y_product_valid;
            y_valid_l2 <= y_valid_l1;
            y_valid_l3 <= y_valid_l2;
            y_valid_l4 <= y_valid_l3;

            if (mult_valid_d2 && (mult_phase_d2 == PHASE_X)) begin
                y_product_record <= mult_record_d2;
                y_product_group <= mult_group_d2;
                for (tree_index = 0; tree_index < 64;
                     tree_index = tree_index + 1) begin
                    c_product_pipe[tree_index] <= c_product[tree_index];
                end
            end
            if (y_product_valid) begin
                y_record_l1 <= y_product_record;
                y_group_l1 <= y_product_group;
                for (tree_index = 0; tree_index < 32;
                     tree_index = tree_index + 1) begin
                    y_level1[tree_index]
                        <= $signed({c_product_pipe[tree_index*2][39],
                                    c_product_pipe[tree_index*2]})
                         + $signed({c_product_pipe[tree_index*2+1][39],
                                    c_product_pipe[tree_index*2+1]});
                end
            end
            if (y_valid_l1) begin
                y_record_l2 <= y_record_l1;
                y_group_l2 <= y_group_l1;
                for (tree_index = 0; tree_index < 16;
                     tree_index = tree_index + 1) begin
                    y_level2[tree_index]
                        <= $signed({y_level1[tree_index*2][40],
                                    y_level1[tree_index*2]})
                         + $signed({y_level1[tree_index*2+1][40],
                                    y_level1[tree_index*2+1]});
                end
            end
            if (y_valid_l2) begin
                y_record_l3 <= y_record_l2;
                y_group_l3 <= y_group_l2;
                for (tree_index = 0; tree_index < 8;
                     tree_index = tree_index + 1) begin
                    y_level3[tree_index]
                        <= $signed({y_level2[tree_index*2][41],
                                    y_level2[tree_index*2]})
                         + $signed({y_level2[tree_index*2+1][41],
                                    y_level2[tree_index*2+1]});
                end
            end
            if (y_valid_l3) begin
                y_record_l4 <= y_record_l3;
                y_group_l4 <= y_group_l3;
                for (tree_index = 0; tree_index < 4;
                     tree_index = tree_index + 1) begin
                    y_level4[tree_index]
                        <= $signed({y_level3[tree_index*2][42],
                                    y_level3[tree_index*2]})
                         + $signed({y_level3[tree_index*2+1][42],
                                    y_level3[tree_index*2+1]});
                end
            end
            if (y_valid_l4) begin
                out_record_addr <= y_record_l4;
                out_channel_base <= y_group_l4*4;
                out_y_q24[47:0] <= {{4{y_level4[0][43]}}, y_level4[0]};
                out_y_q24[95:48] <= {{4{y_level4[1][43]}}, y_level4[1]};
                out_y_q24[143:96] <= {{4{y_level4[2][43]}}, y_level4[2]};
                out_y_q24[191:144] <= {{4{y_level4[3][43]}}, y_level4[3]};
                if ((y_record_l4 == RECORD_COUNT-1)
                    && (y_group_l4 == GROUP_COUNT-1)) begin
                    done <= 1'b1;
                end
            end

            if (compute_start && (fsm_state == ST_IDLE)
                && !frame_active) begin
                current_record <= 10'd0;
                issue_group <= 6'd0;
                frame_active <= 1'b1;
                if (record_zero_ready)
                    fsm_state <= ST_PRE_B_INPUT;
                else
                    fsm_state <= ST_IDLE;
                mult_valid_d0 <= 1'b0;
                mult_valid_d1 <= 1'b0;
                mult_valid_d2 <= 1'b0;
                y_product_valid <= 1'b0;
                y_valid_l1 <= 1'b0;
                y_valid_l2 <= 1'b0;
                y_valid_l3 <= 1'b0;
                y_valid_l4 <= 1'b0;
                out_valid <= 1'b0;
            end else begin
                case (fsm_state)
                    ST_IDLE: begin
                        issue_group <= 6'd0;
                        if (frame_active && current_record_ready)
                            fsm_state <= ST_PRE_B_INPUT;
                    end

                    ST_PRE_B_INPUT: begin
                        issue_group <= 6'd0;
                        fsm_state <= ST_PRE_B_LUT;
                    end

                    ST_PRE_B_LUT: begin
                        issue_group <= 6'd0;
                        fsm_state <= ST_ISSUE_B;
                    end

                    ST_ISSUE_B: begin
                        if (issue_group == GROUP_COUNT-1) begin
                            issue_group <= 6'd0;
                            fsm_state <= ST_ISSUE_A;
                        end else begin
                            issue_group <= issue_group + 1'b1;
                        end
                    end

                    ST_ISSUE_A: begin
                        if (issue_group == GROUP_COUNT-1) begin
                            issue_group <= 6'd0;
                            fsm_state <= ST_ISSUE_X;
                        end else begin
                            issue_group <= issue_group + 1'b1;
                        end
                    end

                    ST_ISSUE_X: begin
                        if (issue_group == GROUP_COUNT-1) begin
                            issue_group <= 6'd0;
                            if (continuous_switch) begin
                                // LUT B0(r+1) was requested on this same edge;
                                // it is ready when the next clock issues B0.
                                current_record <= current_record + 1'b1;
                                fsm_state <= ST_ISSUE_B;
                            end else begin
                                fsm_state <= ST_WAIT_X;
                            end
                        end else begin
                            issue_group <= issue_group + 1'b1;
                        end
                    end

                    ST_WAIT_X: begin
                        if (mult_valid_d2 && (mult_phase_d2 == PHASE_X)
                            && (mult_group_d2 == GROUP_COUNT-1)) begin
                            issue_group <= 6'd0;
                            if (current_record == RECORD_COUNT-1) begin
                                fsm_state <= ST_IDLE;
                                frame_active <= 1'b0;
                            end else begin
                                current_record <= current_record + 1'b1;
                                if (next_record_ready)
                                    fsm_state <= ST_PRE_B_INPUT;
                                else
                                    fsm_state <= ST_IDLE;
                            end
                        end
                    end

                    default: begin
                        fsm_state <= ST_IDLE;
                    end
                endcase
            end

        end
    end
endmodule

`default_nettype wire
