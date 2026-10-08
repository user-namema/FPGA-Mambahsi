`timescale 1ns / 1ps
`default_nettype none

// Fixed INT8 requantization table.  The address is the raw two's-complement
// INT8 code and the 9-bit result is the exactly rounded, still-unclipped
// aligned value.  Keeping this as a 256x9 distributed ROM removes the
// dynamic multiplier and barrel-shift cone from residual fusion.
module fixed_int8_scale_lutrom #(
    parameter integer INPUT_BITS = 8,
    parameter signed [15:0] MULTIPLIER = 16'sd1,
    parameter signed [6:0]  SHIFT = 7'sd0
) (
    input  wire [INPUT_BITS-1:0] addr,
    output wire signed [8:0] data
);
    (* rom_style = "distributed" *) reg signed [8:0] rom [0:(1<<INPUT_BITS)-1];

    function signed [8:0] make_entry;
        input [INPUT_BITS-1:0] code;
        reg signed [INPUT_BITS-1:0] sample;
        reg signed [INPUT_BITS+15:0] product;
        reg signed [31:0] magnitude;
        reg signed [31:0] rounded;
        reg signed [31:0] signed_result;
        begin
            sample = $signed(code);
            product = sample * MULTIPLIER;
            magnitude = (product < 0) ? -product : product;
            if (SHIFT > 0)
                rounded = (magnitude + (32'sd1 <<< (SHIFT-1))) >>> SHIFT;
            else if (SHIFT < 0)
                rounded = magnitude <<< (-SHIFT);
            else
                rounded = magnitude;
            signed_result = (product < 0) ? -rounded : rounded;
            make_entry = signed_result[8:0];
        end
    endfunction

    integer lut_index;
    initial begin
        for (lut_index = 0; lut_index < (1<<INPUT_BITS); lut_index = lut_index + 1)
            rom[lut_index] = make_entry(lut_index[INPUT_BITS-1:0]);
    end
    assign data = rom[addr];
endmodule


// Four physical lanes service one 8-lane slot in two clocks.  Each lane owns
// one A and one B 256x9 LUTROM.  The auxiliary word travels with the slot so
// the second residual stage cannot be misaligned when the II changes to two.
module align_add_int8_lanes #(
    parameter integer LANES = 8,
    parameter integer A_BITS = 8,
    parameter integer AUX_WIDTH = 64,
    parameter integer META_WIDTH = 16,
    parameter signed [15:0] FIXED_A_MULTIPLIER = 16'sd1,
    parameter signed [6:0]  FIXED_A_SHIFT = 7'sd0,
    parameter signed [15:0] FIXED_B_MULTIPLIER = 16'sd1,
    parameter signed [6:0]  FIXED_B_SHIFT = 7'sd0
) (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         frame_start,
    input  wire                         in_valid,
    input  wire [LANES*A_BITS-1:0]      in_a,
    input  wire [LANES*8-1:0]           in_b,
    input  wire signed [15:0]           cfg_a_multiplier,
    input  wire signed [6:0]            cfg_a_shift,
    input  wire signed [15:0]           cfg_b_multiplier,
    input  wire signed [6:0]            cfg_b_shift,
    input  wire [META_WIDTH-1:0]        in_metadata,
    input  wire                         in_last,
    input  wire [AUX_WIDTH-1:0]         in_aux,
    output wire                         in_ready,
    output reg                          out_valid,
    output reg [LANES*8-1:0]            out_data,
    output reg [META_WIDTH-1:0]         out_metadata,
    output reg                          out_last,
    output reg [AUX_WIDTH-1:0]          out_aux
);
    localparam integer PHYSICAL_LANES = 4;
    reg high_half_pending;
    reg [LANES*A_BITS-1:0] captured_a;
    reg [63:0] captured_b;
    reg [AUX_WIDTH-1:0] captured_aux;
    reg [31:0] low_result;
    reg [META_WIDTH-1:0] captured_metadata;
    reg captured_last;
    wire [A_BITS-1:0] lut_addr_a [0:PHYSICAL_LANES-1];
    wire [7:0] lut_addr_b [0:PHYSICAL_LANES-1];
    wire signed [8:0] aligned_a [0:PHYSICAL_LANES-1];
    wire signed [8:0] aligned_b [0:PHYSICAL_LANES-1];
    genvar lane;

    assign in_ready = !high_half_pending;
    generate
        for (lane = 0; lane < PHYSICAL_LANES; lane = lane + 1) begin : GEN_LUT
            assign lut_addr_a[lane]
                = high_half_pending ? captured_a[(lane+4)*A_BITS +: A_BITS]
                                    : in_a[lane*A_BITS +: A_BITS];
            assign lut_addr_b[lane]
                = high_half_pending ? captured_b[(lane+4)*8 +: 8]
                                    : in_b[lane*8 +: 8];
            fixed_int8_scale_lutrom #(
                .INPUT_BITS(A_BITS),
                .MULTIPLIER(FIXED_A_MULTIPLIER), .SHIFT(FIXED_A_SHIFT)
            ) u_a_lut (
                .addr(lut_addr_a[lane]), .data(aligned_a[lane])
            );
            fixed_int8_scale_lutrom #(
                .MULTIPLIER(FIXED_B_MULTIPLIER), .SHIFT(FIXED_B_SHIFT)
            ) u_b_lut (
                .addr(lut_addr_b[lane]), .data(aligned_b[lane])
            );
        end
    endgenerate

    function [7:0] clip_int8;
        input signed [32:0] value;
        begin
            if (value > 33'sd127)
                clip_int8 = 8'h7f;
            else if (value < -33'sd128)
                clip_int8 = 8'h80;
            else
                clip_int8 = value[7:0];
        end
    endfunction

    integer pipeline_lane;
    reg signed [9:0] aligned_sum;

    always @(posedge clk) begin
        if (!rst_n) begin
            high_half_pending <= 1'b0;
            out_valid <= 1'b0;
            out_last <= 1'b0;
        end else begin
            if (frame_start) begin
                high_half_pending <= 1'b0;
                out_valid <= 1'b0;
                out_last <= 1'b0;
            end else begin
                out_valid <= high_half_pending;
                out_last <= high_half_pending && captured_last;
                if (high_half_pending)
                    high_half_pending <= 1'b0;
                else if (in_valid && in_ready)
                    high_half_pending <= 1'b1;
            end

            // Payload capture is valid-qualified, not frame-start-qualified.
            if (in_valid && in_ready) begin
                captured_a <= in_a;
                captured_b <= in_b;
                captured_aux <= in_aux;
                captured_metadata <= in_metadata;
                captured_last <= in_last;
                for (pipeline_lane = 0; pipeline_lane < PHYSICAL_LANES;
                     pipeline_lane = pipeline_lane + 1) begin
                    aligned_sum
                        = $signed(aligned_a[pipeline_lane])
                        + $signed(aligned_b[pipeline_lane]);
                    low_result[pipeline_lane*8 +: 8]
                        <= clip_int8(aligned_sum);
                end
            end

            if (high_half_pending) begin
                out_metadata <= captured_metadata;
                out_aux <= captured_aux;
                out_data[31:0] <= low_result;
                for (pipeline_lane = 0; pipeline_lane < PHYSICAL_LANES;
                     pipeline_lane = pipeline_lane + 1) begin
                    aligned_sum
                        = $signed(aligned_a[pipeline_lane])
                        + $signed(aligned_b[pipeline_lane]);
                    out_data[(pipeline_lane+4)*8 +: 8]
                        <= clip_int8(aligned_sum);
                end
            end
        end
    end

    wire unused_cfg = cfg_a_multiplier[0] ^ cfg_a_shift[0]
                    ^ cfg_b_multiplier[0] ^ cfg_b_shift[0];
endmodule


// One ordered 8-lane slot per address.  Reuse the existing 1024x128
// simple-dual-port Block Memory Generator and leave the upper half unused.
// Port A accepts future branch results while port B reads the oldest slot that
// is complete in all three streams.
module block0_stream_slot_bram #(
    parameter integer DATA_WIDTH = 64
) (
    input  wire         clk,
    input  wire         wr_en,
    input  wire [9:0]   wr_addr,
    input  wire [DATA_WIDTH-1:0] wr_data,
    input  wire         rd_en,
    input  wire [9:0]   rd_addr,
    output wire [DATA_WIDTH-1:0] rd_data
);
    wire [127:0] rd_word;
    ssm_bc_bram_1024x128 u_bram (
        .clka(clk),
        .ena(wr_en),
        .wea({wr_en}),
        .addra(wr_addr),
        .dina({{(128-DATA_WIDTH){1'b0}}, wr_data}),
        .clkb(clk),
        .enb(rd_en),
        .addrb(rd_addr),
        .doutb(rd_word)
    );
    assign rd_data = rd_word[DATA_WIDTH-1:0];
endmodule


module block0_residual_fusion_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer INPUT_BITS=(BLOCK_ID==0)?8:10,
    parameter integer PIXEL_COUNT=256,
    parameter integer TOTAL_SLOTS=PIXEL_COUNT*4,
    parameter signed [15:0] FUSION_A_MULT=(BLOCK_ID==0)?16'sd25292:(BLOCK_ID==1)?16'sd25951:16'sd18646,
    parameter signed [6:0]  FUSION_A_SHIFT=7'sd15,
    parameter signed [15:0] FUSION_B_MULT=(BLOCK_ID==0)?16'sd30815:(BLOCK_ID==1)?16'sd28388:16'sd28409,
    parameter signed [6:0]  FUSION_B_SHIFT=(BLOCK_ID==2)?7'sd16:7'sd17,
    parameter signed [15:0] BLOCK_A_MULT=(BLOCK_ID==0)?16'sd21625:(BLOCK_ID==1)?16'sd27952:16'sd29302,
    parameter signed [6:0]  BLOCK_A_SHIFT=(BLOCK_ID==0)?7'sd15:7'sd17,
    parameter signed [15:0] BLOCK_B_MULT=(BLOCK_ID==0)?16'sd22344:(BLOCK_ID==1)?16'sd19179:16'sd28059,
    parameter signed [6:0]  BLOCK_B_SHIFT=(BLOCK_ID==0)?7'sd16:(BLOCK_ID==1)?7'sd17:7'sd18
) (
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,

    input  wire                 input_valid,
    input  wire [7:0]           input_pixel_addr,
    input  wire [4:0]           input_channel_base,
    input  wire [4*INPUT_BITS-1:0] input_data,

    input  wire                 spa_valid,
    input  wire [7:0]           spa_pixel_addr,
    input  wire [4:0]           spa_channel_base,
    input  wire [31:0]          spa_data,
    input  wire                 spa_done,
    input  wire                 spe_valid,
    input  wire [7:0]           spe_pixel_addr,
    input  wire [1:0]           spe_token,
    input  wire [63:0]          spe_data,
    input  wire                 spe_done,

    input  wire signed [15:0]   cfg_fusion_a_multiplier,
    input  wire signed [6:0]    cfg_fusion_a_shift,
    input  wire signed [15:0]   cfg_fusion_b_multiplier,
    input  wire signed [6:0]    cfg_fusion_b_shift,
    input  wire signed [15:0]   cfg_block_a_multiplier,
    input  wire signed [6:0]    cfg_block_a_shift,
    input  wire signed [15:0]   cfg_block_b_multiplier,
    input  wire signed [6:0]    cfg_block_b_shift,

    output wire                 spa_result_valid,
    output wire [15:0]          spa_result_metadata,
    output wire [31:0]          spa_result_data,
    output wire                 spe_result_valid,
    output wire [15:0]          spe_result_metadata,
    output wire [63:0]          spe_result_data,
    output wire                 fusion_valid,
    output wire [15:0]          fusion_metadata,
    output wire [63:0]          fusion_data,
    output wire                 out_valid,
    output wire [7:0]           out_pixel_addr,
    output wire [4:0]           out_channel_base,
    output wire [63:0]          out_data,
    output wire                 done
);
    // The current model has no residual inside SpaMamba or SpeMamba.
    // Branch results are exactly ReLU(out_proj), already quantized to the
    // branch fake-quant boundary scale.
    assign spa_result_valid = spa_valid;
    assign spa_result_metadata = {3'd0, spa_pixel_addr, spa_channel_base};
    assign spa_result_data = spa_data;
    assign spe_result_valid = spe_valid;
    assign spe_result_metadata = {6'd0, spe_pixel_addr, spe_token};
    assign spe_result_data = spe_data;

    // Input and Spa arrive as 4-lane beats.  Pair low/high halves into one
    // 8-lane slot; Spe already produces one complete slot per valid cycle.
    // slot={pixel,group/token}=pixel*4+group, in the exact final output order.
    reg [4*INPUT_BITS-1:0] input_low_half;
    reg [9:0] input_low_slot;
    reg input_low_valid;
    reg [31:0] spa_low_half;
    reg [9:0] spa_low_slot;
    reg spa_low_valid;

    wire [9:0] input_arrival_slot
        = {input_pixel_addr, input_channel_base[4:3]};
    wire [9:0] spa_arrival_slot
        = {spa_pixel_addr, spa_channel_base[4:3]};
    wire [9:0] spe_arrival_slot = {spe_pixel_addr, spe_token};
    wire input_slot_write = input_valid
        && (input_channel_base[2:0] == 3'b100);
    wire spa_slot_write = spa_valid
        && (spa_channel_base[2:0] == 3'b100);
    wire [8*INPUT_BITS-1:0] input_slot_write_data = {input_data, input_low_half};
    wire [63:0] spa_slot_write_data = {spa_data, spa_low_half};

    // Ordered prefix watermarks replace 3x1024 resettable ready bitmaps.  They
    // are safe because all three producers are record/group monotonic; the
    // simulation assertions below make that contract explicit.
    // Absolute scene watermarks: Block0 contributes 1024 slots per tile.
    // A 16-bit producer wraps before tile 64 has drained. RAM addresses
    // below intentionally remain tile-local; only the watermarks are wide.
    reg [31:0] input_ready_slots;
    reg [31:0] spa_ready_slots;
    reg [31:0] spe_ready_slots;
    reg [31:0] fusion_issue_count;
    localparam integer SLOT_INDEX_BITS=$clog2(TOTAL_SLOTS);
    wire [9:0] fusion_issue_slot=
        {{(10-SLOT_INDEX_BITS){1'b0}},
          fusion_issue_count[SLOT_INDEX_BITS-1:0]};
    wire [9:0] input_expected_slot=
        {{(10-SLOT_INDEX_BITS){1'b0}},
          input_ready_slots[SLOT_INDEX_BITS-1:0]};
    wire [9:0] spa_expected_slot=
        {{(10-SLOT_INDEX_BITS){1'b0}},
          spa_ready_slots[SLOT_INDEX_BITS-1:0]};
    wire [9:0] spe_expected_slot=
        {{(10-SLOT_INDEX_BITS){1'b0}},
          spe_ready_slots[SLOT_INDEX_BITS-1:0]};
    reg fusion_read_valid;
    reg [9:0] fusion_read_slot;
    reg fusion_read_last;
    wire fusion_engine_ready;
    wire block_engine_ready;
    wire fusion_read_issue
        = (input_ready_slots > fusion_issue_count)
       && (spa_ready_slots > fusion_issue_count)
       && (spe_ready_slots > fusion_issue_count)
       // A request is launched every other cycle.  Its one-cycle BRAM
       // latency makes the returned word coincide with the 4-lane engine's
       // next accept edge, even though in_ready is low on this issue edge.
       && !fusion_read_valid;

    wire [8*INPUT_BITS-1:0] input_slot_read_data;
    wire [63:0] spa_slot_read_data;
    wire [63:0] spe_slot_read_data;

    // Reuse the existing 128-bit physical RAM: 64 payload bits for Block0,
    // 80 payload bits for Block1/2. No IP change and no read-latency change.
    block0_stream_slot_bram #(.DATA_WIDTH(8*INPUT_BITS)) u_input_slot_bram (
        .clk(clk),
        .wr_en(input_slot_write),
        .wr_addr(input_arrival_slot),
        .wr_data(input_slot_write_data),
        .rd_en(fusion_read_issue),
        .rd_addr(fusion_issue_slot),
        .rd_data(input_slot_read_data)
    );

    block0_stream_slot_bram u_spa_slot_bram (
        .clk(clk),
        .wr_en(spa_slot_write),
        .wr_addr(spa_arrival_slot),
        .wr_data(spa_slot_write_data),
        .rd_en(fusion_read_issue),
        .rd_addr(fusion_issue_slot),
        .rd_data(spa_slot_read_data)
    );

    block0_stream_slot_bram u_spe_slot_bram (
        .clk(clk),
        .wr_en(spe_valid),
        .wr_addr(spe_arrival_slot),
        .wr_data(spe_data),
        .rd_en(fusion_read_issue),
        .rd_addr(fusion_issue_slot),
        .rd_data(spe_slot_read_data)
    );

    // Low halves are payload; frame_start resets only the slot-valid/control
    // state below.  This avoids a 64-bit frame CE/reset branch.
    always @(posedge clk) begin
        if (rst_n) begin
            if (input_valid && (input_channel_base[2:0] == 3'b000))
                input_low_half <= input_data;
            if (spa_valid && (spa_channel_base[2:0] == 3'b000))
                spa_low_half <= spa_data;
        end
    end

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            input_low_valid <= 1'b0;
            spa_low_valid <= 1'b0;
            input_ready_slots <= 11'd0;
            spa_ready_slots <= 11'd0;
            spe_ready_slots <= 11'd0;
            fusion_issue_count <= 11'd0;
            fusion_read_valid <= 1'b0;
        end else begin
            fusion_read_valid <= fusion_read_issue;
            if (fusion_read_issue) begin
                fusion_read_slot <= fusion_issue_slot;
                fusion_read_last <= (fusion_issue_slot == TOTAL_SLOTS-1);
                fusion_issue_count <= fusion_issue_count + 1'b1;
            end

            if (input_valid) begin
                if (input_channel_base[2:0] == 3'b000) begin
                    input_low_slot <= input_arrival_slot;
                    input_low_valid <= 1'b1;
                end else if (input_channel_base[2:0] == 3'b100) begin
                    input_low_valid <= 1'b0;
                    input_ready_slots <= input_ready_slots + 1'b1;
                end
            end

            if (spa_valid) begin
                if (spa_channel_base[2:0] == 3'b000) begin
                    spa_low_slot <= spa_arrival_slot;
                    spa_low_valid <= 1'b1;
                end else if (spa_channel_base[2:0] == 3'b100) begin
                    spa_low_valid <= 1'b0;
                    spa_ready_slots <= spa_ready_slots + 1'b1;
                end
            end

            if (spe_valid)
                spe_ready_slots <= spe_ready_slots + 1'b1;
        end
    end

`ifndef SYNTHESIS
    always @(posedge clk) begin
        if (rst_n && !frame_start) begin
            if (input_slot_write
                && (!input_low_valid
                    || (input_low_slot != input_arrival_slot)))
                $error("Block input 4-lane halves are not a matched slot");
            if (spa_slot_write
                && (!spa_low_valid || (spa_low_slot != spa_arrival_slot)))
                $error("Spa 4-lane halves are not a matched slot");
            if (input_slot_write
                && (input_arrival_slot != input_expected_slot))
                $error("Block input slots are not monotonic/gap-free");
            if (spa_slot_write
                && (spa_arrival_slot != spa_expected_slot))
                $error("Spa slots are not monotonic/gap-free");
            if (spe_valid
                && (spe_arrival_slot != spe_expected_slot))
                $error("Spe slots are not monotonic/gap-free");
            if (fusion_read_issue
                && ((fusion_issue_count >= input_ready_slots)
                    || (fusion_issue_count >= spa_ready_slots)
                    || (fusion_issue_count >= spe_ready_slots)))
                $error("Fusion read crossed a producer readiness watermark");
            if (fusion_read_valid && !fusion_engine_ready)
                $error("Fusion 4-lane engine lost a returned BRAM word");
            if (fusion_valid && !block_engine_ready)
                $error("Block-residual 4-lane engine lost a fusion word");
        end
    end
`endif

    wire fusion_last;
    wire [8*INPUT_BITS-1:0] fusion_input_aux;
    align_add_int8_lanes #(
        .LANES(8),
        .AUX_WIDTH(8*INPUT_BITS),
        .META_WIDTH(16),
        .FIXED_A_MULTIPLIER(FUSION_A_MULT),
        .FIXED_A_SHIFT(FUSION_A_SHIFT),
        .FIXED_B_MULTIPLIER(FUSION_B_MULT),
        .FIXED_B_SHIFT(FUSION_B_SHIFT)
    ) u_fusion (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .in_valid(fusion_read_valid),
        .in_a(spa_slot_read_data),
        .in_b(spe_slot_read_data),
        .cfg_a_multiplier(cfg_fusion_a_multiplier),
        .cfg_a_shift(cfg_fusion_a_shift),
        .cfg_b_multiplier(cfg_fusion_b_multiplier),
        .cfg_b_shift(cfg_fusion_b_shift),
        .in_metadata({6'd0, fusion_read_slot}),
        .in_last(fusion_read_valid && fusion_read_last),
        .in_aux(input_slot_read_data),
        .in_ready(fusion_engine_ready),
        .out_valid(fusion_valid),
        .out_data(fusion_data),
        .out_metadata(fusion_metadata),
        .out_last(fusion_last),
        .out_aux(fusion_input_aux)
    );

    wire [15:0] block_metadata;
    wire [63:0] unused_block_aux;
    align_add_int8_lanes #(
        .LANES(8),
        .A_BITS(INPUT_BITS),
        .META_WIDTH(16),
        .FIXED_A_MULTIPLIER(BLOCK_A_MULT),
        .FIXED_A_SHIFT(BLOCK_A_SHIFT),
        .FIXED_B_MULTIPLIER(BLOCK_B_MULT),
        .FIXED_B_SHIFT(BLOCK_B_SHIFT)
    ) u_block_residual (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .in_valid(fusion_valid),
        .in_a(fusion_input_aux),
        .in_b(fusion_data),
        .cfg_a_multiplier(cfg_block_a_multiplier),
        .cfg_a_shift(cfg_block_a_shift),
        .cfg_b_multiplier(cfg_block_b_multiplier),
        .cfg_b_shift(cfg_block_b_shift),
        .in_metadata(fusion_metadata),
        .in_last(fusion_last),
        .in_aux(64'd0),
        .in_ready(block_engine_ready),
        .out_valid(out_valid),
        .out_data(out_data),
        .out_metadata(block_metadata),
        .out_last(done),
        .out_aux(unused_block_aux)
    );

    assign out_pixel_addr = block_metadata[9:2];
    assign out_channel_base = {block_metadata[1:0], 3'b000};
    wire unused_cfg = cfg_fusion_a_multiplier[0] ^ cfg_fusion_a_shift[0]
                    ^ cfg_fusion_b_multiplier[0] ^ cfg_fusion_b_shift[0]
                    ^ cfg_block_a_multiplier[0] ^ cfg_block_a_shift[0]
                    ^ cfg_block_b_multiplier[0] ^ cfg_block_b_shift[0]
                    ^ block_engine_ready;
endmodule

`default_nettype wire
