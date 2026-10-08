`timescale 1ns/1ps
`default_nettype none

// One local modulo-four row bank for the Block0 Spa x_proj completed sums.
// All nine rows are written together when a record finishes, so this is
// intentionally a small register file rather than a multi-write inferred RAM.
// Requant lane N reads only bank N, limiting the row mux and its routes to one
// physical hierarchy instead of selecting from a monolithic 4x34 array.
module block0_spa_completed_sum_reg_bank (
    input  wire                 clk,
    input  wire                 commit_valid,
    input  wire [1:0]           commit_record,
    input  wire [188:0]         commit_data,
    input  wire                 read_record_load,
    input  wire [1:0]           read_record,
    input  wire [3:0]           read_addr,
    output wire signed [20:0]   read_data
);
    (* ram_style = "registers" *) reg signed [20:0] rows [0:3][0:8];
    // Capture the modulo-record selector once when this bank's requant walk
    // starts.  The selector then stays physically local for all nine reads;
    // rq_record no longer drives the muxes of all four 189-bit banks.
    (* keep = "true", max_fanout = 16 *) reg [1:0] read_record_local;
    integer row_index;
    always @(posedge clk) begin
        if (read_record_load)
            read_record_local <= read_record;
        if (commit_valid)
            for (row_index=0; row_index<9; row_index=row_index+1)
                rows[commit_record][row_index]
                    <= commit_data[row_index*21 +: 21];
    end
    assign read_data = (read_addr < 9)
        ? rows[read_record_local][read_addr] : 21'sd0;
endmodule

// Block0 Spa x_proj in STEAM-style input-channel-major order.
//
// Four activation channels update all 34 output partial sums each cycle.  A
// four-record activation queue absorbs the bursty eight-channel Conv output;
// record_credit_ready is returned to the Block0 in_proj scheduler before it
// launches another pixel.  The 34 completed sums are requantized four rows per
// cycle, so this stage remains balanced to the 16-cycle Spa SSM record rate.
module block0_spa_x_proj_steam #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] DT_MULTIPLIER=16'sd31024,
    parameter signed [6:0]  DT_SHIFT=7'sd19,
    parameter signed [15:0] B_MULTIPLIER=16'sd17810,
    parameter signed [6:0]  B_SHIFT=7'sd21,
    parameter signed [15:0] C_MULTIPLIER=16'sd16685,
    parameter signed [6:0]  C_SHIFT=7'sd21
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 frame_start,
    input  wire                 downstream_record_ready,
    input  wire                 in_valid,
    input  wire [7:0]           in_pixel_addr,
    input  wire [5:0]           in_channel_base,
    input  wire [63:0]          in_data,
    input  wire                 cfg_weight_we,
    input  wire [11:0]          cfg_weight_addr,
    input  wire signed [7:0]    cfg_weight_data,
    input  wire signed [31:0]   cfg_dt_multiplier,
    input  wire signed [6:0]    cfg_dt_shift,
    input  wire signed [31:0]   cfg_b_multiplier,
    input  wire signed [6:0]    cfg_b_shift,
    input  wire signed [31:0]   cfg_c_multiplier,
    input  wire signed [6:0]    cfg_c_shift,
    output wire                 record_credit_ready,
    output reg                  dt_valid,
    output reg [7:0]            dt_pixel_addr,
    output reg [17:0]           dt_data,
    output reg                  b_valid,
    output reg [7:0]            b_pixel_addr,
    output reg [127:0]          b_data,
    output reg                  c_valid,
    output reg [7:0]            c_pixel_addr,
    output reg [127:0]          c_data,
    output reg                  u_valid,
    output reg [7:0]            u_pixel_addr,
    output reg [5:0]            u_channel_base,
    output reg [63:0]           u_data,
    output wire                 input_record_consumed,
    output wire                 output_record_launch,
    output reg                  done
);
    wire unused_cfg = cfg_weight_we ^ cfg_weight_addr[0]
                    ^ cfg_weight_data[0] ^ cfg_dt_multiplier[0]
                    ^ cfg_dt_shift[0] ^ cfg_b_multiplier[0]
                    ^ cfg_b_shift[0] ^ cfg_c_multiplier[0]
                    ^ cfg_c_shift[0];

    reg [511:0] activation_slot [0:3];
    reg [7:0] slot_record [0:3];
    reg [3:0] slot_active;
    reg [3:0] slot_full;
    reg [7:0] capture_next_record;
    reg worker_active;
    reg [7:0] worker_record;
    reg [3:0] worker_group;
    // U is independent of the 34-row projection result.  Launch it into the
    // credit-protected SSM record slot as soon as the activation record is
    // complete instead of holding the activation until requant group 8.
    reg u_stream_active;
    reg [7:0] u_stream_record;
    reg [2:0] u_stream_group;
    // Only one 64-bit activation group is selected from the four 512-bit
    // slots at a time.  This replaces the former 512-bit slot mux controlled
    // by u_stream_record with a local 64-bit look-ahead register.
    reg [63:0] u_stream_data_q;
    // Do not use a finite-width cumulative count to decide whether U for a
    // record has launched.  In a continuous multi-tile stream the 9-bit
    // counter wrapped exactly after two 256-record tiles and permanently
    // blocked requant at the boundary.  Four tagged launch tokens match the
    // four downstream credits and remain unambiguous across record-address
    // wrap.  A token is retired when that record starts requantization.
    reg [3:0] u_launch_valid;
    reg [7:0] u_launch_record [0:3];
    reg [3:0] slot_worker_done;
    reg [3:0] slot_u_done;
    reg rq_active;
    reg [7:0] rq_record;
    reg [3:0] rq_group;
    // Match the completed-result storage depth to the four activation slots.
    // The previous two-entry parity store was arithmetically correct, but its
    // five-cycle write/visibility latency stalled the 16-cycle worker once
    // every four records (64 * 5 = 320 cycles per tile).
    reg [3:0] completed_valid;
    reg [7:0] completed_record [0:3];
    wire completed_pop = rq_active && (rq_group == 4'd8);
    wire [1:0] worker_slot = worker_record[1:0];
    wire [1:0] capture_slot = capture_next_record[1:0];
    wire [1:0] u_stream_slot = u_stream_record[1:0];
    wire worker_last_issue = worker_active && (worker_group == 4'd15);
    wire u_stream_issue = u_stream_active;
    wire u_stream_last_issue = u_stream_active
        && (u_stream_group == 3'd7);
    wire worker_releases_slot = worker_last_issue
        && (slot_u_done[worker_slot]
            || (u_stream_last_issue
                && (u_stream_record == worker_record)));
    wire u_releases_slot = u_stream_last_issue
        && (slot_worker_done[u_stream_slot]
            || (worker_last_issue
                && (worker_record == u_stream_record)));
    wire slot_release = worker_releases_slot || u_releases_slot;
    wire [7:0] slot_release_record = worker_releases_slot
        ? worker_record : u_stream_record;
    wire [1:0] slot_release_index = slot_release_record[1:0];
    // Result storage is selected by record parity, independently of the four
    // activation slots.  Do not use slot_active here: after record 0 is
    // consumed, slot 0 may already hold record 4 while worker record 2 is
    // waiting for result bank 0.  Treating that new activation as an occupied
    // result bank deadlocks the worker at record 2.
    wire worker_result_bank_ready=!completed_valid[worker_record[1:0]]
        ||(completed_pop&&(rq_record[1:0]==worker_record[1:0]));
    wire [7:0] worker_next_record = worker_record + 1'b1;
    wire [1:0] worker_next_slot = worker_next_record[1:0];
    wire worker_next_result_bank_ready = !completed_valid[worker_next_record[1:0]]
        || (completed_pop
            && (rq_record[1:0] == worker_next_record[1:0]));
    wire worker_next_ready = slot_full[worker_next_slot]
        && (slot_record[worker_next_slot] == worker_next_record)
        && worker_next_result_bank_ready;
    wire [2:0] active_slot_count = slot_active[0] + slot_active[1]
        + slot_active[2] + slot_active[3];
    // Keep one physical slot in reserve for the record already launched by
    // in_proj but not yet visible on this interface.
    // The block-level credit counter reserves a slot when in_proj launches a
    // record, so no extra headroom is needed here.  This endpoint only checks
    // that the modulo-addressed capture slot itself is free.
    // A slot is reusable as soon as both independent readers have consumed
    // it: worker group 15 and U group 7.  DT/B/C requant reads the completed
    // sum banks and therefore must not retain this activation credit.
    assign record_credit_ready = !slot_active[capture_slot]
        || (slot_release && (capture_slot == slot_release_index));

    integer capture_lane;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            slot_active <= 4'b0000;
            slot_full <= 4'b0000;
            slot_worker_done <= 4'b0000;
            slot_u_done <= 4'b0000;
            capture_next_record <= 8'd0;
        end else begin
            if (worker_last_issue)
                slot_worker_done[worker_slot] <= 1'b1;
            if (u_stream_last_issue)
                slot_u_done[u_stream_slot] <= 1'b1;
            if (slot_release) begin
                slot_active[slot_release_index] <= 1'b0;
                slot_full[slot_release_index] <= 1'b0;
                slot_worker_done[slot_release_index] <= 1'b0;
                slot_u_done[slot_release_index] <= 1'b0;
            end
            if (in_valid) begin
                for (capture_lane=0; capture_lane<8;
                     capture_lane=capture_lane+1)
                    activation_slot[in_pixel_addr[1:0]][
                        (in_channel_base+capture_lane)*8 +: 8
                    ] <= in_data[capture_lane*8 +: 8];
                if (in_channel_base == 0) begin
                    slot_active[in_pixel_addr[1:0]] <= 1'b1;
                    slot_record[in_pixel_addr[1:0]] <= in_pixel_addr;
                    slot_worker_done[in_pixel_addr[1:0]] <= 1'b0;
                    slot_u_done[in_pixel_addr[1:0]] <= 1'b0;
                    capture_next_record <= in_pixel_addr + 1'b1;
                end
                if (in_channel_base == 6'd56)
                    slot_full[in_pixel_addr[1:0]] <= 1'b1;
            end
        end
    end

    // Eight U beats are sent independently of requant.  output_record_launch
    // reserves the matching four-slot SSM context before the first beat.  The
    // one-cycle start state is below the 16-cycle worker service rate and does
    // not affect the target record II.
    wire u_stream_start = !u_stream_active
        && slot_full[u_stream_slot]
        && (slot_record[u_stream_slot] == u_stream_record)
        && downstream_record_ready;
    assign output_record_launch = u_stream_start;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            u_stream_active <= 1'b0;
            u_stream_record <= 8'd0;
            u_stream_group <= 3'd0;
        end else begin
            if (u_stream_start) begin
                u_stream_active <= 1'b1;
                u_stream_group <= 3'd0;
                u_stream_data_q <= activation_slot[u_stream_slot][0 +: 64];
            end else if (u_stream_active) begin
                if (u_stream_group == 3'd7) begin
                    u_stream_active <= 1'b0;
                    u_stream_group <= 3'd0;
                    u_stream_record <= u_stream_record + 1'b1;
                end else begin
                    u_stream_group <= u_stream_group + 1'b1;
                    u_stream_data_q <= activation_slot[u_stream_slot]
                        [(u_stream_group+1'b1)*64 +: 64];
                end
            end
        end
    end

    wire worker_issue = worker_active;
    wire [31:0] worker_activation
        = activation_slot[worker_slot][worker_group*32 +: 32];

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            worker_active <= 1'b0;
            worker_record <= 8'd0;
            worker_group <= 4'd0;
        end else begin
            if (!worker_active
                && slot_full[worker_record[1:0]]
                && (slot_record[worker_record[1:0]] == worker_record)
                && worker_result_bank_ready) begin
                worker_active <= 1'b1;
                worker_group <= 4'd0;
            end else if (worker_active) begin
                if (worker_group == 4'd15) begin
                    // The dot-product pipeline accepts one group every clock.
                    // Move directly from group 15 of record N to group 0 of
                    // record N+1 whenever its four-slot credit and result bank
                    // are already available.  The former unconditional idle
                    // clock cost one cycle per record (255 cycles/tile).
                    worker_active <= worker_next_ready;
                    worker_group <= 4'd0;
                    worker_record <= worker_next_record;
                end else begin
                    worker_group <= worker_group + 1'b1;
                end
            end
        end
    end

    // Address=input channel group; each word holds 34 rows x 4 INT8 weights.
    // All three blocks share this datapath.  Only the selected BLOCK_ID ROM
    // exists after generate elaboration, so scaling the stream architecture
    // does not replicate a three-way parameter mux in LUTs.
    wire [1087:0] stream_weight_word;
    mamba_spa_x_stream_weight_rom #(.BLOCK_ID(BLOCK_ID))
    u_stream_weight_rom(.clk(clk), .en(worker_issue),
        .addr(worker_group), .data(stream_weight_word));

    reg rom_valid;
    reg [31:0] rom_activation;
    reg [11:0] rom_metadata;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rom_valid <= 1'b0;
            rom_metadata <= 12'd0;
        end else begin
            rom_valid <= worker_issue;
            if (worker_issue) begin
                rom_activation <= worker_activation;
                rom_metadata <= {worker_record,worker_group};
            end
        end
    end

    wire pair_valid [0:16];
    wire signed [17:0] pair_low [0:16];
    wire signed [17:0] pair_high [0:16];
    wire [11:0] pair_meta [0:16];
    genvar pair_index;
    generate
        for (pair_index=0; pair_index<17; pair_index=pair_index+1)
        begin : G_X_PAIR
            mamba_dot4_pair_pipeline #(.META_WIDTH(12)) u_dot4_pair(
                .clk(clk), .rst_n(rst_n), .frame_start(frame_start),
                .in_valid(rom_valid), .activation(rom_activation),
                .weight_low(stream_weight_word[(pair_index*8)*8 +: 32]),
                .weight_high(stream_weight_word[(pair_index*8+4)*8 +: 32]),
                .in_meta(rom_metadata), .out_valid(pair_valid[pair_index]),
                .out_sum_low(pair_low[pair_index]),
                .out_sum_high(pair_high[pair_index]),
                .out_meta(pair_meta[pair_index]));
        end
    endgenerate

    wire partial_valid = pair_valid[0];
    wire partial_valid_bank [0:3];
    genvar valid_bank_index;
    generate
        for (valid_bank_index=0; valid_bank_index<4;
             valid_bank_index=valid_bank_index+1) begin : G_X_VALID_BANK
            ssm_local_control_buffer u_partial_valid_buffer (
                .control_in(partial_valid),
                .control_out(partial_valid_bank[valid_bank_index]));
        end
    endgenerate
    wire [7:0] partial_record = pair_meta[0][11:4];
    wire [3:0] partial_group = pair_meta[0][3:0];
    reg signed [20:0] partial_accumulator [0:33];
    wire signed [20:0] partial_value [0:33];
    genvar partial_row;
    generate
        for (partial_row=0; partial_row<34; partial_row=partial_row+1)
            begin : G_X_PARTIAL_VALUE
            localparam integer PAIR_NUMBER = partial_row / 2;
            if ((partial_row % 2) != 0) begin : G_HIGH
                assign partial_value[partial_row]
                    = {{3{pair_high[PAIR_NUMBER][17]}},
                       pair_high[PAIR_NUMBER]};
            end else begin : G_LOW
                assign partial_value[partial_row]
                    = {{3{pair_low[PAIR_NUMBER][17]}},
                       pair_low[PAIR_NUMBER]};
            end
        end
    endgenerate

    // Two modulo-record copies remain, but rows are physically divided by
    // row_index[1:0].  Four independent nine-row banks feed the four requant
    // lanes directly and remove the former global 34-row selection network.
    wire [188:0] completed_commit_data [0:3];
    wire signed [20:0] completed_bank_read [0:3];
    wire rq_start = !rq_active
        && completed_valid[rq_record[1:0]]
        && (completed_record[rq_record[1:0]] == rq_record)
        // DT/B/C may enter the SSM only after the early-U scheduler has
        // reserved that record's downstream context.
        && u_launch_valid[rq_record[1:0]]
        && (u_launch_record[rq_record[1:0]] == rq_record);
    genvar completed_bank;
    genvar completed_row;
    generate
        for (completed_bank=0; completed_bank<4;
             completed_bank=completed_bank+1) begin : G_COMPLETED_BANK
            for (completed_row=0; completed_row<9;
                 completed_row=completed_row+1) begin : G_COMMIT_ROW
                localparam integer GLOBAL_ROW
                    = completed_bank + completed_row*4;
                if (GLOBAL_ROW < 34) begin : G_VALID_ROW
                    assign completed_commit_data[completed_bank]
                        [completed_row*21 +: 21]
                        = partial_accumulator[GLOBAL_ROW]
                        + partial_value[GLOBAL_ROW];
                end else begin : G_PAD_ROW
                    assign completed_commit_data[completed_bank]
                        [completed_row*21 +: 21] = 21'sd0;
                end
            end
            (* keep_hierarchy = "yes" *)
            block0_spa_completed_sum_reg_bank u_completed_bank (
                .clk(clk),
                .commit_valid(partial_valid_bank[completed_bank]
                              && (partial_group == 4'd15)),
                .commit_record(partial_record[1:0]),
                .commit_data(completed_commit_data[completed_bank]),
                .read_record_load(rq_start),
                .read_record(rq_record[1:0]),
                .read_addr(rq_group),
                .read_data(completed_bank_read[completed_bank])
            );
        end
    endgenerate
    assign input_record_consumed=slot_release;

    // U-launch tokens are separate from activation-slot ownership.  This
    // allows the activation slot to be returned immediately after its two
    // readers finish while preserving the U-before-DT/B/C ordering contract.
    // Set has priority over retire when record N+4 reuses the same physical
    // tag slot on the cycle record N starts requantization.
    integer u_launch_index;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            u_launch_valid <= 4'b0000;
            for (u_launch_index=0; u_launch_index<4;
                 u_launch_index=u_launch_index+1)
                u_launch_record[u_launch_index] <= 8'd0;
        end else begin
            if (rq_start)
                u_launch_valid[rq_record[1:0]] <= 1'b0;
            if (u_stream_start) begin
                u_launch_valid[u_stream_record[1:0]] <= 1'b1;
                u_launch_record[u_stream_record[1:0]] <= u_stream_record;
            end
        end
    end
    integer sum_index;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            completed_valid <= 4'b0000;
        end else begin
            if (completed_pop)
                completed_valid[rq_record[1:0]] <= 1'b0;
            for (sum_index=0; sum_index<34; sum_index=sum_index+1) begin
                    // Four preserved LUT1 endpoints keep the accumulator CE
                    // local to groups of at most nine rows.  They are
                    // transparent, so arithmetic latency is unchanged.
                    if (partial_valid_bank[sum_index/9]) begin
                        if (partial_group == 0)
                            partial_accumulator[sum_index]
                                <= partial_value[sum_index];
                        else
                            partial_accumulator[sum_index]
                                <= partial_accumulator[sum_index]
                                 + partial_value[sum_index];
                    end
            end
            if (partial_valid && (partial_group == 4'd15)) begin
                completed_valid[partial_record[1:0]] <= 1'b1;
                completed_record[partial_record[1:0]] <= partial_record;
            end
        end
    end

`ifndef SYNTHESIS
    integer sim_worker_records;
    integer sim_requant_records;
    integer sim_credit_stall_cycles;
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin
            sim_worker_records<=0;
            sim_requant_records<=0;
            sim_credit_stall_cycles<=0;
        end else begin
            if(partial_valid&&(partial_group==4'd15))
                sim_worker_records<=sim_worker_records+1;
            if(completed_pop)sim_requant_records<=sim_requant_records+1;
            if(!record_credit_ready)
                sim_credit_stall_cycles<=sim_credit_stall_cycles+1;
            if(worker_releases_slot && u_releases_slot
                && (worker_record != u_stream_record))
                $fatal(1,"Spa activation released twice in one cycle: worker=%0d U=%0d",
                    worker_record,u_stream_record);
        end
    end
`endif

    wire rq_issue = rq_active;
    wire [5:0] rq_row_base = rq_group*4;
    wire signed [20:0] rq_sum [0:3];
    wire signed [15:0] rq_multiplier [0:3];
    wire signed [36:0] rq_product [0:3];
    genvar rq_lane;
    generate
        for (rq_lane=0; rq_lane<4; rq_lane=rq_lane+1) begin : G_X_RQ
            wire [5:0] row_index = rq_row_base + rq_lane;
            assign rq_sum[rq_lane] = (row_index < 34)
                ? completed_bank_read[rq_lane] : 21'sd0;
            assign rq_multiplier[rq_lane] = (row_index < 2)
                ? DT_MULTIPLIER : (row_index < 18)
                ? B_MULTIPLIER : C_MULTIPLIER;
            requant_mult_21x16 u_requant(
                .CLK(clk), .A(rq_sum[rq_lane]), .B(rq_multiplier[rq_lane]),
                .P(rq_product[rq_lane]));
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_active <= 1'b0;
            rq_record <= 8'd0;
            rq_group <= 4'd0;
        end else begin
            if (rq_start) begin
                rq_active <= 1'b1;
                rq_group <= 4'd0;
            end else if (rq_active) begin
                if (rq_group == 4'd8) begin
                    rq_active <= 1'b0;
                    rq_group <= 4'd0;
                    rq_record <= rq_record + 1'b1;
                end else begin
                    rq_group <= rq_group + 1'b1;
                end
            end
        end
    end

    reg rq_valid_d0,rq_valid_d1,rq_valid_d2;
    reg [11:0] rq_meta_d0,rq_meta_d1,rq_meta_d2;
    always @(posedge clk) begin
        if (!rst_n || frame_start) begin
            rq_valid_d0<=0; rq_valid_d1<=0; rq_valid_d2<=0;
            rq_meta_d0<=0; rq_meta_d1<=0; rq_meta_d2<=0;
        end else begin
            rq_valid_d0<=rq_issue;
            rq_valid_d1<=rq_valid_d0;
            rq_valid_d2<=rq_valid_d1;
            if (rq_issue) rq_meta_d0<={rq_record,rq_group};
            if (rq_valid_d0) rq_meta_d1<=rq_meta_d0;
            if (rq_valid_d1) rq_meta_d2<=rq_meta_d1;
        end
    end

    function [8:0] q9;
        input signed [36:0] p;
        input signed [6:0] shift;
        reg signed [63:0] v,m,r;
        begin
            v={{27{p[36]}},p}; m=v<0?-v:v;
            if(shift>0) r=(m+(64'sd1<<<(shift-1)))>>>shift;
            else if(shift<0) r=m<<<(-shift); else r=m;
            if(v<0) r=-r;
            if(r>255) q9=9'h0ff; else if(r< -256) q9=9'h100;
            else q9=r[8:0];
        end
    endfunction
    function [7:0] q8;
        input signed [36:0] p;
        input signed [6:0] shift;
        reg signed [63:0] v,m,r;
        begin
            v={{27{p[36]}},p}; m=v<0?-v:v;
            if(shift>0) r=(m+(64'sd1<<<(shift-1)))>>>shift;
            else if(shift<0) r=m<<<(-shift); else r=m;
            if(v<0) r=-r;
            if(r>127) q8=8'h7f; else if(r< -128) q8=8'h80;
            else q8=r[7:0];
        end
    endfunction

    integer output_lane;
    integer output_row;
    always @(posedge clk) begin
        if (!rst_n) begin
            dt_valid<=0; b_valid<=0; c_valid<=0; u_valid<=0; done<=0;
            dt_data<=0; b_data<=0; c_data<=0;
            u_pixel_addr<=0;u_channel_base<=0;u_data<=0;
        end else if (frame_start) begin
            dt_valid<=0; b_valid<=0; c_valid<=0; u_valid<=0; done<=0;
        end else begin
            dt_valid<=0; b_valid<=0; c_valid<=0; u_valid<=0; done<=0;
            if(u_stream_issue)begin
                u_valid<=1;
                u_pixel_addr<=u_stream_record;
                u_channel_base<=u_stream_group*8;
                u_data<=u_stream_data_q;
            end
            if (rq_valid_d2) begin
                for (output_lane=0; output_lane<4;
                     output_lane=output_lane+1) begin
                    output_row = rq_meta_d2[3:0]*4 + output_lane;
                    if (output_row < 2) begin
                        dt_data[output_row*9 +: 9]
                            <= q9(rq_product[output_lane],DT_SHIFT);
                        if (output_row == 1) begin
                            dt_valid<=1;
                            dt_pixel_addr<=rq_meta_d2[11:4];
                        end
                    end else if (output_row < 18) begin
                        b_data[(output_row-2)*8 +: 8]
                            <= q8(rq_product[output_lane],B_SHIFT);
                        if (output_row == 17) begin
                            b_valid<=1;
                            b_pixel_addr<=rq_meta_d2[11:4];
                        end
                    end else if (output_row < 34) begin
                        c_data[(output_row-18)*8 +: 8]
                            <= q8(rq_product[output_lane],C_SHIFT);
                        if (output_row == 33) begin
                            c_valid<=1;
                            c_pixel_addr<=rq_meta_d2[11:4];
                            if (rq_meta_d2[11:4] == PIXEL_COUNT-1)
                                done<=1;
                        end
                    end
                end
            end
        end
    end
endmodule


// Block0 Spa out_proj consumes each four-channel SSM group immediately.  It
// holds only 32 local partial sums instead of materializing a 64-channel
// activation record and computes four output rows per cycle after group 15.
module block0_spa_out_proj_steam #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter signed [15:0] FIXED_MULTIPLIER=16'sd30313,
    parameter signed [6:0]  FIXED_SHIFT=7'sd19
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[9:0]in_record_addr,
    input wire[5:0]in_channel_base,input wire[31:0]in_data,
    input wire cfg_weight_we,input wire[10:0]cfg_weight_addr,
    input wire signed[7:0]cfg_weight_data,input wire cfg_bias_we,
    input wire[4:0]cfg_bias_addr,input wire signed[31:0]cfg_bias_data,
    input wire signed[15:0]cfg_multiplier,input wire signed[6:0]cfg_shift,
    output reg out_valid,output reg[7:0]out_pixel_addr,
    output reg[4:0]out_channel_base,output reg[31:0]out_raw_data,
    output reg[31:0]out_data,output reg done
);
    wire unused_cfg=cfg_weight_we^cfg_weight_addr[0]^cfg_weight_data[0]
        ^cfg_bias_we^cfg_bias_addr[0]^cfg_bias_data[0]
        ^cfg_multiplier[0]^cfg_shift[0];

`include "spa_out_bias_function.vh"

    wire [3:0] input_group=in_channel_base[5:2];
    wire [1023:0] stream_weight_word;
    mamba_spa_out_stream_weight_rom #(.BLOCK_ID(BLOCK_ID))
    u_stream_weight_rom(.clk(clk),.en(in_valid),.addr(input_group),
        .data(stream_weight_word));

    reg rom_valid;
    reg [31:0]rom_activation;
    reg [13:0]rom_metadata;
    always @(posedge clk) begin
        if(!rst_n||frame_start) begin rom_valid<=0;rom_metadata<=0;end
        else begin
            rom_valid<=in_valid;
            if(in_valid) begin
                rom_activation<=in_data;
                rom_metadata<={in_record_addr,input_group};
            end
        end
    end

    wire pair_valid[0:15];
    wire signed[17:0]pair_low[0:15],pair_high[0:15];
    wire[13:0]pair_meta[0:15];
    genvar pair_index;
    generate for(pair_index=0;pair_index<16;pair_index=pair_index+1)begin:G_O_PAIR
        mamba_dot4_pair_pipeline#(.META_WIDTH(14))u_dot4_pair(
            .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
            .in_valid(rom_valid),.activation(rom_activation),
            .weight_low(stream_weight_word[(pair_index*8)*8 +:32]),
            .weight_high(stream_weight_word[(pair_index*8+4)*8 +:32]),
            .in_meta(rom_metadata),.out_valid(pair_valid[pair_index]),
            .out_sum_low(pair_low[pair_index]),
            .out_sum_high(pair_high[pair_index]),
            .out_meta(pair_meta[pair_index]));
    end endgenerate

    function signed[20:0] add_bias_21;
        input signed[20:0] accumulated;
        input signed[17:0] partial;
        input [4:0] row;
        reg signed[32:0] wide;
        reg signed[31:0] bias_value;
        begin
            bias_value=spa_out_bias(BLOCK_ID,row);
            wide=$signed({{12{accumulated[20]}},accumulated})
                +$signed({{15{partial[17]}},partial})
                +$signed({bias_value[31],bias_value});
            add_bias_21=wide[20:0];
        end
    endfunction

    wire partial_valid=pair_valid[0];
    wire partial_valid_bank[0:3];
    genvar out_valid_bank_index;
    generate
        for(out_valid_bank_index=0;out_valid_bank_index<4;
            out_valid_bank_index=out_valid_bank_index+1)begin:G_O_VALID_BANK
            ssm_local_control_buffer u_partial_valid_buffer(
                .control_in(partial_valid),
                .control_out(partial_valid_bank[out_valid_bank_index]));
        end
    endgenerate
    wire[9:0]partial_record=pair_meta[0][13:4];
    wire[3:0]partial_group=pair_meta[0][3:0];
    reg signed[20:0]partial_accumulator[0:31];
    // Input accumulation is 16 clocks/record and output requant is 8, hence
    // two result banks are lossless and halve the 32x21 wide selection mux.
    reg signed[20:0]result_sum[0:1][0:31];
    reg[1:0]result_valid;
    reg[9:0]result_record[0:1];
    reg rq_active;
    reg[9:0]rq_record;
    reg[2:0]rq_group;
    wire result_pop=rq_active&&(rq_group==3'd7);
    integer row_index;
    integer pair_number;
    reg signed[17:0]selected_partial;
    always @(posedge clk) begin
        if(!rst_n||frame_start) result_valid<=0;
        else begin
            if(result_pop)result_valid[rq_record[0]]<=0;
            for(row_index=0;row_index<32;row_index=row_index+1)begin
                    pair_number=row_index>>1;
                    selected_partial=(row_index%2)!=0
                        ? pair_high[pair_number]:pair_low[pair_number];
                    if(partial_valid_bank[row_index/8])begin
                        if(partial_group==0)
                            partial_accumulator[row_index]
                                <= {{3{selected_partial[17]}},selected_partial};
                        else
                            partial_accumulator[row_index]
                                <= partial_accumulator[row_index]+selected_partial;
                        if(partial_group==15)
                            result_sum[partial_record[0]][row_index]
                                <= add_bias_21(partial_accumulator[row_index],
                                               selected_partial,row_index);
                    end
            end
            if(partial_valid&&(partial_group==15))begin
                result_valid[partial_record[0]]<=1;
                result_record[partial_record[0]]<=partial_record;
            end
        end
    end

`ifndef SYNTHESIS
    integer sim_input_records;
    integer sim_output_records;
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin
            sim_input_records<=0;
            sim_output_records<=0;
        end else begin
            if(partial_valid&&(partial_group==4'd15))
                sim_input_records<=sim_input_records+1;
            if(result_pop)sim_output_records<=sim_output_records+1;
        end
    end
`endif

    wire rq_issue=rq_active;
    wire signed[20:0]rq_sum[0:3];
    wire signed[36:0]rq_product[0:3];
    genvar lane;
    generate for(lane=0;lane<4;lane=lane+1)begin:G_O_RQ
        assign rq_sum[lane]=result_sum[rq_record[0]][rq_group*4+lane];
        requant_mult_21x16 u_rq(.CLK(clk),.A(rq_sum[lane]),
            .B(FIXED_MULTIPLIER),.P(rq_product[lane]));
    end endgenerate
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin rq_active<=0;rq_record<=0;rq_group<=0;end
        else begin
            if(!rq_active&&result_valid[rq_record[0]]
                &&result_record[rq_record[0]]==rq_record)begin
                rq_active<=1;rq_group<=0;
            end else if(rq_active)begin
                if(rq_group==7)begin
                    rq_active<=0;rq_group<=0;
                    if(rq_record==PIXEL_COUNT-1)rq_record<=0;
                    else rq_record<=rq_record+1'b1;
                end else rq_group<=rq_group+1'b1;
            end
        end
    end

    reg rv0,rv1,rv2;
    reg[12:0]rm0,rm1,rm2;
    always @(posedge clk) begin
        if(!rst_n||frame_start)begin rv0<=0;rv1<=0;rv2<=0;rm0<=0;rm1<=0;rm2<=0;end
        else begin
            rv0<=rq_issue;rv1<=rv0;rv2<=rv1;
            if(rq_issue)rm0<={rq_record,rq_group};
            if(rv0)rm1<=rm0;if(rv1)rm2<=rm1;
        end
    end
    function[7:0]q8;
        input signed[36:0]v;
        reg signed[63:0]a,r;
        begin
            a=v<0?-$signed({{27{v[36]}},v}):$signed({{27{v[36]}},v});
            r=FIXED_SHIFT>0?(a+(64'sd1<<<(FIXED_SHIFT-1)))>>>FIXED_SHIFT:a;
            if(v<0)r=-r;
            if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];
        end
    endfunction
    integer output_lane;
    reg[7:0]raw;
    always @(posedge clk) begin
        if(!rst_n)begin out_valid<=0;out_pixel_addr<=0;out_channel_base<=0;
            out_raw_data<=0;out_data<=0;done<=0;end
        else if(frame_start)begin out_valid<=0;out_pixel_addr<=0;
            out_channel_base<=0;done<=0;end
        else begin
            out_valid<=rv2;done<=0;
            if(rv2)begin
                out_pixel_addr<=rm2[10:3];
                out_channel_base<={rm2[2:0],2'b00};
                for(output_lane=0;output_lane<4;output_lane=output_lane+1)begin
                    raw=q8(rq_product[output_lane]);
                    out_raw_data[output_lane*8+:8]<=raw;
                    out_data[output_lane*8+:8]<=raw[7]?8'd0:raw;
                end
                if(rm2[12:3]==PIXEL_COUNT-1&&rm2[2:0]==7)done<=1;
            end
        end
    end
endmodule

`default_nettype wire
