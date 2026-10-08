`timescale 1ns/1ps
`default_nettype none

// Streaming 2x2 / stride-2 pooling hand-off used between Mamba blocks.
// The QAT/FPGA contract keeps the signed integer sum and divides the physical
// scale by four.  One existing 256x256 simple-dual-port BRAM stores complete
// input pixels.  As soon as the bottom-right source pixel is present, the
// corresponding pooled pixel is read and emitted; the whole feature map does
// not need to be received first.
module avgpool2x2_sum_stream #(
    parameter integer IN_SIDE=16,
    parameter integer UNCONDITIONAL_PAYLOAD=0
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[7:0]in_pixel_addr,
    input wire[4:0]in_channel_base,input wire[63:0]in_data,
    output reg out_valid,output reg[7:0]out_pixel_addr,
    output reg[4:0]out_channel_base,output reg[39:0]out_data,
    output reg done
);
    localparam integer OUT_SIDE=IN_SIDE/2;
    localparam integer OUT_PIXELS=OUT_SIDE*OUT_SIDE;
    localparam[3:0] ST_WAIT=0,ST_R0=1,ST_C0=2,ST_R1=3,ST_C1=4,
        ST_R2=5,ST_C2=6,ST_R3=7,ST_C3=8,ST_SUM=9,ST_OUT=10;

    reg[191:0]collect;
    wire pixel_complete=in_valid&&(in_channel_base==5'd24);
    wire[255:0]pixel_write_data={in_data,collect};
    localparam integer IN_PIXEL_SHIFT=$clog2(IN_SIDE*IN_SIDE);
    reg[31:0]written_count;
    wire write_context=written_count[IN_PIXEL_SHIFT];

    reg rd_en;
    reg[7:0]rd_addr;
    reg read_context_q;
    wire[255:0]rd_data_ctx0,rd_data_ctx1;
    wire[255:0]rd_data=read_context_q?rd_data_ctx1:rd_data_ctx0;
    mamba_input_bram_256x256 u_pixel_bram_ctx0(
        .clka(clk),.ena(pixel_complete&&!write_context),
        .wea({pixel_complete&&!write_context}),
        .addra(in_pixel_addr),.dina(pixel_write_data),
        .clkb(clk),.enb(rd_en&&!read_context_q),
        .addrb(rd_addr),.doutb(rd_data_ctx0));
    mamba_input_bram_256x256 u_pixel_bram_ctx1(
        .clka(clk),.ena(pixel_complete&&write_context),
        .wea({pixel_complete&&write_context}),
        .addra(in_pixel_addr),.dina(pixel_write_data),
        .clkb(clk),.enb(rd_en&&read_context_q),
        .addrb(rd_addr),.doutb(rd_data_ctx1));

    // collect is payload: it is overwritten beat-by-beat before
    // pixel_complete consumes it.  Only the completion counter needs reset;
    // leaving collect free of reset avoids a 192-bit frame_start/reset net.
    always@(posedge clk)begin
        if(!rst_n||frame_start)
            written_count<=0;
        else if(pixel_complete)
            written_count<=written_count+1'b1;

        if(in_valid&&(in_channel_base!=5'd24))
            collect[in_channel_base*8+:64]<=in_data;
    end

    reg[3:0]state;
    reg[7:0]out_index;
    reg[31:0]out_tile_sequence;
    reg[2:0]out_beat;
    // Unconditional read-data pipeline.  With the one wait state between
    // capture states, the four 2x2 pixels are rd_pipe5, rd_pipe3,
    // rd_pipe1 and rd_data when ST_C3 executes.  Payload registers therefore
    // have neither reset nor FSM/frame_start clock-enable inputs.
    reg[255:0]rd_pipe0,rd_pipe1,rd_pipe2,rd_pipe3,rd_pipe4,rd_pipe5;
    reg signed[8:0]pair01[0:31];
    reg signed[8:0]pair23[0:31];
    // Four signed INT10 sums per beat; channel count and timing are unchanged.
    reg[319:0]pool_word;
    integer lane;
    integer payload_lane;

    wire[7:0]pool_row=out_index/OUT_SIDE;
    wire[7:0]pool_col=out_index%OUT_SIDE;
    wire[7:0]source_row={pool_row[6:0],1'b0};
    wire[7:0]source_col={pool_col[6:0],1'b0};
    wire[7:0]addr0=source_row*IN_SIDE+source_col;
    wire[7:0]addr1=addr0+1'b1;
    wire[7:0]addr2=addr0+IN_SIDE;
    wire[7:0]addr3=addr2+1'b1;
    // Scene sequence, not a RAM address. Keep the arithmetic at 32 bits;
    // the physical BRAM address and ping-pong context remain unchanged.
    wire[31:0]tile_written_base = out_tile_sequence << IN_PIXEL_SHIFT;
    wire[31:0]required_written = tile_written_base + {24'd0,addr3} + 32'd1;
    wire current_ready=(written_count>=required_written);

    function[9:0]sum_to_s10;
        input signed[8:0]a;
        input signed[8:0]b;
        reg signed[9:0]s;
        begin
            s=$signed({a[8],a})+$signed({b[8],b});
            sum_to_s10=s;
        end
    endfunction

    generate
        if (UNCONDITIONAL_PAYLOAD != 0) begin : G_UNCONDITIONAL_PAYLOAD
            always @(posedge clk) begin
                rd_pipe0 <= rd_data;
                rd_pipe1 <= rd_pipe0;
                rd_pipe2 <= rd_pipe1;
                rd_pipe3 <= rd_pipe2;
                rd_pipe4 <= rd_pipe3;
                rd_pipe5 <= rd_pipe4;

                // The BRAM payload and first arithmetic level run every
                // cycle.  ST_SUM captures one complete pooled pixel; that
                // 320-bit word must remain stable through eight 40-bit beats.
                for (payload_lane=0;payload_lane<32;
                     payload_lane=payload_lane+1) begin
                    pair01[payload_lane]
                        <= $signed(rd_pipe5[payload_lane*8+:8])
                         + $signed(rd_pipe3[payload_lane*8+:8]);
                    pair23[payload_lane]
                        <= $signed(rd_pipe1[payload_lane*8+:8])
                         + $signed(rd_data[payload_lane*8+:8]);
                end

                if (state == ST_SUM) begin
                    for (payload_lane=0;payload_lane<32;
                         payload_lane=payload_lane+1)
                        pool_word[payload_lane*10+:10]
                            <= sum_to_s10(pair01[payload_lane],pair23[payload_lane]);
                end

                // Output payload is intentionally unconditional.  Its value
                // is architecturally visible only when out_valid is high.
                out_pixel_addr <= out_index;
                out_channel_base <= {out_beat,2'b00};
                out_data <= pool_word[out_beat*40+:40];
            end
        end else begin : G_LEGACY_CAPTURE_PAYLOAD
            // Pool1 keeps its previous capture structure; the registers are
            // placed outside the frame_start reset branch so frame_start is
            // still absent from their control sets.
            reg [255:0] pixel0,pixel1,pixel2;
            integer legacy_lane;
            always @(posedge clk) begin
                if (state == ST_C0) pixel0 <= rd_data;
                if (state == ST_C1) pixel1 <= rd_data;
                if (state == ST_C2) pixel2 <= rd_data;
                if (state == ST_C3) begin
                    for (legacy_lane=0;legacy_lane<32;
                         legacy_lane=legacy_lane+1) begin
                        pair01[legacy_lane]
                            <= $signed(pixel0[legacy_lane*8+:8])
                             + $signed(pixel1[legacy_lane*8+:8]);
                        pair23[legacy_lane]
                            <= $signed(pixel2[legacy_lane*8+:8])
                             + $signed(rd_data[legacy_lane*8+:8]);
                    end
                end
                if (state == ST_SUM) begin
                    for (legacy_lane=0;legacy_lane<32;
                         legacy_lane=legacy_lane+1)
                        pool_word[legacy_lane*10+:10]
                            <= sum_to_s10(pair01[legacy_lane],pair23[legacy_lane]);
                end
                if (state == ST_OUT) begin
                    out_pixel_addr <= out_index;
                    out_channel_base <= {out_beat,2'b00};
                    out_data <= pool_word[out_beat*40+:40];
                end
            end
        end
    endgenerate

    always@(posedge clk)begin
        if(!rst_n||frame_start)begin
            state<=ST_WAIT;out_index<=0;out_tile_sequence<=0;out_beat<=0;
            rd_en<=0;rd_addr<=0;read_context_q<=0;out_valid<=0;done<=0;
        end else begin
            rd_en<=0;out_valid<=0;done<=0;
            case(state)
                ST_WAIT:begin
                    out_beat<=0;
                    if((out_index<OUT_PIXELS)&&current_ready)begin
                        rd_en<=1;rd_addr<=addr0;
                        read_context_q<=out_tile_sequence[0];state<=ST_R0;
                    end
                end
                ST_R0:state<=ST_C0;
                ST_C0:begin rd_en<=1;rd_addr<=addr1;state<=ST_R1;end
                ST_R1:state<=ST_C1;
                ST_C1:begin rd_en<=1;rd_addr<=addr2;state<=ST_R2;end
                ST_R2:state<=ST_C2;
                ST_C2:begin rd_en<=1;rd_addr<=addr3;state<=ST_R3;end
                ST_R3:state<=ST_C3;
                ST_C3:state<=ST_SUM;
                ST_SUM:state<=ST_OUT;
                ST_OUT:begin
                    out_valid<=1;
                    if(out_beat==7)begin
                        out_beat<=0;
                        if(out_index==OUT_PIXELS-1)begin
                            done<=1;out_index<=0;
                            out_tile_sequence<=out_tile_sequence+1'b1;
                            state<=ST_WAIT;
                        end
                        else begin out_index<=out_index+1'b1;state<=ST_WAIT;end
                    end else out_beat<=out_beat+1'b1;
                end
                default:state<=ST_WAIT;
            endcase
        end
    end

`ifndef SYNTHESIS
    reg signed [10:0] checked_sum;
    integer wide_sum_reports=0;
    always@(posedge clk)begin
        if(rst_n&&(state==ST_SUM))begin
            for(lane=0;lane<32;lane=lane+1)begin
                checked_sum=$signed({{2{pair01[lane][8]}},pair01[lane]})
                           +$signed({{2{pair23[lane][8]}},pair23[lane]});
                if ((^checked_sum)===1'bx || checked_sum>508 || checked_sum< -512)
                    $fatal(1,"Pool INT10 contract violation: tile=%0d output=%0d channel=%0d sum=%0d",
                        out_tile_sequence,out_index,lane,checked_sum);
                if ((checked_sum>127 || checked_sum< -128) && wide_sum_reports<8) begin
                    $display("POOL10 preserved: side=%0d tile=%0d output=%0d channel=%0d sum=%0d",
                        IN_SIDE,out_tile_sequence,out_index,lane,checked_sum);
                    wide_sum_reports=wide_sum_reports+1;
                end
            end
        end
    end
`endif
endmodule

`default_nettype wire
