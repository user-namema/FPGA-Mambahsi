`timescale 1ns/1ps
`default_nettype none

// Block1 scratch has only 64 Spa groups / 16 Spe groups.  Keeping its
// 512/1024-bit words in a fixed-depth BRAM wastes most physical BRAM rows, so
// infer exact-depth simple-dual-port distributed RAM.  Registering only the
// read address preserves the previous one-clock memory contract without a
// WIDTH-bit output register.
module ssm_block1_scratch_lutram #(
    parameter integer WIDTH=512,
    parameter integer DEPTH=64,
    parameter integer ADDR_WIDTH=6
)(input wire clk,input wire wr_en,
  input wire[ADDR_WIDTH-1:0]wr_addr,input wire[WIDTH-1:0]wr_data,
  input wire rd_en,input wire[ADDR_WIDTH-1:0]rd_addr,
  output wire[WIDTH-1:0]rd_data);
    (* ram_style="distributed" *) reg[WIDTH-1:0]memory[0:DEPTH-1];
    reg[ADDR_WIDTH-1:0]rd_addr_q;
    always@(posedge clk)begin
        if(wr_en)memory[wr_addr]<=wr_data;
        // The matching valid pipeline qualifies rd_data.  Advance the read
        // address every clock so rd_en cannot become a wide LUTRAM CE net.
        rd_addr_q<=rd_addr;
    end
    assign rd_data=memory[rd_addr_q];
endmodule

// Block1 state storage implemented as independent 128-bit physical BRAM
// banks.  The existing 256x128 IP is intentionally reused with zero-extended
// addresses: its synchronous port-B output preserves the former registered-
// address LUTRAM's one-clock read contract.  Keeping the adapter generic also
// lets legacy 512-bit state words use four local banks without recreating a
// wide address/write-enable network in LUT fabric.
module ssm_block1_state_bram_128banked #(
    parameter integer WIDTH=128,
    parameter integer DEPTH=64,
    parameter integer ADDR_WIDTH=6
)(
    input  wire                  clk,
    input  wire                  wr_en,
    input  wire [ADDR_WIDTH-1:0] wr_addr,
    input  wire [WIDTH-1:0]      wr_data,
    input  wire                  rd_en,
    input  wire [ADDR_WIDTH-1:0] rd_addr,
    output wire [WIDTH-1:0]      rd_data
);
    localparam integer BANK_COUNT=WIDTH/128;
    wire [7:0] wr_addr_256={{(8-ADDR_WIDTH){1'b0}},wr_addr};
    wire [7:0] rd_addr_256={{(8-ADDR_WIDTH){1'b0}},rd_addr};
`ifndef SYNTHESIS
    initial begin
        if ((WIDTH % 128) != 0 || ADDR_WIDTH > 8 || DEPTH > 256) begin
            $display("Unsupported Block1 128-bit BRAM bank geometry");
            $finish;
        end
    end
`endif
    genvar bank_index;
    generate
        for(bank_index=0;bank_index<BANK_COUNT;bank_index=bank_index+1)
            begin : G_BRAM128_BANK
            ssm_scratch_bram_256x128 u_state_bank (
                .clka(clk), .ena(wr_en), .wea({wr_en}),
                .addra(wr_addr_256),
                .dina(wr_data[bank_index*128 +: 128]),
                .clkb(clk), .enb(1'b1), .addrb(rd_addr_256),
                .doutb(rd_data[bank_index*128 +: 128]));
        end
    endgenerate
    wire unused_rd_en=rd_en;
endmodule

// BRAM scratch adapter retained for Block2.  The legacy Block1 branches stay
// available for old isolated testbenches, but the active Block1 core below no
// longer instantiates them.
module ssm_scaled_scratch_bram #(
    parameter integer BLOCK_ID=1,
    parameter integer IS_SPE=0,
    parameter integer IS_AH=0
)(input wire clk,input wire wr_en,input wire[7:0]wr_addr,input wire[1023:0]wr_data,
  input wire rd_en,input wire[7:0]rd_addr,output wire[1023:0]rd_data);
    generate
        if((BLOCK_ID==1)&&(IS_AH==0))begin:G_B1_STATE
            wire[511:0]q;ssm_scratch_bram_64x512 u(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr[5:0]),.dina(wr_data[511:0]),.clkb(clk),.enb(rd_en),.addrb(rd_addr[5:0]),.doutb(q));
            assign rd_data={512'd0,q};
        end else if((BLOCK_ID==1)&&(IS_AH!=0))begin:G_B1_AH
            ssm_scratch_bram_64x1024 u(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr[5:0]),.dina(wr_data),.clkb(clk),.enb(rd_en),.addrb(rd_addr[5:0]),.doutb(rd_data));
        end else if(IS_AH==0)begin:G_B2_STATE
            wire[127:0]q0,q1;
            ssm_scratch_bram_256x128 u0(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr),.dina(wr_data[127:0]),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(q0));
            ssm_scratch_bram_256x128 u1(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr),.dina(wr_data[255:128]),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(q1));
            assign rd_data={768'd0,q1,q0};
        end else begin:G_B2_AH
            wire[255:0]q0,q1;
            ssm_scratch_bram_256x256 u0(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr),.dina(wr_data[255:0]),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(q0));
            ssm_scratch_bram_256x256 u1(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(wr_addr),.dina(wr_data[511:256]),.clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(q1));
            assign rd_data={512'd0,q1,q0};
        end
    endgenerate
endmodule

module ssm_scaled_parallel_core #(
    parameter integer BLOCK_ID=1,
    parameter integer IS_SPE=0,
    parameter integer CHANNEL_COUNT=64,
    parameter integer RECORD_COUNT=64,
    parameter integer K_FRACTION_BITS=24,
    parameter integer RESET_PERIOD=0
)(
    input wire clk,input wire rst_n,
    input wire u_valid,input wire[9:0]u_record_addr,input wire[5:0]u_channel_base,input wire[63:0]u_data,
    input wire dt_valid,input wire[9:0]dt_record_addr,input wire[5:0]dt_channel_base,input wire[63:0]dt_data,
    input wire b_valid,input wire[9:0]b_record_addr,input wire[127:0]b_data,
    input wire c_valid,input wire[9:0]c_record_addr,input wire[127:0]c_data,
    input wire cfg_lut_we,input wire[7:0]cfg_lut_addr,input wire[418:0]cfg_lut_data,
    input wire frame_start,input wire compute_start,
    output reg out_valid,output reg[9:0]out_record_addr,output reg[5:0]out_channel_base,
    output reg[191:0]out_y_q24,output reg done,output wire busy
);
    localparam integer STATE_LANES=(BLOCK_ID==1)?16:8;
    localparam integer STATE_CHUNKS=16/STATE_LANES;
    localparam integer MULTIPLIER_COUNT=STATE_LANES;
    localparam integer GROUP_COUNT=CHANNEL_COUNT*STATE_CHUNKS;
    localparam integer BLOCK1_SCRATCH_ADDR_WIDTH=(GROUP_COUNT<=16)?4:6;
    // Block1 has at least 16 groups per phase, which is longer than the
    // operand/multiplier pipeline.  Its B results are therefore committed
    // before the matching X reads, and its A results before X consumes them.
    // Keep Block2 on the conservative drained schedule.
    localparam integer CONTINUOUS_BA=(BLOCK_ID==1);
    localparam integer RECORD_ADDR_WIDTH=(RECORD_COUNT<=256)?8:10;
    localparam integer CHANNEL_BEATS=CHANNEL_COUNT/8;
    localparam integer LAST_CHANNEL_BASE=CHANNEL_COUNT-8;
    localparam[1:0]PHASE_B=0,PHASE_A=1,PHASE_X=2;
    localparam[2:0]ST_IDLE=0,ST_B=1,ST_DRAIN_B=2,ST_A=3,ST_DRAIN_A=4,ST_X=5,ST_DRAIN_X=6;

    reg[RECORD_ADDR_WIDTH:0]u_ready,dt_ready,b_ready,c_ready;
    reg[RECORD_ADDR_WIDTH-1:0]current_record;reg frame_active;reg[2:0]fsm_state;reg[7:0]request_group;
    wire u_complete=u_valid&&(u_channel_base==LAST_CHANNEL_BASE);
    wire dt_complete=dt_valid&&(dt_channel_base==LAST_CHANNEL_BASE);
    wire[RECORD_ADDR_WIDTH:0]current_ext={1'b0,current_record};
    wire[RECORD_ADDR_WIDTH:0]next_ext=current_ext+1'b1;
    wire current_ready=(u_ready>current_ext)&&(dt_ready>current_ext)&&(b_ready>current_ext)&&(c_ready>current_ext);
    wire next_ready=(u_ready>next_ext)&&(dt_ready>next_ext)&&(b_ready>next_ext)&&(c_ready>next_ext);
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin u_ready<=0;dt_ready<=0;b_ready<=0;c_ready<=0;end else begin
            if(u_complete&&u_ready<RECORD_COUNT)u_ready<=u_ready+1'b1;
            if(dt_complete&&dt_ready<RECORD_COUNT)dt_ready<=dt_ready+1'b1;
            if(b_valid&&b_ready<RECORD_COUNT)b_ready<=b_ready+1'b1;
            if(c_valid&&c_ready<RECORD_COUNT)c_ready<=c_ready+1'b1;
        end
    end

    // Keep the B/C read ports disabled while a record is still being loaded.
    // The synchronous read issued with the first phase request is available
    // before p1 drives the multiplier, so no extra PRE state is required.
    wire request_valid;
    wire runtime_read_en=request_valid;
    wire[127:0]b_word,c_word;
    ssm_bc_record_bram #(.DEPTH((RECORD_COUNT<=256)?256:1024),.ADDR_WIDTH(RECORD_ADDR_WIDTH))u_b(
        .clk(clk),.wr_en(b_valid),.wr_addr(b_record_addr[RECORD_ADDR_WIDTH-1:0]),.wr_data(b_data),.rd_en(runtime_read_en),.rd_addr(current_record),.rd_data(b_word));
    ssm_bc_record_bram #(.DEPTH((RECORD_COUNT<=256)?256:1024),.ADDR_WIDTH(RECORD_ADDR_WIDTH))u_c(
        .clk(clk),.wr_en(c_valid),.wr_addr(c_record_addr[RECORD_ADDR_WIDTH-1:0]),.wr_data(c_data),.rd_en(runtime_read_en),.rd_addr(current_record),.rd_data(c_word));
    wire[10:0]u_wr_addr=u_record_addr*CHANNEL_BEATS+(u_channel_base>>3);
    wire[10:0]dt_wr_addr=dt_record_addr*CHANNEL_BEATS+(dt_channel_base>>3);

    assign request_valid=(fsm_state==ST_B)||(fsm_state==ST_A)||(fsm_state==ST_X);
    reg[1:0]request_phase;
    always@*begin case(fsm_state)ST_A:request_phase=PHASE_A;ST_X:request_phase=PHASE_X;default:request_phase=PHASE_B;endcase end
    wire[6:0]request_channel=request_group/STATE_CHUNKS;
    wire[2:0]request_state_chunk=request_group%STATE_CHUNKS;
    wire[10:0]input_rd_addr=current_record*CHANNEL_BEATS+(request_channel>>3);
    wire[63:0]u_read_word,dt_read_word;
    ssm_udt_beat_bram u_u(.clk(clk),.wr_en(u_valid),.wr_addr(u_wr_addr),.wr_data(u_data),.rd_en(request_valid),.rd_addr(input_rd_addr),.rd_data(u_read_word));
    ssm_udt_beat_bram u_dt(.clk(clk),.wr_en(dt_valid),.wr_addr(dt_wr_addr),.wr_data(dt_data),.rd_en(request_valid),.rd_addr(input_rd_addr),.rd_data(dt_read_word));

    wire scratch_read_en=request_valid&&(request_phase!=PHASE_B);
    wire state_wr_en,ah_wr_en,bbar_wr_en;wire[7:0]result_group;
    wire[1023:0]state_read,bbar_read,ah_read,state_write,bbar_write,ah_write;
    generate if(BLOCK_ID==1)begin:G_BLOCK1_SCRATCH_LUTRAM
        wire[511:0]state_q,bbar_q;
        ssm_block1_state_bram_128banked #(
            .WIDTH(512),.DEPTH(GROUP_COUNT),
            .ADDR_WIDTH(BLOCK1_SCRATCH_ADDR_WIDTH)
        )u_state_bram(
            .clk(clk),.wr_en(state_wr_en),
            .wr_addr(result_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .wr_data(state_write[511:0]),
            .rd_en(scratch_read_en&&(request_phase==PHASE_A)),
            .rd_addr(request_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .rd_data(state_q));
        ssm_block1_scratch_lutram #(
            .WIDTH(512),.DEPTH(GROUP_COUNT),
            .ADDR_WIDTH(BLOCK1_SCRATCH_ADDR_WIDTH)
        )u_bbar_lutram(
            .clk(clk),.wr_en(bbar_wr_en),
            .wr_addr(result_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .wr_data(bbar_write[511:0]),
            .rd_en(scratch_read_en&&(request_phase==PHASE_X)),
            .rd_addr(request_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .rd_data(bbar_q));
        ssm_block1_scratch_lutram #(
            .WIDTH(1024),.DEPTH(GROUP_COUNT),
            .ADDR_WIDTH(BLOCK1_SCRATCH_ADDR_WIDTH)
        )u_ah_lutram(
            .clk(clk),.wr_en(ah_wr_en),
            .wr_addr(result_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .wr_data(ah_write),
            .rd_en(scratch_read_en&&(request_phase==PHASE_X)),
            .rd_addr(request_group[BLOCK1_SCRATCH_ADDR_WIDTH-1:0]),
            .rd_data(ah_read));
        assign state_read={512'd0,state_q};
        assign bbar_read={512'd0,bbar_q};
    end else begin:G_BLOCK2_SCRATCH_BRAM
        ssm_scaled_scratch_bram #(
            .BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE),.IS_AH(0)
        )u_state(
            .clk(clk),.wr_en(state_wr_en),.wr_addr(result_group),
            .wr_data(state_write),
            .rd_en(scratch_read_en&&(request_phase==PHASE_A)),
            .rd_addr(request_group),.rd_data(state_read));
        ssm_scaled_scratch_bram #(
            .BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE),.IS_AH(0)
        )u_bbar(
            .clk(clk),.wr_en(bbar_wr_en),.wr_addr(result_group),
            .wr_data(bbar_write),
            .rd_en(scratch_read_en&&(request_phase==PHASE_X)),
            .rd_addr(request_group),.rd_data(bbar_read));
        ssm_scaled_scratch_bram #(
            .BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE),.IS_AH(1)
        )u_ah(
            .clk(clk),.wr_en(ah_wr_en),.wr_addr(result_group),
            .wr_data(ah_write),
            .rd_en(scratch_read_en&&(request_phase==PHASE_X)),
            .rd_addr(request_group),.rd_data(ah_read));
    end endgenerate

    reg p0_valid,p1_valid;reg[7:0]p0_group,p1_group;reg[1:0]p0_phase,p1_phase;
    reg[7:0]p1_u_int8;
    reg[1023:0]scratch_hold,ah_hold;
    wire[7:0]selected_u=u_read_word[p0_group/STATE_CHUNKS%8*8 +:8];
    wire[7:0]selected_dt=dt_read_word[p0_group/STATE_CHUNKS%8*8 +:8];
    function[7:0]int8_lut_address;input[7:0]code;begin int8_lut_address={~code[7],code[6:0]};end endfunction
    wire[7:0]lut_addr=int8_lut_address((p0_phase==PHASE_X)?selected_u:selected_dt);
    wire[418:0]lut_word;
    mamba_ssm_lut_rom_1r #(.BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE))u_lut(.clk(clk),.en(p0_valid),.addr(lut_addr),.data(lut_word));
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin p0_valid<=0;p1_valid<=0;p0_group<=0;p1_group<=0;p0_phase<=0;p1_phase<=0;p1_u_int8<=0;scratch_hold<=0;ah_hold<=0;end else begin
            p0_valid<=request_valid;p1_valid<=p0_valid;
            if(request_valid)begin p0_group<=request_group;p0_phase<=request_phase;end
            if(p0_valid)begin p1_group<=p0_group;p1_phase<=p0_phase;p1_u_int8<=selected_u;scratch_hold<=(p0_phase==PHASE_A)?state_read:bbar_read;ah_hold<=ah_read;end
        end
    end

    wire issue_valid=p1_valid;wire[2:0]issue_state_chunk=p1_group%STATE_CHUNKS;
    wire[4:0]issue_state_base=issue_state_chunk*STATE_LANES;
    wire reset_state=(RESET_PERIOD==4)?(current_record[1:0]==0):(current_record==0);
    wire[18:0]lut_k=lut_word[418:400];
    wire signed[32:0]mult_a[0:MULTIPLIER_COUNT-1];wire signed[31:0]mult_b[0:MULTIPLIER_COUNT-1];
    wire signed[64:0]mult_p[0:MULTIPLIER_COUNT-1];genvar i;
    generate for(i=0;i<MULTIPLIER_COUNT;i=i+1)begin:G_MULT
        localparam integer LANE=i;
        wire[4:0]state_index=issue_state_base+LANE;
        assign mult_a[i]=(p1_phase==PHASE_A)?{8'd0,lut_word[state_index*25+:25]}:
                         (p1_phase==PHASE_X)?{scratch_hold[i*32+31],scratch_hold[i*32+:32]}:{14'd0,lut_k};
        assign mult_b[i]=(p1_phase==PHASE_A)?(reset_state?32'sd0:$signed(scratch_hold[i*32+:32])):
                         (p1_phase==PHASE_X)?{{24{p1_u_int8[7]}},p1_u_int8}:{{24{b_word[state_index*8+7]}},b_word[state_index*8+:8]};
        ssm_mult_2dsp u(.CLK(clk),.VALID(issue_valid),.WIDE_MODE(p1_phase==PHASE_A),.A(mult_a[i]),.B(mult_b[i]),.P(mult_p[i]));
    end endgenerate

    reg mv0,mv1,mv2;reg[1:0]mp0,mp1,mp2;reg[7:0]mg0,mg1,mg2;reg[RECORD_ADDR_WIDTH-1:0]mr0,mr1,mr2;
    reg[1023:0]ah_d0,ah_d1,ah_d2;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin mv0<=0;mv1<=0;mv2<=0;mp0<=0;mp1<=0;mp2<=0;mg0<=0;mg1<=0;mg2<=0;mr0<=0;mr1<=0;mr2<=0;ah_d0<=0;ah_d1<=0;ah_d2<=0;end else begin
            mv0<=issue_valid;mv1<=mv0;mv2<=mv1;
            if(issue_valid)begin mp0<=p1_phase;mg0<=p1_group;mr0<=current_record;ah_d0<=ah_hold;end
            if(mv0)begin mp1<=mp0;mg1<=mg0;mr1<=mr0;ah_d1<=ah_d0;end
            if(mv1)begin mp2<=mp1;mg2<=mg1;mr2<=mr1;ah_d2<=ah_d1;end
        end
    end
    assign result_group=mg2;
    function signed[31:0]round_shift_to_int32;input signed[64:0]value;input integer sh;reg signed[65:0]mag,rnd,srnd;begin
        mag=(value<0)?-$signed({value[64],value}):$signed({1'b0,value});
        if(sh>0)rnd=(mag+(66'sd1<<<(sh-1)))>>>sh;else rnd=mag<<<(-sh);srnd=(value<0)?-rnd:rnd;
        if(srnd[65:31]=={35{srnd[31]}})round_shift_to_int32=srnd[31:0];else round_shift_to_int32=srnd[65]?32'sh80000000:32'sh7fffffff;end endfunction
    localparam integer KBU_TO_Q48_SHIFT=48-K_FRACTION_BITS;
    wire signed[64:0]state_acc[0:MULTIPLIER_COUNT-1];wire signed[64:0]kbu_q48[0:MULTIPLIER_COUNT-1];wire signed[31:0]kb_next[0:MULTIPLIER_COUNT-1];
    wire signed[31:0]state_next[0:MULTIPLIER_COUNT-1];wire signed[39:0]c_product[0:15];
    generate for(i=0;i<MULTIPLIER_COUNT;i=i+1)begin:G_RESULT
        localparam integer LANE=i;
        wire[4:0]rstate=(mg2%STATE_CHUNKS)*STATE_LANES+LANE;
        assign kbu_q48[i]=$signed(mult_p[i])<<<KBU_TO_Q48_SHIFT;
        assign state_acc[i]=$signed({ah_d2[i*64+63],ah_d2[i*64+:64]})+kbu_q48[i];
        assign kb_next[i]=mult_p[i][31:0];
        assign state_next[i]=round_shift_to_int32(state_acc[i],24);
        assign bbar_write[i*32+:32]=kb_next[i];assign ah_write[i*64+:64]=mult_p[i][63:0];assign state_write[i*32+:32]=state_next[i];
        (* use_dsp="yes" *) wire signed[39:0]cp=state_next[i]*$signed(c_word[rstate*8+:8]);assign c_product[i]=cp;
    end endgenerate
    generate if(MULTIPLIER_COUNT==4)begin:G_PAD4
        assign bbar_write[1023:128]=896'd0;assign ah_write[1023:256]=768'd0;assign state_write[1023:128]=896'd0;
        for(i=4;i<16;i=i+1)begin:G_ZERO assign c_product[i]=40'sd0;end
    end else if(MULTIPLIER_COUNT==8)begin:G_PAD8
        assign bbar_write[1023:256]=768'd0;assign ah_write[1023:512]=512'd0;assign state_write[1023:256]=768'd0;
        for(i=8;i<16;i=i+1)begin:G_ZERO assign c_product[i]=40'sd0;end
    end else begin:G_FULL16
        assign bbar_write[1023:512]=512'd0;assign state_write[1023:512]=512'd0;
    end endgenerate
    assign bbar_wr_en=mv2&&(mp2==PHASE_B);assign ah_wr_en=mv2&&(mp2==PHASE_A);assign state_wr_en=mv2&&(mp2==PHASE_X);

    reg signed[39:0]cp0[0:15];reg signed[40:0]yl1[0:7];reg signed[41:0]yl2[0:3];reg signed[42:0]yl3[0:1];reg signed[43:0]yl4;
    reg yv0,yv1,yv2,yv3,yv4;reg[7:0]yg0,yg1,yg2,yg3,yg4;reg[RECORD_ADDR_WIDTH-1:0]yr0,yr1,yr2,yr3,yr4;integer k;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin yv0<=0;yv1<=0;yv2<=0;yv3<=0;yv4<=0;yg0<=0;yg1<=0;yg2<=0;yg3<=0;yg4<=0;yr0<=0;yr1<=0;yr2<=0;yr3<=0;yr4<=0;end else begin
            yv0<=mv2&&(mp2==PHASE_X);yv1<=yv0;yv2<=yv1;yv3<=yv2;yv4<=yv3;
            if(mv2&&(mp2==PHASE_X))begin yg0<=mg2;yr0<=mr2;for(k=0;k<16;k=k+1)cp0[k]<=c_product[k];end
            if(yv0)begin yg1<=yg0;yr1<=yr0;for(k=0;k<8;k=k+1)yl1[k]<=$signed({cp0[k*2][39],cp0[k*2]})+$signed({cp0[k*2+1][39],cp0[k*2+1]});end
            if(yv1)begin yg2<=yg1;yr2<=yr1;for(k=0;k<4;k=k+1)yl2[k]<=$signed({yl1[k*2][40],yl1[k*2]})+$signed({yl1[k*2+1][40],yl1[k*2+1]});end
            if(yv2)begin yg3<=yg2;yr3<=yr2;for(k=0;k<2;k=k+1)yl3[k]<=$signed({yl2[k*2][41],yl2[k*2]})+$signed({yl2[k*2+1][41],yl2[k*2+1]});end
            if(yv3)begin yg4<=yg3;yr4<=yr3;yl4<=$signed({yl3[0][42],yl3[0]})+$signed({yl3[1][42],yl3[1]});end
        end
    end
    wire[6:0]y_channel=yg4/STATE_CHUNKS;wire[2:0]y_state_chunk=yg4%STATE_CHUNKS;
    wire signed[47:0]partial_y={{4{yl4[43]}},yl4};reg signed[47:0]state_y_acc;reg[143:0]out_pack;
    wire signed[47:0]complete_y=(y_state_chunk==0)?partial_y:(state_y_acc+partial_y);
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin out_valid<=0;done<=0;out_record_addr<=0;out_channel_base<=0;out_y_q24<=0;state_y_acc<=0;out_pack<=0;end else begin
            out_valid<=0;done<=0;
            if(yv4)begin
                if(y_state_chunk==0)state_y_acc<=partial_y;else state_y_acc<=state_y_acc+partial_y;
                if(y_state_chunk==STATE_CHUNKS-1)begin
                    if(y_channel[1:0]==3)begin
                        out_valid<=1;out_record_addr<=yr4;out_channel_base<={y_channel[5:2],2'b00};out_y_q24<={complete_y,out_pack};
                        if((yr4==RECORD_COUNT-1)&&(y_channel==CHANNEL_COUNT-1))done<=1;
                    end else out_pack[y_channel[1:0]*48+:48]<=complete_y;
                end
            end
        end
    end

    always@(posedge clk)begin
        if(!rst_n||frame_start)begin fsm_state<=ST_IDLE;request_group<=0;current_record<=0;frame_active<=rst_n&&frame_start;end else begin
            if(compute_start&&!frame_active)begin frame_active<=1;current_record<=0;request_group<=0;fsm_state<=ST_IDLE;end
            else case(fsm_state)
                ST_IDLE:begin request_group<=0;if(frame_active&&current_ready)fsm_state<=ST_B;end
                ST_B:if(request_group==GROUP_COUNT-1)begin request_group<=0;if(CONTINUOUS_BA)fsm_state<=ST_A;else fsm_state<=ST_DRAIN_B;end else request_group<=request_group+1'b1;
                ST_DRAIN_B:if(mv2&&(mp2==PHASE_B)&&(mg2==GROUP_COUNT-1))begin request_group<=0;fsm_state<=ST_A;end
                ST_A:if(request_group==GROUP_COUNT-1)begin request_group<=0;if(CONTINUOUS_BA)fsm_state<=ST_X;else fsm_state<=ST_DRAIN_A;end else request_group<=request_group+1'b1;
                ST_DRAIN_A:if(mv2&&(mp2==PHASE_A)&&(mg2==GROUP_COUNT-1))begin request_group<=0;fsm_state<=ST_X;end
                ST_X:if(request_group==GROUP_COUNT-1)begin request_group<=0;fsm_state<=ST_DRAIN_X;end else request_group<=request_group+1'b1;
                ST_DRAIN_X:if(mv2&&(mp2==PHASE_X)&&(mg2==GROUP_COUNT-1))begin request_group<=0;if(current_record==RECORD_COUNT-1)begin fsm_state<=ST_IDLE;frame_active<=0;end else begin current_record<=current_record+1'b1;if(next_ready)fsm_state<=ST_B;else fsm_state<=ST_IDLE;end end
                default:fsm_state<=ST_IDLE;
            endcase
        end
    end
    assign busy=frame_active||(fsm_state!=ST_IDLE)||p0_valid||p1_valid||mv0||mv1||mv2||yv0||yv1||yv2||yv3||yv4||out_valid;
    wire unused_cfg=cfg_lut_we^cfg_lut_addr[0]^cfg_lut_data[0];
endmodule

// Compatibility wrapper.  All six branches use the fused 3-DSP state update;
// Block0 has 64 lanes, Block1 retains 16 lanes, and Block2 retains 8 lanes.
module ssm64_parallel_core #(
    parameter integer BLOCK_ID=0,parameter integer IS_SPE=0,parameter integer CHANNEL_COUNT=64,
    parameter integer RECORD_COUNT=256,parameter integer GROUP_COUNT=16,parameter integer K_FRACTION_BITS=24,
    parameter integer RESET_PERIOD=0,parameter integer STREAM_INPUT=0,
    parameter integer CONTINUOUS_TILES=0
)(input wire clk,input wire rst_n,input wire u_valid,input wire[9:0]u_record_addr,input wire[5:0]u_channel_base,input wire[63:0]u_data,
input wire dt_valid,input wire[9:0]dt_record_addr,input wire[5:0]dt_channel_base,input wire[63:0]dt_data,input wire b_valid,input wire[9:0]b_record_addr,
input wire[127:0]b_data,input wire c_valid,input wire[9:0]c_record_addr,input wire[127:0]c_data,input wire cfg_lut_we,input wire[7:0]cfg_lut_addr,
input wire[418:0]cfg_lut_data,input wire frame_start,input wire compute_start,output wire out_valid,output wire[9:0]out_record_addr,
output wire[5:0]out_channel_base,output wire[191:0]out_y_q24,output wire done,output wire busy,
output wire input_record_ready,output wire input_record_consumed);
    generate if(BLOCK_ID==0)begin:G0_FUSED
        ssm64_parallel_core_block0_spa_fused #(.BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE),.CHANNEL_COUNT(CHANNEL_COUNT),.RECORD_COUNT(RECORD_COUNT),.GROUP_COUNT(GROUP_COUNT),.K_FRACTION_BITS(K_FRACTION_BITS),.RESET_PERIOD(RESET_PERIOD),.STREAM_INPUT(STREAM_INPUT),.CONTINUOUS_TILES(CONTINUOUS_TILES))u(
            .clk(clk),.rst_n(rst_n),.u_valid(u_valid),.u_record_addr(u_record_addr),.u_channel_base(u_channel_base),.u_data(u_data),.dt_valid(dt_valid),.dt_record_addr(dt_record_addr),.dt_channel_base(dt_channel_base),.dt_data(dt_data),
            .b_valid(b_valid),.b_record_addr(b_record_addr),.b_data(b_data),.c_valid(c_valid),.c_record_addr(c_record_addr),.c_data(c_data),.cfg_lut_we(cfg_lut_we),.cfg_lut_addr(cfg_lut_addr),.cfg_lut_data(cfg_lut_data),
            .frame_start(frame_start),.compute_start(compute_start),.out_valid(out_valid),.out_record_addr(out_record_addr),.out_channel_base(out_channel_base),.out_y_q24(out_y_q24),.done(done),.busy(busy),.input_record_ready(input_record_ready),.input_record_consumed(input_record_consumed));
    end else if((BLOCK_ID==1)&&(IS_SPE!=0))begin:G1_SPE_DUAL
        assign input_record_ready=1'b1;
        assign input_record_consumed=1'b0;
        ssm_block1_spe_dual_fused_core #(.RECORD_COUNT(RECORD_COUNT),.K_FRACTION_BITS(K_FRACTION_BITS),.RESET_PERIOD(RESET_PERIOD))u(
            .clk(clk),.rst_n(rst_n),.u_valid(u_valid),.u_record_addr(u_record_addr),.u_channel_base(u_channel_base),.u_data(u_data),.dt_valid(dt_valid),.dt_record_addr(dt_record_addr),.dt_channel_base(dt_channel_base),.dt_data(dt_data),
            .b_valid(b_valid),.b_record_addr(b_record_addr),.b_data(b_data),.c_valid(c_valid),.c_record_addr(c_record_addr),.c_data(c_data),.cfg_lut_we(cfg_lut_we),.cfg_lut_addr(cfg_lut_addr),.cfg_lut_data(cfg_lut_data),
            .frame_start(frame_start),.compute_start(compute_start),.out_valid(out_valid),.out_record_addr(out_record_addr),.out_channel_base(out_channel_base),.out_y_q24(out_y_q24),.done(done),.busy(busy));
    end else begin:GS_FUSED
        assign input_record_ready=1'b1;
        assign input_record_consumed=1'b0;
        ssm_scaled_fused_core #(.BLOCK_ID(BLOCK_ID),.IS_SPE(IS_SPE),.CHANNEL_COUNT(CHANNEL_COUNT),.RECORD_COUNT(RECORD_COUNT),.K_FRACTION_BITS(K_FRACTION_BITS),.RESET_PERIOD(RESET_PERIOD))u(
            .clk(clk),.rst_n(rst_n),.u_valid(u_valid),.u_record_addr(u_record_addr),.u_channel_base(u_channel_base),.u_data(u_data),.dt_valid(dt_valid),.dt_record_addr(dt_record_addr),.dt_channel_base(dt_channel_base),.dt_data(dt_data),
            .b_valid(b_valid),.b_record_addr(b_record_addr),.b_data(b_data),.c_valid(c_valid),.c_record_addr(c_record_addr),.c_data(c_data),.cfg_lut_we(cfg_lut_we),.cfg_lut_addr(cfg_lut_addr),.cfg_lut_data(cfg_lut_data),
            .frame_start(frame_start),.compute_start(compute_start),.out_valid(out_valid),.out_record_addr(out_record_addr),.out_channel_base(out_channel_base),.out_y_q24(out_y_q24),.done(done),.busy(busy));
    end endgenerate
endmodule

`default_nettype wire
