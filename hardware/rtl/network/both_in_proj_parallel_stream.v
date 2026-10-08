`timescale 1ns/1ps
`default_nettype none

// DSP-tail in_proj shared by all blocks.
// Block0 uses eight output rows/cycle: (8 issue + 3 BRAM control) * 256
// = 2816 service cycles.  Block1/2 use two rows/cycle and need 2240/560
// cycles respectively.  All three are below 4000 cycles at 100 MHz.
module both_in_proj_parallel_stream #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=256,
    parameter integer OUTPUT_LANES=(BLOCK_ID==0)?8:2,
    parameter signed[15:0]SPA_INPUT_MULTIPLIER=(BLOCK_ID==0)?16'sd32741:(BLOCK_ID==1)?16'sd32576:16'sd16391,
    parameter signed[6:0]SPA_INPUT_SHIFT=(BLOCK_ID==0)?7'sd15:(BLOCK_ID==1)?7'sd17:7'sd16,
    parameter signed[15:0]SPE_INPUT_MULTIPLIER=(BLOCK_ID==0)?16'sd16406:(BLOCK_ID==1)?16'sd32588:16'sd32756,
    parameter signed[6:0]SPE_INPUT_SHIFT=(BLOCK_ID==0)?7'sd14:(BLOCK_ID==1)?7'sd17:7'sd17,
    parameter signed[15:0]SPA_REQUANT_MULTIPLIER=16'sd18177,
    parameter signed[6:0]SPA_REQUANT_SHIFT=7'sd20,
    parameter signed[15:0]SPE_REQUANT_MULTIPLIER=16'sd23177,
    parameter signed[6:0]SPE_REQUANT_SHIFT=7'sd19
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire record_credit_ready,
    input wire patch_valid,input wire[7:0]patch_pixel_addr,
    input wire[4:0]patch_channel_base,input wire[((BLOCK_ID==0)?32:40)-1:0]patch_data,
    input wire spa_weight_we,input wire[10:0]spa_weight_addr,input wire signed[7:0]spa_weight_data,
    input wire signed[31:0]spa_requant_multiplier,input wire signed[6:0]spa_requant_shift,
    input wire spe_weight_we,input wire[6:0]spe_weight_addr,input wire signed[7:0]spe_weight_data,
    input wire signed[31:0]spe_requant_multiplier,input wire signed[6:0]spe_requant_shift,
    output wire record_launch,
    output reg spa_out_valid,output reg[7:0]spa_out_pixel_addr,
    output reg[5:0]spa_out_channel_base,output reg[63:0]spa_out_data,output reg spa_done,
    output reg spe_out_valid,output reg[7:0]spe_out_pixel_addr,output reg[1:0]spe_out_token,
    output reg[3:0]spe_out_channel_base,output reg[63:0]spe_out_data,output reg spe_done
);
    localparam integer ISSUE_STEPS=64/OUTPUT_LANES;
    localparam integer SPE_GROUPS=16/OUTPUT_LANES;
    wire unused_cfg=spa_weight_we^spa_weight_addr[0]^spa_weight_data[0]
        ^spa_requant_multiplier[0]^spa_requant_shift[0]^spe_weight_we
        ^spe_weight_addr[0]^spe_weight_data[0]^spe_requant_multiplier[0]
        ^spe_requant_shift[0];

    // AvgPool exports four complete signed INT10 sums, not saturated INT8.
    // Four 1024x8 distributed-ROM reads per Spa/Spe observer perform the
    // original v5 multiplier/round/saturate mapping before the INT8 input RAM.
    // Keep the existing LUT-output register and valid/metadata latency.
    // D1: all branches use one registered input-LUT cycle; record II is unchanged.
    wire [31:0] spa_lut_data;
    wire [31:0] spe_lut_data;
    reg [31:0] spa_lut_data_q;
    reg [31:0] spe_lut_data_q;
    reg input_lut_valid_q;
    reg [7:0] input_lut_pixel_q;
    reg [4:0] input_lut_channel_q;
    genvar ql;
    generate
        if (BLOCK_ID == 0) begin : G_INPUT_Q_BYPASS
            for (ql=0; ql<4; ql=ql+1) begin : G_BYTE
                mamba_pool_q8_lut_1024x8 #(
                    .MULTIPLIER(SPA_INPUT_MULTIPLIER), .SHIFT(SPA_INPUT_SHIFT)
                ) u_spa_lut (
                    .address({{2{patch_data[ql*8+7]}},patch_data[ql*8 +: 8]}),
                    .data(spa_lut_data[ql*8 +: 8]));
                mamba_pool_q8_lut_1024x8 #(
                    .MULTIPLIER(SPE_INPUT_MULTIPLIER), .SHIFT(SPE_INPUT_SHIFT)
                ) u_spe_lut (
                    .address({{2{patch_data[ql*8+7]}},patch_data[ql*8 +: 8]}),
                    .data(spe_lut_data[ql*8 +: 8]));
            end
        end else begin : G_INPUT_Q_LUT
            for (ql=0; ql<4; ql=ql+1) begin : G_BYTE
                mamba_pool_q8_lut_1024x8 #(
                    .MULTIPLIER(SPA_INPUT_MULTIPLIER), .SHIFT(SPA_INPUT_SHIFT)
                ) u_spa_lut (
                    .address(patch_data[ql*10 +: 10]),
                    .data(spa_lut_data[ql*8 +: 8]));
                mamba_pool_q8_lut_1024x8 #(
                    .MULTIPLIER(SPE_INPUT_MULTIPLIER), .SHIFT(SPE_INPUT_SHIFT)
                ) u_spe_lut (
                    .address(patch_data[ql*10 +: 10]),
                    .data(spe_lut_data[ql*8 +: 8]));
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n || frame_start)
            input_lut_valid_q <= 1'b0;
        else
            input_lut_valid_q <= patch_valid;
        // Lookup data and metadata are free-running payload.  Only the valid
        // bit is reset/qualified; avoiding patch_valid as a 64-bit payload CE
        // keeps this new boundary register local and inexpensive to route.
        spa_lut_data_q <= spa_lut_data;
        spe_lut_data_q <= spe_lut_data;
        input_lut_pixel_q <= patch_pixel_addr;
        input_lut_channel_q <= patch_channel_base;
    end

    wire input_q_valid = input_lut_valid_q;
    wire [7:0] input_q_pixel = input_lut_pixel_q;
    wire [4:0] input_q_channel = input_lut_channel_q;
    wire [31:0] spa_patch_q = spa_lut_data_q;
    wire [31:0] spe_patch_q = spe_lut_data_q;

    reg[255:0]spa_collect_pixel,spe_collect_pixel;
    wire wr_en=input_q_valid&&(input_q_channel==28);
    wire[255:0]spa_wr_data={spa_patch_q,spa_collect_pixel[223:0]};
    wire[255:0]spe_wr_data={spe_patch_q,spe_collect_pixel[223:0]};
    // Two physical tile contexts.  The producer and consumer use independent
    // monotonically increasing sequence counters; the context bit is the bit
    // immediately above the power-of-two pixel address.  This lets tile N+1
    // fill low addresses while tile N is still draining its high addresses.
    localparam integer PIXEL_SHIFT=$clog2(PIXEL_COUNT);
    reg rd_en;reg[7:0]rd_addr;
    reg[31:0]ready_count;
    wire write_context=ready_count[PIXEL_SHIFT];
    reg read_context_q;
    wire[255:0]spa_rd_ctx0,spa_rd_ctx1,spe_rd_ctx0,spe_rd_ctx1;
    wire[255:0]spa_rd_data=read_context_q?spa_rd_ctx1:spa_rd_ctx0;
    wire[255:0]spe_rd_data=read_context_q?spe_rd_ctx1:spe_rd_ctx0;
    generate if(1==0)begin:G_SHARED_INPUT_RAM
        mamba_input_bram_256x256 u_input_ctx0(
            .clka(clk),.ena(wr_en&&!write_context),
            .wea({wr_en&&!write_context}),.addra(input_q_pixel),
            .dina(spa_wr_data),.clkb(clk),.enb(rd_en&&!read_context_q),
            .addrb(rd_addr),.doutb(spa_rd_ctx0));
        mamba_input_bram_256x256 u_input_ctx1(
            .clka(clk),.ena(wr_en&&write_context),
            .wea({wr_en&&write_context}),.addra(input_q_pixel),
            .dina(spa_wr_data),.clkb(clk),.enb(rd_en&&read_context_q),
            .addrb(rd_addr),.doutb(spa_rd_ctx1));
        assign spe_rd_ctx0=spa_rd_ctx0;
        assign spe_rd_ctx1=spa_rd_ctx1;
    end else begin:G_SPLIT_INPUT_RAM
        mamba_input_bram_256x256 u_spa_input_ctx0(
            .clka(clk),.ena(wr_en&&!write_context),
            .wea({wr_en&&!write_context}),.addra(input_q_pixel),
            .dina(spa_wr_data),.clkb(clk),.enb(rd_en&&!read_context_q),
            .addrb(rd_addr),.doutb(spa_rd_ctx0));
        mamba_input_bram_256x256 u_spa_input_ctx1(
            .clka(clk),.ena(wr_en&&write_context),
            .wea({wr_en&&write_context}),.addra(input_q_pixel),
            .dina(spa_wr_data),.clkb(clk),.enb(rd_en&&read_context_q),
            .addrb(rd_addr),.doutb(spa_rd_ctx1));
        mamba_input_bram_256x256 u_spe_input_ctx0(
            .clka(clk),.ena(wr_en&&!write_context),
            .wea({wr_en&&!write_context}),.addra(input_q_pixel),
            .dina(spe_wr_data),.clkb(clk),.enb(rd_en&&!read_context_q),
            .addrb(rd_addr),.doutb(spe_rd_ctx0));
        mamba_input_bram_256x256 u_spe_input_ctx1(
            .clka(clk),.ena(wr_en&&write_context),
            .wea({wr_en&&write_context}),.addra(input_q_pixel),
            .dina(spe_wr_data),.clkb(clk),.enb(rd_en&&read_context_q),
            .addrb(rd_addr),.doutb(spe_rd_ctx1));
    end endgenerate
    // The collect payload is completely overwritten before wr_en.  Do not put
    // the two 256-bit registers on the global frame_start/reset control set.
    always@(posedge clk)begin
        if(!rst_n||frame_start)ready_count<=0;
        else if(input_q_valid&&input_q_channel==28)
            ready_count<=ready_count+1'b1;
        if(input_q_valid)begin
            spa_collect_pixel[input_q_channel*8+:32]<=spa_patch_q;
            spe_collect_pixel[input_q_channel*8+:32]<=spe_patch_q;
        end
    end

    localparam[2:0]ST_WAIT=0,ST_READ=1,ST_CAPTURE=2,ST_ISSUE=3;
    reg[2:0]state;reg[31:0]pixel_q;reg[5:0]step;reg[255:0]spa_activation,spe_activation;
    wire[7:0]pixel_local_addr=
        {{(8-PIXEL_SHIFT){1'b0}},pixel_q[PIXEL_SHIFT-1:0]};
    wire request=state==ST_ISSUE;
    // Reserve one downstream record slot at the scheduler boundary, before
    // the BRAM/read/dot-product pipeline creates any in-flight payload.
    assign record_launch=(state==ST_WAIT)&&(pixel_q<ready_count)
        &&record_credit_ready;
    always@(posedge clk)begin
        if(!rst_n)begin state<=ST_WAIT;pixel_q<=0;step<=0;rd_en<=0;rd_addr<=0;read_context_q<=0;end
        else if(frame_start)begin state<=ST_WAIT;pixel_q<=0;step<=0;rd_en<=0;rd_addr<=0;read_context_q<=0;end
        else begin
            // BRAM dout is stable between reads.  Free-running payload copies
            // avoid a 512-bit ST_CAPTURE clock-enable control set.
            spa_activation<=spa_rd_data;
            spe_activation<=spe_rd_data;
            rd_en<=0;case(state)
            ST_WAIT:begin step<=0;if(record_launch)begin rd_en<=1;rd_addr<=pixel_local_addr;read_context_q<=pixel_q[PIXEL_SHIFT];state<=ST_READ;end end
            ST_READ:state<=ST_CAPTURE;
            ST_CAPTURE:state<=ST_ISSUE;
            ST_ISSUE:if(step==ISSUE_STEPS-1)begin step<=0;pixel_q<=pixel_q+1'b1;state<=ST_WAIT;end else step<=step+1'b1;
            default:state<=ST_WAIT;
        endcase end
    end

    wire[5:0]spa_row_base=step*OUTPUT_LANES;
    wire[1:0]spe_token_issue=step/SPE_GROUPS;
    wire[3:0]spe_row_base=(step%SPE_GROUPS)*OUTPUT_LANES;
    wire[2047:0]spa_weights8;wire[511:0]spe_weights8;
    wire[511:0]spa_weights2;wire[127:0]spe_weights2;
    generate if(OUTPUT_LANES==8)begin:G_ROM8
        mamba_spa_in_weight_rom_8r#(.BLOCK_ID(BLOCK_ID))s(.clk(clk),.en(request),.base_addr(spa_row_base),.data(spa_weights8));
        mamba_spe_in_weight_rom_8r#(.BLOCK_ID(BLOCK_ID))p(.clk(clk),.en(request),.base_addr(spe_row_base),.data(spe_weights8));
        assign spa_weights2=512'd0;assign spe_weights2=128'd0;
    end else begin:G_ROM2
        mamba_spa_in_weight_rom_2r#(.BLOCK_ID(BLOCK_ID))s(.clk(clk),.en(request),.addr0(spa_row_base),.addr1(spa_row_base+1'b1),.data0(spa_weights2[255:0]),.data1(spa_weights2[511:256]));
        mamba_spe_in_weight_rom_2r#(.BLOCK_ID(BLOCK_ID))p(.clk(clk),.en(request),.addr0(spe_row_base),.addr1(spe_row_base+1'b1),.data0(spe_weights2[63:0]),.data1(spe_weights2[127:64]));
        assign spa_weights8=2048'd0;assign spe_weights8=512'd0;
    end endgenerate

    // The Block0 8-read adapters contain one pair-local address register.
    // Delay only that geometry by one clock; Block1/2 retain the original 2r
    // ROM latency.  Payload free-runs and is qualified solely by the valid.
    reg rom8_issue_valid;
    reg [13:0]rom8_issue_meta;
    reg [255:0]rom8_spa_act,rom8_spe_act;
    reg rom_valid;reg[13:0]rom_meta;reg[255:0]rom_spa_act,rom_spe_act;
    wire rom_source_valid=(OUTPUT_LANES==8)?rom8_issue_valid:request;
    wire[13:0]rom_source_meta=(OUTPUT_LANES==8)?rom8_issue_meta:
        {pixel_local_addr,step};
    wire[255:0]rom_source_spa_act=(OUTPUT_LANES==8)?rom8_spa_act:
        spa_activation;
    wire[255:0]rom_source_spe_act=(OUTPUT_LANES==8)?rom8_spe_act:
        spe_activation;
    wire[OUTPUT_LANES*256-1:0]rom_spa_w;
    wire[OUTPUT_LANES*64-1:0]rom_spe_w;
    generate if(OUTPUT_LANES==8)begin:G_ACTIVE_W8
        assign rom_spa_w=spa_weights8;assign rom_spe_w=spe_weights8;
    end else begin:G_ACTIVE_W2
        assign rom_spa_w=spa_weights2;assign rom_spe_w=spe_weights2;
    end endgenerate
    always@(posedge clk)begin
        if(!rst_n)begin rom8_issue_valid<=0;rom_valid<=0;end
        else if(frame_start)begin rom8_issue_valid<=0;rom_valid<=0;end
        else begin rom8_issue_valid<=request;rom_valid<=rom_source_valid;end
        // Metadata/activation are payload and are qualified only by rom_valid.
        rom8_issue_meta<={pixel_local_addr,step};
        rom8_spa_act<=spa_activation;
        rom8_spe_act<=spe_activation;
        rom_meta<=rom_source_meta;
        rom_spa_act<=rom_source_spa_act;
        rom_spe_act<=rom_source_spe_act;
    end

    wire signed[15:0]spa_product[0:OUTPUT_LANES*32-1];
    wire signed[15:0]spe_product[0:OUTPUT_LANES*8-1];genvar l,f;
    generate for(l=0;l<OUTPUT_LANES/2;l=l+1)begin:G_LANE_PAIR
        for(f=0;f<32;f=f+1)begin:G_SPA_M
            mamba_packed_signed_mult_2x8_3cyc m(.clk(clk),
                .activation(rom_spa_act[f*8+:8]),
                .weight_low(rom_spa_w[((l*2)*32+f)*8+:8]),
                .weight_high(rom_spa_w[((l*2+1)*32+f)*8+:8]),
                .product_low(spa_product[(l*2)*32+f]),
                .product_high(spa_product[(l*2+1)*32+f]));
        end
        for(f=0;f<8;f=f+1)begin:G_SPE_M
            wire[7:0]a=rom_spe_act[(rom_meta[5:0]/SPE_GROUPS*8+f)*8+:8];
            mamba_packed_signed_mult_2x8_3cyc m(.clk(clk),.activation(a),
                .weight_low(rom_spe_w[((l*2)*8+f)*8+:8]),
                .weight_high(rom_spe_w[((l*2+1)*8+f)*8+:8]),
                .product_low(spe_product[(l*2)*8+f]),
                .product_high(spe_product[(l*2+1)*8+f]));
        end
    end endgenerate

    reg mv0,mv1,mv2;reg[13:0]mm0,mm1,mm2;
    reg signed[16:0]spa_l1[0:OUTPUT_LANES*16-1];reg signed[17:0]spa_l2[0:OUTPUT_LANES*8-1];
    reg signed[18:0]spa_l3[0:OUTPUT_LANES*4-1];
    (* use_dsp="no" *)reg signed[19:0]spa_l4[0:OUTPUT_LANES*2-1];
    (* use_dsp="no" *)reg signed[20:0]spa_l5[0:OUTPUT_LANES-1];
    reg signed[16:0]spe_l1[0:OUTPUT_LANES*4-1];
    (* use_dsp="no" *)reg signed[17:0]spe_l2[0:OUTPUT_LANES*2-1];
    (* use_dsp="no" *)reg signed[18:0]spe_l3[0:OUTPUT_LANES-1];
    reg sv1,sv2,sv3,sv4,sv5,pv1,pv2,pv3;reg[13:0]sm1,sm2,sm3,sm4,sm5,pm1,pm2,pm3;integer i;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin mv0<=0;mv1<=0;mv2<=0;sv1<=0;sv2<=0;sv3<=0;sv4<=0;sv5<=0;pv1<=0;pv2<=0;pv3<=0;end
        else begin mv0<=rom_valid;mv1<=mv0;mv2<=mv1;
            sv1<=mv2;sv2<=sv1;sv3<=sv2;sv4<=sv3;sv5<=sv4;pv1<=mv2;pv2<=pv1;pv3<=pv2;end
        // Payload and metadata deliberately free-run.  Only the valid
        // pipelines are reset/qualified; keeping these assignments outside
        // the reset branch prevents frame_start from becoming a CE on every
        // reduction register in all three in_proj instances.
        mm0<=rom_meta;mm1<=mm0;mm2<=mm1;
        sm1<=mm2;sm2<=sm1;sm3<=sm2;sm4<=sm3;sm5<=sm4;
        pm1<=mm2;pm2<=pm1;pm3<=pm2;
        for(i=0;i<OUTPUT_LANES*16;i=i+1)spa_l1[i]<=$signed({spa_product[i*2][15],spa_product[i*2]})+$signed({spa_product[i*2+1][15],spa_product[i*2+1]});
        for(i=0;i<OUTPUT_LANES*4;i=i+1)spe_l1[i]<=$signed({spe_product[i*2][15],spe_product[i*2]})+$signed({spe_product[i*2+1][15],spe_product[i*2+1]});
        for(i=0;i<OUTPUT_LANES*8;i=i+1)spa_l2[i]<=$signed({spa_l1[i*2][16],spa_l1[i*2]})+$signed({spa_l1[i*2+1][16],spa_l1[i*2+1]});
        for(i=0;i<OUTPUT_LANES*4;i=i+1)spa_l3[i]<=$signed({spa_l2[i*2][17],spa_l2[i*2]})+$signed({spa_l2[i*2+1][17],spa_l2[i*2+1]});
        for(i=0;i<OUTPUT_LANES*2;i=i+1)spa_l4[i]<=$signed({spa_l3[i*2][18],spa_l3[i*2]})+$signed({spa_l3[i*2+1][18],spa_l3[i*2+1]});
        for(i=0;i<OUTPUT_LANES;i=i+1)spa_l5[i]<=$signed({spa_l4[i*2][19],spa_l4[i*2]})+$signed({spa_l4[i*2+1][19],spa_l4[i*2+1]});
        for(i=0;i<OUTPUT_LANES*2;i=i+1)spe_l2[i]<=$signed({spe_l1[i*2][16],spe_l1[i*2]})+$signed({spe_l1[i*2+1][16],spe_l1[i*2+1]});
        for(i=0;i<OUTPUT_LANES;i=i+1)spe_l3[i]<=$signed({spe_l2[i*2][17],spe_l2[i*2]})+$signed({spe_l2[i*2+1][17],spe_l2[i*2+1]});
    end

    wire signed[36:0]spa_rq[0:OUTPUT_LANES-1];wire signed[36:0]spe_rq[0:OUTPUT_LANES-1];
    generate for(l=0;l<OUTPUT_LANES;l=l+1)begin:G_RQ
        requant_mult_21x16 sr(.CLK(clk),.A(spa_l5[l]),.B(SPA_REQUANT_MULTIPLIER),.P(spa_rq[l]));
        requant_mult_21x16 pr(.CLK(clk),.A({{2{spe_l3[l][18]}},spe_l3[l]}),.B(SPE_REQUANT_MULTIPLIER),.P(spe_rq[l]));
    end endgenerate
    reg srv0,srv1,srv2,prv0,prv1,prv2;reg[13:0]srm0,srm1,srm2,prm0,prm1,prm2;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin srv0<=0;srv1<=0;srv2<=0;prv0<=0;prv1<=0;prv2<=0;end
        else begin srv0<=sv5;srv1<=srv0;srv2<=srv1;prv0<=pv3;prv1<=prv0;prv2<=prv1;end
        srm0<=sm5;srm1<=srm0;srm2<=srm1;
        prm0<=pm3;prm1<=prm0;prm2<=prm1;
    end
    function[7:0]q8;input signed[36:0]p;input signed[6:0]s;reg signed[63:0]v,m,r;begin
        v={{27{p[36]}},p};m=v<0?-v:v;if(s>0)r=(m+(64'sd1<<<(s-1)))>>>s;else if(s<0)r=m<<<(-s);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    wire[7:0]sq[0:OUTPUT_LANES-1];wire[7:0]pq[0:OUTPUT_LANES-1];
    generate for(l=0;l<OUTPUT_LANES;l=l+1)begin:G_Q assign sq[l]=q8(spa_rq[l],SPA_REQUANT_SHIFT);assign pq[l]=q8(spe_rq[l],SPE_REQUANT_SHIFT);end endgenerate
    reg[47:0]spa_pack,spe_pack;integer o;integer spa_base;integer spe_base;integer spe_tok;
    always@(posedge clk)begin
        if(!rst_n)begin spa_out_valid<=0;spe_out_valid<=0;spa_done<=0;spe_done<=0;spa_out_data<=0;spe_out_data<=0;spa_pack<=0;spe_pack<=0;
            spa_out_pixel_addr<=0;spa_out_channel_base<=0;spe_out_pixel_addr<=0;spe_out_token<=0;spe_out_channel_base<=0;end
        else if(frame_start)begin spa_out_valid<=0;spe_out_valid<=0;spa_done<=0;spe_done<=0;
            spa_out_pixel_addr<=0;spa_out_channel_base<=0;spe_out_pixel_addr<=0;spe_out_token<=0;spe_out_channel_base<=0;end
        else begin spa_out_valid<=0;spe_out_valid<=0;spa_done<=0;spe_done<=0;
            if(srv2)begin spa_base=srm2[5:0]*OUTPUT_LANES;
                if(OUTPUT_LANES==8)begin spa_out_valid<=1;spa_out_pixel_addr<=srm2[13:6];spa_out_channel_base<=spa_base[5:0];for(o=0;o<8;o=o+1)spa_out_data[o*8+:8]<=sq[o];
                    if(srm2[13:6]==PIXEL_COUNT-1&&spa_base==56)spa_done<=1;end
                else if(spa_base[2:0]==6)begin spa_out_valid<=1;spa_out_pixel_addr<=srm2[13:6];spa_out_channel_base<={spa_base[5:3],3'b0};spa_out_data<={sq[1],sq[0],spa_pack};
                    if(srm2[13:6]==PIXEL_COUNT-1&&spa_base==62)spa_done<=1;end
                else spa_pack[spa_base[2:0]*8+:16]<={sq[1],sq[0]};end
            if(prv2)begin spe_tok=prm2[5:0]/SPE_GROUPS;spe_base=(prm2[5:0]%SPE_GROUPS)*OUTPUT_LANES;
                if(OUTPUT_LANES==8)begin spe_out_valid<=1;spe_out_pixel_addr<=prm2[13:6];spe_out_token<=spe_tok[1:0];spe_out_channel_base<=spe_base[3:0];for(o=0;o<8;o=o+1)spe_out_data[o*8+:8]<=pq[o];
                    if(prm2[13:6]==PIXEL_COUNT-1&&spe_tok==3&&spe_base==8)spe_done<=1;end
                else if(spe_base[2:0]==6)begin spe_out_valid<=1;spe_out_pixel_addr<=prm2[13:6];spe_out_token<=spe_tok[1:0];spe_out_channel_base<={spe_base[3],3'b0};spe_out_data<={pq[1],pq[0],spe_pack};
                    if(prm2[13:6]==PIXEL_COUNT-1&&spe_tok==3&&spe_base==14)spe_done<=1;end
                else spe_pack[spe_base[2:0]*8+:16]<={pq[1],pq[0]};end
        end
    end
endmodule

`default_nettype wire
