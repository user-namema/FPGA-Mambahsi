`timescale 1ns/1ps
`default_nettype none

// Area-scaled in_proj for block1/block2.  Block1 uses 16 Spa + 4 Spe
// multipliers; block2 uses 4 Spa + 1 Spe multiplier.  Block2 Spa is widened
// independently so the late spatial branch no longer dominates the tail.
module both_in_proj_stream_small #(
    parameter integer BLOCK_ID=1,
    parameter integer PIXEL_COUNT=(BLOCK_ID==1)?64:16,
    // AvgPool is exported as a signed integer sum with scale divided by four.
    // The two QAT branches requantize that code before in_proj. Block1 Spe's
    // observer ratio is just below 1/4, so exact half cases round toward zero;
    // 32767/2^17 reproduces every exported Block1 first-tile code. The other
    // three branch contracts are exactly 1/4 with half-away rounding.
    parameter signed[15:0] SPA_INPUT_MULTIPLIER=16'sd16384,
    parameter signed[6:0] SPA_INPUT_SHIFT=7'sd16,
    parameter signed[15:0] SPE_INPUT_MULTIPLIER=(BLOCK_ID==1)?16'sd32767:16'sd16384,
    parameter signed[6:0] SPE_INPUT_SHIFT=(BLOCK_ID==1)?7'sd17:7'sd16,
    parameter signed[15:0] SPA_REQUANT_MULTIPLIER=16'sd24229,
    parameter signed[6:0] SPA_REQUANT_SHIFT=7'sd20,
    parameter signed[15:0] SPE_REQUANT_MULTIPLIER=16'sd23907,
    parameter signed[6:0] SPE_REQUANT_SHIFT=7'sd19
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire patch_valid,input wire[7:0]patch_pixel_addr,
    input wire[4:0]patch_channel_base,input wire[31:0]patch_data,
    input wire spa_weight_we,input wire[10:0]spa_weight_addr,input wire signed[7:0]spa_weight_data,
    input wire signed[31:0]spa_requant_multiplier,input wire signed[6:0]spa_requant_shift,
    input wire spe_weight_we,input wire[6:0]spe_weight_addr,input wire signed[7:0]spe_weight_data,
    input wire signed[31:0]spe_requant_multiplier,input wire signed[6:0]spe_requant_shift,
    output reg spa_out_valid,output reg[7:0]spa_out_pixel_addr,
    output reg[5:0]spa_out_channel_base,output reg[63:0]spa_out_data,output reg spa_done,
    output reg spe_out_valid,output reg[7:0]spe_out_pixel_addr,output reg[1:0]spe_out_token,
    output reg[3:0]spe_out_channel_base,output reg[63:0]spe_out_data,output reg spe_done
);
    localparam integer SPA_PAR=(BLOCK_ID==1)?16:4;
    localparam integer SPA_CHUNKS=32/SPA_PAR;
    localparam integer SPA_CHUNK_BITS=(BLOCK_ID==1)?1:3;
    localparam integer SPA_STEPS=64*SPA_CHUNKS;
    localparam integer SPE_PAR=(BLOCK_ID==1)?4:1;
    localparam integer SPE_CHUNKS=8/SPE_PAR;
    localparam integer SPE_CHUNK_BITS=(BLOCK_ID==1)?1:3;
    localparam integer SPE_STEPS=4*16*SPE_CHUNKS;
    wire unused_cfg=spa_weight_we^spa_weight_addr[0]^spa_weight_data[0]
        ^spa_requant_multiplier[0]^spa_requant_shift[0]^spe_weight_we
        ^spe_weight_addr[0]^spe_weight_data[0]^spe_requant_multiplier[0]^spe_requant_shift[0];

    wire[31:0]spa_patch_data_q,spe_patch_data_q;
    genvar input_lane;
    generate for(input_lane=0;input_lane<4;input_lane=input_lane+1)begin:G_INPUT_REQUANT
        assign spa_patch_data_q[input_lane*8+:8]=requant_input8(
            $signed(patch_data[input_lane*8+:8]),SPA_INPUT_MULTIPLIER,SPA_INPUT_SHIFT);
        assign spe_patch_data_q[input_lane*8+:8]=requant_input8(
            $signed(patch_data[input_lane*8+:8]),SPE_INPUT_MULTIPLIER,SPE_INPUT_SHIFT);
    end endgenerate
    reg[255:0]spa_collect,spe_collect;wire wr_en=patch_valid&&(patch_channel_base==5'd28);
    wire[255:0]spa_wr_data={spa_patch_data_q,spa_collect[223:0]};
    wire[255:0]spe_wr_data={spe_patch_data_q,spe_collect[223:0]};
    reg rd_en;reg[7:0]rd_addr;wire[255:0]spa_rd_data,spe_rd_data;
    mamba_input_bram_256x256 u_spa_ram(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(patch_pixel_addr),.dina(spa_wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(spa_rd_data));
    mamba_input_bram_256x256 u_spe_ram(.clka(clk),.ena(wr_en),.wea({wr_en}),.addra(patch_pixel_addr),.dina(spe_wr_data),
        .clkb(clk),.enb(rd_en),.addrb(rd_addr),.doutb(spe_rd_data));
    reg[8:0]written;always@(posedge clk)begin
        if(!rst_n||frame_start)begin spa_collect<=0;spe_collect<=0;written<=0;end
        else if(patch_valid)begin
            spa_collect[patch_channel_base*8+:32]<=spa_patch_data_q;
            spe_collect[patch_channel_base*8+:32]<=spe_patch_data_q;
            if(patch_channel_base==28)written<=written+1'b1;
        end end

    localparam[1:0]W=0,R=1,C=2,I=3;reg[1:0]state;reg[8:0]pixel;reg[10:0]step;
    reg[255:0]spa_activation,spe_activation;
    wire issue=(state==I);wire spe_issue=issue&&(step<SPE_STEPS);
    always@(posedge clk)begin
        if(!rst_n)begin state<=W;pixel<=0;step<=0;spa_activation<=0;spe_activation<=0;rd_en<=0;rd_addr<=0;end
        else if(frame_start)begin state<=W;pixel<=0;step<=0;rd_en<=0;rd_addr<=0;end
        else begin rd_en<=0;case(state)
            W:begin step<=0;if(pixel<written)begin rd_en<=1;rd_addr<=pixel[7:0];state<=R;end end
            R:state<=C;
            C:begin spa_activation<=spa_rd_data;spe_activation<=spe_rd_data;state<=I;end
            I:if(step==SPA_STEPS-1)begin step<=0;pixel<=pixel+1'b1;state<=W;end else step<=step+1'b1;
            default:state<=W;endcase end end

    wire[5:0]spa_row=step>>SPA_CHUNK_BITS;wire[3:0]spa_chunk=step[3:0]&((1<<SPA_CHUNK_BITS)-1);
    wire[9:0]spe_linear=step>>SPE_CHUNK_BITS;wire[3:0]spe_row=spe_linear[3:0];wire[1:0]spe_token=spe_linear[5:4];
    wire[2:0]spe_chunk=step[2:0]&((1<<SPE_CHUNK_BITS)-1);
    wire[255:0]spa_w0,spa_w1;wire[63:0]spe_w0,spe_w1;
    mamba_spa_in_weight_rom_2r #(.BLOCK_ID(BLOCK_ID))u_sw(.clk(clk),.en(issue),.addr0(spa_row),.addr1(6'd0),.data0(spa_w0),.data1(spa_w1));
    mamba_spe_in_weight_rom_2r #(.BLOCK_ID(BLOCK_ID))u_pw(.clk(clk),.en(spe_issue),.addr0(spe_row),.addr1(4'd0),.data0(spe_w0),.data1(spe_w1));
    reg rv,prv;reg[17:0]rm;reg[16:0]prm;reg[255:0]ra,pra;
    always@(posedge clk)begin if(!rst_n||frame_start)begin rv<=0;prv<=0;rm<=0;prm<=0;ra<=0;pra<=0;end else begin
        rv<=issue;prv<=spe_issue;if(issue)begin rm<={pixel[7:0],spa_row,spa_chunk};ra<=spa_activation;end
        if(spe_issue)begin prm<={pixel[7:0],spe_token,spe_row,spe_chunk};pra<=spe_activation;end end end

    function[7:0]requant_input8;
        input signed[7:0]value;
        input signed[15:0]multiplier;
        input signed[6:0]shift_value;
        reg signed[23:0]product;
        reg signed[24:0]magnitude,rounded,signed_rounded;
        begin
            product=value*multiplier;
            magnitude=(product<0)?-$signed({product[23],product}):$signed({1'b0,product});
            if(shift_value>0)
                rounded=(magnitude+(25'sd1<<<(shift_value-1)))>>>shift_value;
            else if(shift_value<0)
                rounded=magnitude<<<(-shift_value);
            else
                rounded=magnitude;
            signed_rounded=(product<0)?-rounded:rounded;
            if(signed_rounded>127)requant_input8=8'h7f;
            else if(signed_rounded< -128)requant_input8=8'h80;
            else requant_input8=signed_rounded[7:0];
        end
    endfunction
    wire sdv4,sdv16;wire signed[17:0]sds4;wire signed[19:0]sds16;wire[17:0]sdm4,sdm16;
    generate if(BLOCK_ID==1)begin:GSI16
        mamba_dot16_pipeline#(18)u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(rv),
            .activation(ra[(rm[3:0]*16)*8+:128]),.weight(spa_w0[(rm[3:0]*16)*8+:128]),.in_meta(rm),.out_valid(sdv16),.out_sum(sds16),.out_meta(sdm16));
        assign sdv4=0;assign sds4=0;assign sdm4=0;
    end else begin:GSI4
        mamba_dot4_pipeline#(18)u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(rv),
            .activation(ra[(rm[3:0]*4)*8+:32]),.weight(spa_w0[(rm[3:0]*4)*8+:32]),.in_meta(rm),.out_valid(sdv4),.out_sum(sds4),.out_meta(sdm4));
        assign sdv16=0;assign sds16=0;assign sdm16=0;
    end endgenerate
    wire sdv=sdv16|sdv4;wire[17:0]sdm=sdv16?sdm16:sdm4;wire signed[20:0]sdp=sdv16?{{1{sds16[19]}},sds16}:{{3{sds4[17]}},sds4};

    wire pdv1,pdv4;wire signed[15:0]pds1;wire signed[17:0]pds4;wire[16:0]pdm1,pdm4;
    generate if(BLOCK_ID==1)begin:GPI4
        mamba_dot4_pipeline#(17)u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(prv),
            .activation(pra[(prm[8:7]*8+prm[2:0]*4)*8+:32]),.weight(spe_w0[(prm[2:0]*4)*8+:32]),.in_meta(prm),.out_valid(pdv4),.out_sum(pds4),.out_meta(pdm4));
        assign pdv1=0;assign pds1=0;assign pdm1=0;
    end else begin:GPI1
        mamba_dot1_pipeline#(17)u(.clk(clk),.rst_n(rst_n),.frame_start(frame_start),.in_valid(prv),
            .activation(pra[(prm[8:7]*8+prm[2:0])*8+:8]),.weight(spe_w0[prm[2:0]*8+:8]),.in_meta(prm),.out_valid(pdv1),.out_sum(pds1),.out_meta(pdm1));
        assign pdv4=0;assign pds4=0;assign pdm4=0;
    end endgenerate
    wire pdv=pdv1|pdv4;wire[16:0]pdm=pdv4?pdm4:pdm1;wire signed[20:0]pdp=pdv4?{{3{pds4[17]}},pds4}:{{5{pds1[15]}},pds1};

    reg signed[20:0]sacc,pacc;wire slast=sdm[3:0]==SPA_CHUNKS-1;wire plast=pdm[2:0]==SPE_CHUNKS-1;
    wire signed[20:0]ssum=(sdm[3:0]==0)?sdp:sacc+sdp;wire signed[20:0]psum=(pdm[2:0]==0)?pdp:pacc+pdp;
    always@(posedge clk)begin if(!rst_n||frame_start)begin sacc<=0;pacc<=0;end else begin
        if(sdv)begin if(sdm[3:0]==0)sacc<=sdp;else sacc<=sacc+sdp;end
        if(pdv)begin if(pdm[2:0]==0)pacc<=pdp;else pacc<=pacc+pdp;end end end
    wire signed[36:0]srqp,prqp;requant_mult_21x16 srq(.CLK(clk),.A(ssum),.B(SPA_REQUANT_MULTIPLIER),.P(srqp));
    requant_mult_21x16 prq(.CLK(clk),.A(psum),.B(SPE_REQUANT_MULTIPLIER),.P(prqp));
    reg sv0,sv1,sv2,pv0,pv1,pv2;reg[13:0]sm0,sm1,sm2;reg[13:0]pm0,pm1,pm2;
    always@(posedge clk)begin if(!rst_n||frame_start)begin sv0<=0;sv1<=0;sv2<=0;pv0<=0;pv1<=0;pv2<=0;sm0<=0;sm1<=0;sm2<=0;pm0<=0;pm1<=0;pm2<=0;end else begin
        sv0<=sdv&&slast;sv1<=sv0;sv2<=sv1;pv0<=pdv&&plast;pv1<=pv0;pv2<=pv1;
        if(sdv&&slast)sm0<=sdm[17:4];if(sv0)sm1<=sm0;if(sv1)sm2<=sm1;
        if(pdv&&plast)pm0<=pdm[16:3];if(pv0)pm1<=pm0;if(pv1)pm2<=pm1;end end
    function[7:0]q8;input signed[36:0]p;input signed[6:0]sh;reg signed[63:0]v,m,r;begin v={{27{p[36]}},p};m=v<0?-v:v;
        if(sh>0)r=(m+(64'sd1<<<(sh-1)))>>>sh;else if(sh<0)r=m<<<(-sh);else r=m;if(v<0)r=-r;
        if(r>127)q8=8'h7f;else if(r< -128)q8=8'h80;else q8=r[7:0];end endfunction
    reg[55:0]spack,ppack;wire[7:0]sb=q8(srqp,SPA_REQUANT_SHIFT),pb=q8(prqp,SPE_REQUANT_SHIFT);
    always@(posedge clk)begin if(!rst_n||frame_start)begin spa_out_valid<=0;spe_out_valid<=0;spa_done<=0;spe_done<=0;spack<=0;ppack<=0;end else begin
        spa_out_valid<=0;spe_out_valid<=0;spa_done<=0;spe_done<=0;
        if(sv2)begin if(sm2[2:0]==7)begin spa_out_valid<=1;spa_out_pixel_addr<=sm2[13:6];spa_out_channel_base<={sm2[5:3],3'b0};spa_out_data<={sb,spack};if(sm2[13:6]==PIXEL_COUNT-1&&sm2[5:0]==63)spa_done<=1;end else spack[sm2[2:0]*8+:8]<=sb;end
        if(pv2)begin if(pm2[2:0]==7)begin spe_out_valid<=1;spe_out_pixel_addr<=pm2[13:6];spe_out_token<=pm2[5:4];spe_out_channel_base<={pm2[3],3'b0};spe_out_data<={pb,ppack};if(pm2[13:6]==PIXEL_COUNT-1&&pm2[5:4]==3&&pm2[3:0]==15)spe_done<=1;end else ppack[pm2[2:0]*8+:8]<=pb;end
    end end
endmodule

// Public wrapper retains the legacy port contract and selects the block0 or
// area-scaled small-block implementation at elaboration time.
module both_in_proj_stream #(
    parameter integer BLOCK_ID=0,parameter integer PIXEL_COUNT=256,
    parameter signed[15:0]SPA_REQUANT_MULTIPLIER=16'sd18177,parameter signed[6:0]SPA_REQUANT_SHIFT=7'sd20,
    parameter signed[15:0]SPE_REQUANT_MULTIPLIER=16'sd23177,parameter signed[6:0]SPE_REQUANT_SHIFT=7'sd19
)(input wire clk,input wire rst_n,input wire frame_start,input wire record_credit_ready,
input wire patch_valid,input wire[7:0]patch_pixel_addr,
input wire[4:0]patch_channel_base,input wire[((BLOCK_ID==0)?32:40)-1:0]patch_data,input wire spa_weight_we,input wire[10:0]spa_weight_addr,
input wire signed[7:0]spa_weight_data,input wire signed[31:0]spa_requant_multiplier,input wire signed[6:0]spa_requant_shift,
input wire spe_weight_we,input wire[6:0]spe_weight_addr,input wire signed[7:0]spe_weight_data,
input wire signed[31:0]spe_requant_multiplier,input wire signed[6:0]spe_requant_shift,
output wire record_launch,
output wire spa_out_valid,output wire[7:0]spa_out_pixel_addr,output wire[5:0]spa_out_channel_base,output wire[63:0]spa_out_data,output wire spa_done,
output wire spe_out_valid,output wire[7:0]spe_out_pixel_addr,output wire[1:0]spe_out_token,output wire[3:0]spe_out_channel_base,output wire[63:0]spe_out_data,output wire spe_done);
    generate begin:G_DSP_PARALLEL
        both_in_proj_parallel_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.OUTPUT_LANES((BLOCK_ID==0)?8:2),.SPA_REQUANT_MULTIPLIER(SPA_REQUANT_MULTIPLIER),.SPA_REQUANT_SHIFT(SPA_REQUANT_SHIFT),.SPE_REQUANT_MULTIPLIER(SPE_REQUANT_MULTIPLIER),.SPE_REQUANT_SHIFT(SPE_REQUANT_SHIFT))u(
            .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
            .record_credit_ready(record_credit_ready),.record_launch(record_launch),.patch_valid(patch_valid),
            .patch_pixel_addr(patch_pixel_addr),.patch_channel_base(patch_channel_base),.patch_data(patch_data),
            .spa_weight_we(spa_weight_we),.spa_weight_addr(spa_weight_addr),.spa_weight_data(spa_weight_data),
            .spa_requant_multiplier(spa_requant_multiplier),.spa_requant_shift(spa_requant_shift),
            .spe_weight_we(spe_weight_we),.spe_weight_addr(spe_weight_addr),.spe_weight_data(spe_weight_data),
            .spe_requant_multiplier(spe_requant_multiplier),.spe_requant_shift(spe_requant_shift),
            .spa_out_valid(spa_out_valid),.spa_out_pixel_addr(spa_out_pixel_addr),
            .spa_out_channel_base(spa_out_channel_base),.spa_out_data(spa_out_data),.spa_done(spa_done),
            .spe_out_valid(spe_out_valid),.spe_out_pixel_addr(spe_out_pixel_addr),.spe_out_token(spe_out_token),
            .spe_out_channel_base(spe_out_channel_base),.spe_out_data(spe_out_data),.spe_done(spe_done));
    end endgenerate
endmodule

`default_nettype wire
