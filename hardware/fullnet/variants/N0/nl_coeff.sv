`timescale 1ns/1ps

// METHOD 0: online polynomial; 1: FastMamba-formula adaptation; 3: joint ROM.
// All three accept one vector request/cycle (II=1), return identical 419-bit layout.
module nl_coeff #(
    parameter integer METHOD=3,
    parameter [31:0] SDT_Q30=32'd62709211,
    parameter [47:0] SBSU_Q40=48'd0,
    parameter [767:0] DECAY_Q24=768'd0,
    parameter integer K_FRAC=24,
    parameter ROM_FILE="blk0_spa_n3.mem"
) (
    input wire clk,rst,in_valid,
    input wire signed [7:0] q_dt,
    output wire out_valid,
    output wire [7:0] out_address,
    output wire [418:0] out_coeff
);
    localparam integer EL=(METHOD==0)?13:4;
    localparam integer SL=(METHOD==0)?43:5;
    localparam integer LATENCY=(METHOD==3)?2:(1+SL+1+EL+1);
    wire [7:0] address=q_dt ^ 8'h80;
    nl_delay #(.WIDTH(8),.DEPTH(LATENCY)) u_tag(
        .clk(clk),.rst(rst),.din(address),.dout(out_address));

    generate if(METHOD==3) begin: G_ROM
        (* rom_style="block" *) reg [418:0] rom [0:255];
        reg [418:0] data_q,out_q;
        reg v1,v2;
        initial $readmemh(ROM_FILE,rom);
        always @(posedge clk) begin
            data_q<=rom[address];
            out_q<=data_q;
            if(rst) begin v1<=0; v2<=0; end
            else begin v1<=in_valid; v2<=v1; end
        end
        assign out_valid=v2;
        assign out_coeff=out_q;
    end else begin: G_ONLINE
        wire [8:0] magnitude=q_dt[7] ? -$signed({q_dt[7],q_dt}) : {1'b0,q_dt};
        wire [40:0] xproduct=magnitude*SDT_Q30;
        reg [47:0] x24;
        reg positive,xv;
        always @(posedge clk) begin
            x24<=({7'd0,xproduct}+48'd32)>>6;
            positive<=!q_dt[7] && q_dt!=0;
            if(rst) xv<=0; else xv<=in_valid;
        end
        wire sv;
        wire [47:0] delta;
        nl_softplus #(.METHOD(METHOD)) u_softplus(
            .clk(clk),.rst(rst),.in_valid(xv),.positive(positive),.x24(x24),
            .out_valid(sv),.delta24(delta));
        reg av;
        always @(posedge clk) begin
            if(rst) av<=0; else av<=sv;
        end
        wire [15:0] ev;
        wire [511:0] exp30;
        genvar j;
        for(j=0;j<16;j=j+1) begin: G_STATE
            wire [47:0] decay=DECAY_Q24[j*48+:48];
            wire [95:0] ap=delta*decay;
            reg [47:0] mag24;
            always @(posedge clk) mag24<=(ap+96'd8388608)>>24;
            nl_exp_neg #(.METHOD(METHOD)) u_exp(
                .clk(clk),.rst(rst),.in_valid(av),.x24(mag24),
                .out_valid(ev[j]),.y30(exp30[j*32+:32]));
        end
        wire [95:0] kp=delta*SBSU_Q40;
        localparam [95:0] K_HALF=96'd1<<(63-K_FRAC);
        wire [95:0] kr=(kp+K_HALF)>>(64-K_FRAC);
        reg [18:0] kq;
        always @(posedge clk) kq <= kr>96'd524287 ? 19'h7ffff : kr[18:0];
        wire [18:0] ka;
        nl_delay #(.WIDTH(19),.DEPTH(EL)) u_k(
            .clk(clk),.rst(rst),.din(kq),.dout(ka));
        reg [418:0] oq;
        reg ov;
        integer state_index;
        always @(posedge clk) begin
            oq[418:400]<=ka;
            for(state_index=0;state_index<16;state_index=state_index+1)
                oq[state_index*25+:25]<=({1'b0,exp30[state_index*32+:32]}+33'd32)>>6;
            if(rst) ov<=0; else ov<=ev[0];
        end
        assign out_valid=ov;
        assign out_coeff=oq;
    end endgenerate
endmodule
