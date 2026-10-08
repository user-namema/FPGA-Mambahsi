`timescale 1ns/1ps
module tb_nf #(
    parameter integer TILE_COUNT=16,
    parameter real PERIOD_NS=10.0
);
    localparam integer II=4096, FIRST_START=1024;
    localparam integer LIMIT=FIRST_START+(TILE_COUNT-1)*II+16000;
    reg clk=0;
    always #(PERIOD_NS/2.0) clk=~clk;
    reg rst_n=0, tile_wr_en=0, start=0;
    reg [7:0] tile_wr_addr=0;
    reg [127:0] tile_wr_data=0;
    wire valid, sof, eof, context_bit, feature_done, done, busy;
    wire [3:0] pixel, out_class;
    wire [71:0] logits;
    nl_fullnet_shell dut (
        .clk(clk), .rst_n(rst_n), .tile_wr_en(tile_wr_en),
        .tile_wr_addr(tile_wr_addr), .tile_wr_data(tile_wr_data), .start(start),
        .out_valid(valid), .out_sof(sof), .out_eof(eof),
        .out_pixel_addr(pixel), .out_logits(logits), .out_class(out_class),
        .out_tile_context(context_bit), .feature_done(feature_done),
        .done(done), .patch_busy(busy)
    );
    reg [127:0] tiles [0:TILE_COUNT*256-1];
    reg [7:0] gold [0:TILE_COUNT*144-1];
    integer cycle=0, starts=0, beats=0, errors=0, bytes_checked=0;
    integer last_start=-1, last_sof=-1, out_min_ii=2147483647, out_max_ii=0;
    integer first_result=-1, last_result=-1, drain_done=-1;
    integer fd, csv, c, i, t, phase, interval;
    initial begin
        $readmemh("tiles.mem",tiles);
        $readmemh("gold.mem",gold);
        csv=$fopen("actual_logits.csv","w");
        $fdisplay(csv,"tile,pixel,class,actual,expected");
        if(TILE_COUNT<1) $fatal(1,"Invalid tile count");
    end
    // Same *absolute* cycle schedule for every method, with no hidden wait
    // dependent on method latency. Patch loads do not overlap patch reads.
    always @(negedge clk) begin
        rst_n = cycle>=200;
        start=0; tile_wr_en=0;
        for(t=0;t<TILE_COUNT;t=t+1) begin
            phase=cycle-(FIRST_START+t*II-512);
            if(phase>=0 && phase<256) begin
                if(busy===1'b1) $fatal(1,"Patch busy during common preload cycle=%0d",cycle);
                tile_wr_en=1;
                tile_wr_addr=phase;
                tile_wr_data=tiles[t*256+phase];
            end
            if(cycle==FIRST_START+t*II) start=1;
        end
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        if(rst_n && start) begin
            if(last_start>=0 && cycle-last_start!=II) $fatal(1,"Input II");
            last_start=cycle;
            starts=starts+1;
        end
        #0.001;
        if(rst_n && valid) begin
            if(beats>=TILE_COUNT*16 || pixel!==(beats%16) ||
               sof!==(pixel==0) || eof!==(pixel==15) ||
               context_bit!==((beats/16)%2))
                $fatal(1,"Output order/SOF/EOF/context at beat %0d",beats);
            if(sof) begin
                if(last_sof>=0) begin
                    interval=cycle-last_sof;
                    if(interval<out_min_ii) out_min_ii=interval;
                    if(interval>out_max_ii) out_max_ii=interval;
                    if(interval!=II) $fatal(1,"Output SOF II=%0d, expected4096",interval);
                end
                last_sof=cycle;
                $display("NF SOF tile=%0d cycle=%0d",beats/16,cycle);
            end
            if(first_result<0) first_result=cycle;
            last_result=cycle;
            for(c=0;c<9;c=c+1) begin
                $fdisplay(csv,"%0d,%0d,%0d,%02h,%02h",beats/16,pixel,c,
                          logits[c*8+:8],gold[beats*9+c]);
                if(logits[c*8+:8]!==gold[beats*9+c]) begin
                    if(errors<20) $display("NF ERROR tile=%0d pixel=%0d class=%0d got=%0d expected=%0d",
                        beats/16,pixel,c,$signed(logits[c*8+:8]),$signed(gold[beats*9+c]));
                    errors=errors+1;
                end
                bytes_checked=bytes_checked+1;
            end
            beats=beats+1;
            if(beats==TILE_COUNT*16) drain_done=cycle;
        end
        // Guard against duplicate late outputs before declaring success.
        if(drain_done>=0 && cycle==drain_done+512) begin
            $fclose(csv);
            if(errors || starts!=TILE_COUNT || bytes_checked!=TILE_COUNT*144)
                $fatal(1,"NF failed errors=%0d starts=%0d checked=%0d",errors,starts,bytes_checked);
            fd=$fopen("PASS.txt","w");
            $fdisplay(fd,"PASS independent method-specific software logits; tiles=%0d bytes=%0d II=4096",TILE_COUNT,bytes_checked);
            $fdisplay(fd,"first_output_cycle=%0d last_output_cycle=%0d first_start_cycle=%0d",first_result,last_result,FIRST_START+1);
            $fdisplay(fd,"output_min_ii=%0d output_max_ii=%0d",out_min_ii,out_max_ii);
            $fclose(fd);
            $display("PASS NF full D1 network; method=%0d tiles=%0d bytes=%0d bits=%0d II=4096",
                `NF_METHOD,TILE_COUNT,bytes_checked,bytes_checked*8);
            $finish;
        end
        if(cycle%5000==0) $display("NF PROGRESS cycle=%0d starts=%0d beats=%0d errors=%0d",cycle,starts,beats,errors);
        if(cycle>LIMIT) $fatal(1,"NF TIMEOUT starts=%0d outputs=%0d",starts,beats);
    end
endmodule
