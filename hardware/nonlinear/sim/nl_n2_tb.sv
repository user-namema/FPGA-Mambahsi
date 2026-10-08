`timescale 1ns/1ps
module nl_n2_tb #(
    parameter CORE="blk0_spa",
    parameter [31:0] SDT_Q30=0,
    parameter [47:0] SBSU_Q40=0,
    parameter [767:0] DECAY_Q24=0,
    parameter integer K_FRAC=24,
    parameter [31:0] EXP_L=0,EXP_B=0,STATE_L=0,STATE_B=0,
    parameter signed [31:0] LOG_L=0,LOG_B=0
);
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1, in_valid=0;
    reg signed [7:0] q_dt=0;
    wire out_valid;
    wire [7:0] out_address;
    wire [418:0] out_coeff;
    nl_coeff_n2 #(.SDT_Q30(SDT_Q30),.SBSU_Q40(SBSU_Q40),.DECAY_Q24(DECAY_Q24),
        .K_FRAC(K_FRAC),.EXP_L(EXP_L),.EXP_B(EXP_B),.LOG_L(LOG_L),.LOG_B(LOG_B),
        .STATE_L(STATE_L),.STATE_B(STATE_B)) dut(.*);
    reg [418:0] expected [0:255];
    reg [7:0] tags [0:2047];
    integer accept_cycle [0:2047];
    integer cycle=0,issued=0,checked=0,last_out=-1,errors=0;
    integer output_file,marker,lat;
    initial begin
        $readmemh({CORE,"_n2.mem"},expected);
        output_file=$fopen({CORE,"_n2_coefficients.csv"},"w");
        if(!output_file) $fatal(1,"Cannot create N2 CSV");
        $fdisplay(output_file,"request,address,latency_registers,actual419,expected419");
    end
    always @(posedge clk) begin
        cycle=cycle+1;
        #1;
        if(rst) begin
            issued=0; checked=0; errors=0; last_out=-1;
            if(out_valid!==0) $fatal(1,"N2 reset did not flush valid");
        end else begin
            if(in_valid) begin
                if(issued>=2048) $fatal(1,"N2 scoreboard overflow");
                tags[issued]=q_dt^8'h80;
                accept_cycle[issued]=cycle;
                issued=issued+1;
            end
            if(out_valid!==0 && out_valid!==1) $fatal(1,"N2 unknown valid");
            if(out_valid) begin
                if(checked>=issued) $fatal(1,"N2 unsolicited output");
                lat=cycle-accept_cycle[checked]+1;
                if(out_address!==tags[checked] || out_coeff!==expected[tags[checked]] || lat!=13) begin
                    if(errors<12) $display("N2 FAIL idx=%0d address=%0d latency=%0d got=%h want=%h",
                        checked,out_address,lat,out_coeff,expected[tags[checked]]);
                    errors=errors+1;
                end
                if(checked>0 && checked<256 && cycle-last_out!=1) $fatal(1,"N2 II is not 1");
                $fdisplay(output_file,"%0d,%0d,%0d,%h,%h",checked,tags[checked],lat,out_coeff,expected[tags[checked]]);
                checked=checked+1; last_out=cycle;
            end
        end
    end
    task drive(input reg en,input reg[7:0] q);
        begin @(negedge clk); in_valid=en; q_dt=q; end
    endtask
    integer i;
    reg [31:0] rng=32'h12345678;
    initial begin
        repeat(4) @(negedge clk);
        rst=0;
        for(i=0;i<10;i=i+1) drive(1,i);
        @(negedge clk); rst=1; in_valid=0;
        repeat(3) @(negedge clk);
        rst=0;
        for(i=0;i<256;i=i+1) drive(1,i^8'h80);
        for(i=0;i<512;i=i+1) begin
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            drive(rng[3:0]!=0,rng[15:8]);
        end
        drive(0,0);
        repeat(80) @(negedge clk);
        if(errors || issued!=checked) $fatal(1,"N2 failed issued=%0d checked=%0d errors=%0d",issued,checked,errors);
        $fclose(output_file);
        marker=$fopen({CORE,"_pass.txt"},"w");
        if(!marker) $fatal(1,"Cannot write N2 PASS marker");
        $fdisplay(marker,"PASS N2 core=%s issued/checked=%0d/%0d latency_registers=13 II=1",CORE,issued,checked);
        $fclose(marker);
        $display("PASS N2 core=%s issued/checked=%0d/%0d all 256 addresses plus gaps/reset; latency_registers=13 II=1",CORE,issued,checked);
        $finish;
    end
    initial begin #200000; $fatal(1,"N2 timeout"); end
endmodule
