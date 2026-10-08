`timescale 1ns/1ps

// Instantiated by generated per-core test tops. No mathematical real functions
// are used to fake DUT behavior: all DUTs are the actual synthesizable RTL.
module nl_tb #(
    parameter CORE="blk0_spa",
    parameter [31:0] SDT_Q30=0,
    parameter [47:0] SBSU_Q40=0,
    parameter [767:0] DECAY_Q24=0,
    parameter integer K_FRAC=24
);
    reg clk=0;
    always #5 clk=~clk;
    reg rst=1,in_valid=0;
    reg signed [7:0] q_dt=0;
    wire [2:0] valid;
    wire [7:0] address [0:2];
    wire [418:0] value [0:2];
    reg [418:0] expected0 [0:255];
    reg [418:0] expected1 [0:255];
    reg [418:0] expected3 [0:255];
    integer cycle=0,issued=0,errors=0;
    integer checked [0:2];
    integer first_out [0:2];
    integer last_out [0:2];
    integer latency_min [0:2];
    integer latency_max [0:2];
    integer coeff_errors [0:2];
    integer max_a_error [0:2];
    integer max_k_error [0:2];
    integer accept_cycle [0:2047];
    reg [7:0] accept_address [0:2047];
    integer logfile;
    integer passfile;
    integer m,j,idx,lat,delta,required_latency;
    reg [418:0] want,refword;
    reg finishing=0;
    genvar g;
    generate for(g=0;g<3;g=g+1) begin: G_DUT
        localparam integer METHOD=(g==2)?3:g;
        nl_coeff #(.METHOD(METHOD),.SDT_Q30(SDT_Q30),.SBSU_Q40(SBSU_Q40),
            .DECAY_Q24(DECAY_Q24),.K_FRAC(K_FRAC),.ROM_FILE({CORE,"_n3.mem"})) dut(
            .clk(clk),.rst(rst),.in_valid(in_valid),.q_dt(q_dt),
            .out_valid(valid[g]),.out_address(address[g]),.out_coeff(value[g]));
    end endgenerate

    initial begin
        $readmemh({CORE,"_n0.mem"},expected0);
        $readmemh({CORE,"_n1.mem"},expected1);
        $readmemh({CORE,"_n3.mem"},expected3);
        logfile=$fopen({CORE,"_coefficients.csv"},"w");
        if(!logfile) $fatal(1,"Cannot open result CSV");
        $fdisplay(logfile,"method,request,address,accept_cycle,output_cycle,latency_registers,actual419,reference419");
    end

    always @(posedge clk) begin
        cycle=cycle+1;
        // Check after NBA. Driver changes at negedge only.
        #1;
        if(rst) begin
            issued=0;
            errors=0;
            for(m=0;m<3;m=m+1) begin
                checked[m]=0;
                first_out[m]=-1;
                last_out[m]=-1;
                latency_min[m]=100000;
                latency_max[m]=0;
                coeff_errors[m]=0;
                max_a_error[m]=0;
                max_k_error[m]=0;
            end
            if(valid!==3'b000) $fatal(1,"Reset failed to clear output valid");
        end else begin
            if(in_valid) begin
                if(issued>=2048) $fatal(1,"Scoreboard overflow");
                accept_cycle[issued]=cycle;
                accept_address[issued]=q_dt^8'h80;
                issued=issued+1;
            end
            for(m=0;m<3;m=m+1) begin
                if(valid[m]!==1'b0 && valid[m]!==1'b1) $fatal(1,"X on output valid");
                if(valid[m]) begin
                    idx=checked[m];
                    if(idx>=issued) $fatal(1,"Unsolicited output method=%0d",m);
                    if(address[m]!==accept_address[idx]) begin
                        $display("TAG FAIL method=%0d index=%0d",m,idx); errors=errors+1;
                    end
                    case(m)
                        0: begin want=expected0[accept_address[idx]]; required_latency=59; end
                        1: begin want=expected1[accept_address[idx]]; required_latency=12; end
                        2: begin want=expected3[accept_address[idx]]; required_latency=2; end
                    endcase
                    refword=expected3[accept_address[idx]];
                    lat=cycle-accept_cycle[idx]+1;
                    if(lat!=required_latency) begin
                        $display("LATENCY FAIL method=%0d got=%0d expected=%0d",m,lat,required_latency);
                        errors=errors+1;
                    end
                    if(value[m]!==want) begin
                        if(errors<12) $display("RTL FAIL N%0d address=%0d got=%h expected=%h",(m==2)?3:m,address[m],value[m],want);
                        errors=errors+1;
                    end
                    if(lat<latency_min[m]) latency_min[m]=lat;
                    if(lat>latency_max[m]) latency_max[m]=lat;
                    if(first_out[m]<0) first_out[m]=cycle;
                    // First 256 requests are consecutive: directly verify sustained II=1.
                    if(idx>0 && idx<256 && cycle-last_out[m]!=1) begin
                        $display("II FAIL method=%0d index=%0d",m,idx); errors=errors+1;
                    end
                    last_out[m]=cycle;
                    for(j=0;j<16;j=j+1) begin
                        delta=$signed({1'b0,value[m][j*25+:25]})-$signed({1'b0,refword[j*25+:25]});
                        if(delta<0) delta=-delta;
                        if(idx<256) begin
                            if(delta!=0) coeff_errors[m]=coeff_errors[m]+1;
                            if(delta>max_a_error[m]) max_a_error[m]=delta;
                        end
                    end
                    delta=$signed({1'b0,value[m][418:400]})-$signed({1'b0,refword[418:400]});
                    if(delta<0) delta=-delta;
                    if(idx<256 && delta>max_k_error[m]) max_k_error[m]=delta;
                    $fdisplay(logfile,"N%0d,%0d,%0d,%0d,%0d,%0d,%h,%h",(m==2)?3:m,idx,
                        accept_address[idx],accept_cycle[idx],cycle,lat,value[m],refword);
                    checked[m]=checked[m]+1;
                end
            end
        end
    end

    integer t;
    reg [31:0] rng=32'h12345678;
    task drive(input reg en,input reg[7:0] code);
        begin @(negedge clk); in_valid=en; q_dt=code; end
    endtask
    initial begin
        repeat(4) @(negedge clk);
        rst=0;
        // In-flight reset flush test, discarded before the measured epoch.
        for(t=0;t<10;t=t+1) drive(1,t);
        @(negedge clk); rst=1; in_valid=0;
        repeat(3) @(negedge clk);
        rst=0;
        for(t=0;t<256;t=t+1) drive(1,t^8'h80);
        for(t=0;t<512;t=t+1) begin
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            drive(rng[3:0]!=0,rng[15:8]);
        end
        drive(0,0);
        repeat(80) @(negedge clk);
        for(t=0;t<3;t=t+1) begin
            if(checked[t]!=issued) errors=errors+1;
            $display("RESULT core=%s N%0d issued/checked=%0d/%0d latency_registers=%0d..%0d first/last=%0d/%0d II=1 A_max_LSB=%0d K_max_LSB=%0d A_mismatch=%0d/4096",
                CORE,(t==2)?3:t,issued,checked[t],latency_min[t],latency_max[t],first_out[t],last_out[t],max_a_error[t],max_k_error[t],coeff_errors[t]);
        end
        $fclose(logfile);
        if(errors) $fatal(1,"FAIL RTL/protocol/latency errors=%0d",errors);
        passfile=$fopen({CORE,"_pass.txt"},"w");
        if(!passfile) $fatal(1,"Cannot create completion marker");
        $fdisplay(passfile,"PASS core=%s checked=%0d N0_latency=%0d N1_latency=%0d N3_latency=%0d II=1",
            CORE,issued,latency_min[0],latency_min[1],latency_min[2]);
        $fclose(passfile);
        $display("PASS: %s N0/N1/N3 RTL matches per-method independent integer model; N3 matches v5; II=1. N0/N1 approximation error is reported, NOT called bit-exact to v5.",CORE);
        $finish;
    end
    initial begin #200000; $fatal(1,"Nonlinear comparison timeout"); end
endmodule
