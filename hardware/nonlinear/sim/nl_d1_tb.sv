`timescale 1ns/1ps
module nl_d1_tb #(
    parameter CORE = "blk0_spa",
    parameter integer BLOCK_ID = 0,
    parameter integer IS_SPE = 0,
    parameter integer CHANNELS = 64
);
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst = 1, read_valid = 0;
    reg [5:0] channel = 0;
    reg [7:0] u_s8 = 0;
    reg signed [47:0] c_readout = 0;
    wire out_valid;
    wire signed [47:0] total_readout;
    nl_d1_readout #(.BLOCK_ID(BLOCK_ID), .IS_SPE(IS_SPE)) dut (.*);

    reg signed [47:0] d_golden [0:CHANNELS*256-1];
    reg signed [47:0] expected [0:65535];
    integer issued_cycle [0:65535];
    integer cycle = 0, issued = 0, checked = 0, last = -1;
    integer marker, ch, raw, j;
    reg [31:0] rng = 32'h735a0241;
    initial $readmemh({CORE,"_d_golden.mem"}, d_golden);

    always @(posedge clk) begin
        cycle = cycle + 1;
        #1;
        if (rst) begin
            issued = 0;
            checked = 0;
            last = -1;
            if (out_valid !== 0) $fatal(1,"D valid survived reset");
        end else begin
            if (read_valid) begin
                if (issued >= 65536 || channel >= CHANNELS) $fatal(1,"D scoreboard range");
                expected[issued] = $signed(c_readout) + $signed(d_golden[channel*256+u_s8]);
                issued_cycle[issued] = cycle;
                issued = issued + 1;
            end
            if (out_valid !== 0 && out_valid !== 1) $fatal(1,"D unknown valid");
            if (out_valid) begin
                if (checked >= issued) $fatal(1,"Unsolicited D output");
                if (cycle-issued_cycle[checked]+1 != 18)
                    $fatal(1,"D C+DU latency mismatch: %0d",cycle-issued_cycle[checked]+1);
                if (total_readout !== expected[checked])
                    $fatal(1,"D mismatch index=%0d got=%0d expected=%0d", checked,total_readout,expected[checked]);
                if (checked > 0 && checked < CHANNELS*256 && cycle-last != 1)
                    $fatal(1,"D II != 1");
                checked = checked+1;
                last = cycle;
            end
        end
    end

    task drive(input reg en, input integer c, input integer u, input reg signed [47:0] value);
        begin
            @(negedge clk);
            read_valid = en;
            channel = c;
            u_s8 = u;
            c_readout = value;
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        rst = 0;
        // Deliberately flush requests while their payload is still in flight.
        for (j=0; j<10; j=j+1) drive(1,j%CHANNELS,j,-48'sd1);
        @(negedge clk); rst=1; read_valid=0;
        repeat (3) @(negedge clk);
        rst=0;
        for (ch=0; ch<CHANNELS; ch=ch+1)
            for (raw=0; raw<256; raw=raw+1)
                // Legal 16-state C sum bounds, signed extremes included.
                drive(1,ch,raw,raw[0] ? 48'sd4398046511104 : -48'sd4398046509056);
        for (j=0; j<512; j=j+1) begin
            rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};
            drive(rng[3:0]!=0,rng[15:8]%CHANNELS,rng[23:16],$signed(rng));
        end
        drive(0,0,0,0);
        repeat (30) @(negedge clk);
        if (issued != checked || checked < CHANNELS*256) $fatal(1,"D output count failed");
        marker=$fopen({CORE,"_d1_pass.txt"},"w");
        if (!marker) $fatal(1,"Cannot write D PASS marker");
        $fdisplay(marker,"PASS D1 %s issued/checked=%0d/%0d; D LUT latency=17, C+D adapter=18, II=1",CORE,issued,checked);
        $fclose(marker);
        $display("PASS D1 %s all channels x 256 U codes, signed C limits, gaps/reset; %0d outputs",CORE,checked);
        $finish;
    end
    initial begin #1000000; $fatal(1,"D1 timeout"); end
endmodule
