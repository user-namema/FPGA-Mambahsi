`timescale 1ns/1ps
module nf_coeff_lanes #(
    parameter integer METHOD=0, BLOCK_ID=0, IS_SPE=0, LANES=1
) (
    input wire clk, reset, in_valid,
    input wire [LANES*8-1:0] q_dt,
    output wire out_valid,
    output wire [418:0] data0, data1, data2, data3
);
    wire [418:0] result [0:3];
    wire [3:0] valid;
    assign data0=result[0]; assign data1=result[1];
    assign data2=result[2]; assign data3=result[3];
    assign out_valid=valid[0];
    genvar lane;
    generate for(lane=0;lane<4;lane=lane+1) begin : G_LANE
      if(lane<LANES) begin : G_USED

        if(BLOCK_ID==0 && IS_SPE==0 && METHOD==0) begin : G_B0_spa_N0
          nl_n0_blk0_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==0 && IS_SPE==0 && METHOD==1) begin : G_B0_spa_N1
          nl_n1_blk0_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==0 && IS_SPE==0 && METHOD==2) begin : G_B0_spa_N2
          nl_n2_blk0_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==0 && IS_SPE==1 && METHOD==0) begin : G_B0_spe_N0
          nl_n0_blk0_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==0 && IS_SPE==1 && METHOD==1) begin : G_B0_spe_N1
          nl_n1_blk0_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==0 && IS_SPE==1 && METHOD==2) begin : G_B0_spe_N2
          nl_n2_blk0_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==0 && METHOD==0) begin : G_B1_spa_N0
          nl_n0_blk1_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==0 && METHOD==1) begin : G_B1_spa_N1
          nl_n1_blk1_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==0 && METHOD==2) begin : G_B1_spa_N2
          nl_n2_blk1_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==1 && METHOD==0) begin : G_B1_spe_N0
          nl_n0_blk1_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==1 && METHOD==1) begin : G_B1_spe_N1
          nl_n1_blk1_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==1 && IS_SPE==1 && METHOD==2) begin : G_B1_spe_N2
          nl_n2_blk1_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==0 && METHOD==0) begin : G_B2_spa_N0
          nl_n0_blk2_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==0 && METHOD==1) begin : G_B2_spa_N1
          nl_n1_blk2_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==0 && METHOD==2) begin : G_B2_spa_N2
          nl_n2_blk2_spa u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==1 && METHOD==0) begin : G_B2_spe_N0
          nl_n0_blk2_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==1 && METHOD==1) begin : G_B2_spe_N1
          nl_n1_blk2_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
        if(BLOCK_ID==2 && IS_SPE==1 && METHOD==2) begin : G_B2_spe_N2
          nl_n2_blk2_spe u_coeff (
            .clk(clk), .rst(reset), .in_valid(in_valid),
            .q_dt(q_dt[lane*8+:8]), .out_valid(valid[lane]),
            .out_address(), .out_coeff(result[lane]));
        end
      end else begin : G_UNUSED
        assign result[lane]=0; assign valid[lane]=0;
      end
    end endgenerate
endmodule
