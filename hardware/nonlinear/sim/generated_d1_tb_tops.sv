`timescale 1ns/1ps

module tb_d1_blk0_spa;
 nl_d1_tb #(.CORE("blk0_spa"), .BLOCK_ID(0), .IS_SPE(0), .CHANNELS(64)) u_tb();
endmodule

module tb_d1_blk0_spe;
 nl_d1_tb #(.CORE("blk0_spe"), .BLOCK_ID(0), .IS_SPE(1), .CHANNELS(16)) u_tb();
endmodule

module tb_d1_blk1_spa;
 nl_d1_tb #(.CORE("blk1_spa"), .BLOCK_ID(1), .IS_SPE(0), .CHANNELS(64)) u_tb();
endmodule

module tb_d1_blk1_spe;
 nl_d1_tb #(.CORE("blk1_spe"), .BLOCK_ID(1), .IS_SPE(1), .CHANNELS(16)) u_tb();
endmodule

module tb_d1_blk2_spa;
 nl_d1_tb #(.CORE("blk2_spa"), .BLOCK_ID(2), .IS_SPE(0), .CHANNELS(64)) u_tb();
endmodule

module tb_d1_blk2_spe;
 nl_d1_tb #(.CORE("blk2_spe"), .BLOCK_ID(2), .IS_SPE(1), .CHANNELS(16)) u_tb();
endmodule
