`timescale 1ns / 1ps
`default_nettype none

// Vendor-ROM adapters.  Each selected branch instantiates only the IP that
// belongs to BLOCK_ID, so one reusable RTL block can be used for block0/1/2
// without placing all three parameter sets in the synthesized hierarchy.

module mamba_spa_in_weight_rom_2r #(
    parameter integer BLOCK_ID = 0
) (
    input  wire         clk,
    input  wire         en,
    input  wire [5:0]   addr0,
    input  wire [5:0]   addr1,
    output wire [255:0] data0,
    output wire [255:0] data1
);
    generate
        if (BLOCK_ID == 0) begin : GEN_BLOCK0
            blk0_spa_in_weight_rom_64x256 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end else if (BLOCK_ID == 1) begin : GEN_BLOCK1
            blk1_spa_in_weight_rom_64x256 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end else begin : GEN_BLOCK2
            blk2_spa_in_weight_rom_64x256 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end
    endgenerate
endmodule

module mamba_spe_in_weight_rom_2r #(
    parameter integer BLOCK_ID = 0
) (
    input  wire        clk,
    input  wire        en,
    input  wire [3:0]  addr0,
    input  wire [3:0]  addr1,
    output wire [63:0] data0,
    output wire [63:0] data1
);
    generate
        if (BLOCK_ID == 0) begin : GEN_BLOCK0
            blk0_spe_in_weight_rom_16x64 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end else if (BLOCK_ID == 1) begin : GEN_BLOCK1
            blk1_spe_in_weight_rom_16x64 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end else begin : GEN_BLOCK2
            blk2_spe_in_weight_rom_16x64 u_rom (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1)
            );
        end
    endgenerate
endmodule

// Eight read ports are built from four small true-dual-port ROM copies.  They
// are used only by Block0's high-throughput in_proj; Block1/2 select the 2r
// adapter and therefore do not instantiate these copies.
module mamba_spa_in_weight_rom_8r #(
    parameter integer BLOCK_ID = 0
)(input wire clk,input wire en,input wire[5:0]base_addr,
  output wire[2047:0]data);
    // Register each physical TDP-ROM pair at the adapter boundary.  A single
    // step/base_addr net previously crossed the complete Block0 in_proj
    // region and directly drove every RAMB36 address pin.  These preserved
    // pair-local endpoints keep each address net inside its owning ROM pair.
    // The ROM ports are intentionally always enabled; the caller delays its
    // valid/metadata by this extra address-register cycle.
    wire unused_en = en;
    genvar p;
    generate for(p=0;p<4;p=p+1)begin:G
        (* keep = "true", max_fanout = 4 *) reg [5:0] addr_even_q;
        (* keep = "true", max_fanout = 4 *) reg [5:0] addr_odd_q;
        always @(posedge clk) begin
            addr_even_q <= base_addr + p*2;
            addr_odd_q  <= base_addr + p*2 + 1;
        end
        mamba_spa_in_weight_rom_2r#(.BLOCK_ID(BLOCK_ID))u(
            .clk(clk),.en(1'b1),.addr0(addr_even_q),.addr1(addr_odd_q),
            .data0(data[(p*2)*256+:256]),.data1(data[(p*2+1)*256+:256]));
    end endgenerate
endmodule

module mamba_spe_in_weight_rom_8r #(
    parameter integer BLOCK_ID = 0
)(input wire clk,input wire en,input wire[3:0]base_addr,
  output wire[511:0]data);
    wire unused_en = en;
    genvar p;
    generate for(p=0;p<4;p=p+1)begin:G
        (* keep = "true", max_fanout = 4 *) reg [3:0] addr_even_q;
        (* keep = "true", max_fanout = 4 *) reg [3:0] addr_odd_q;
        always @(posedge clk) begin
            addr_even_q <= base_addr + p*2;
            addr_odd_q  <= base_addr + p*2 + 1;
        end
        mamba_spe_in_weight_rom_2r#(.BLOCK_ID(BLOCK_ID))u(
            .clk(clk),.en(1'b1),.addr0(addr_even_q),.addr1(addr_odd_q),
            .data0(data[(p*2)*64+:64]),.data1(data[(p*2+1)*64+:64]));
    end endgenerate
endmodule

module mamba_spa_x_weight_rom_1r #(
    parameter integer BLOCK_ID = 0
) (
    input wire clk,
    input wire en,
    input wire [5:0] addr,
    output wire [511:0] data
);
    generate
        if (BLOCK_ID == 0) begin : GEN_BLOCK0
            blk0_spa_x_weight_rom_64x512 u_rom (
                .clka(clk), .ena(en), .addra(addr), .douta(data));
        end else if (BLOCK_ID == 1) begin : GEN_BLOCK1
            blk1_spa_x_weight_rom_64x512 u_rom (
                .clka(clk), .ena(en), .addra(addr), .douta(data));
        end else begin : GEN_BLOCK2
            blk2_spa_x_weight_rom_64x512 u_rom (
                .clka(clk), .ena(en), .addra(addr), .douta(data));
        end
    endgenerate
endmodule

// Four independent read addresses for the high-throughput Spa x_proj.
// The source ROM is only 64x512, so four identical ROM instances are a
// modest BRAM cost and remove the 34-cycle/output-row bottleneck without
// introducing a combinational multi-port mux.
module mamba_spa_x_weight_rom_4r #(
    parameter integer BLOCK_ID = 0
) (
    input wire clk,
    input wire en,
    input wire [5:0] addr0,
    input wire [5:0] addr1,
    input wire [5:0] addr2,
    input wire [5:0] addr3,
    output wire [511:0] data0,
    output wire [511:0] data1,
    output wire [511:0] data2,
    output wire [511:0] data3
);
    mamba_spa_x_weight_rom_1r #(.BLOCK_ID(BLOCK_ID)) u0(
        .clk(clk),.en(en),.addr(addr0),.data(data0));
    mamba_spa_x_weight_rom_1r #(.BLOCK_ID(BLOCK_ID)) u1(
        .clk(clk),.en(en),.addr(addr1),.data(data1));
    mamba_spa_x_weight_rom_1r #(.BLOCK_ID(BLOCK_ID)) u2(
        .clk(clk),.en(en),.addr(addr2),.data(data2));
    mamba_spa_x_weight_rom_1r #(.BLOCK_ID(BLOCK_ID)) u3(
        .clk(clk),.en(en),.addr(addr3),.data(data3));
endmodule

module mamba_spe_x_weight_rom_4r #(
    parameter integer BLOCK_ID = 0
) (
    input wire clk,
    input wire en,
    input wire [5:0] addr0,
    input wire [5:0] addr1,
    input wire [5:0] addr2,
    input wire [5:0] addr3,
    output wire [127:0] data0,
    output wire [127:0] data1,
    output wire [127:0] data2,
    output wire [127:0] data3
);
    generate
        if (BLOCK_ID == 0) begin : GEN_BLOCK0
            blk0_spe_x_weight_rom_64x128_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk0_spe_x_weight_rom_64x128_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else if (BLOCK_ID == 1) begin : GEN_BLOCK1
            blk1_spe_x_weight_rom_64x128_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk1_spe_x_weight_rom_64x128_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else begin : GEN_BLOCK2
            blk2_spe_x_weight_rom_64x128_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk2_spe_x_weight_rom_64x128_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end
    endgenerate
endmodule

// Small blocks consume one output row at a time.  Only one physical TDP ROM
// copy is required; port B is deliberately disabled so it cannot create a
// second read path or force ROM replication.
module mamba_spe_x_weight_rom_1r #(
    parameter integer BLOCK_ID = 1
) (
    input wire clk,
    input wire en,
    input wire [5:0] addr,
    output wire [127:0] data
);
    wire [127:0] unused_data;
    generate
        if (BLOCK_ID == 1) begin : GEN_BLOCK1
            blk1_spe_x_weight_rom_64x128_copy01 u_rom (
                .clka(clk), .ena(en), .addra(addr), .douta(data),
                .clkb(clk), .enb(1'b0), .addrb(6'd0), .doutb(unused_data));
        end else begin : GEN_BLOCK2
            blk2_spe_x_weight_rom_64x128_copy01 u_rom (
                .clka(clk), .ena(en), .addra(addr), .douta(data),
                .clkb(clk), .enb(1'b0), .addrb(6'd0), .doutb(unused_data));
        end
    endgenerate
endmodule

module mamba_ssm_lut_rom_4r #(
    parameter integer BLOCK_ID = 0,
    parameter integer IS_SPE = 0
) (
    input wire clk,
    input wire en,
    input wire [7:0] addr0,
    input wire [7:0] addr1,
    input wire [7:0] addr2,
    input wire [7:0] addr3,
    output wire [418:0] data0,
    output wire [418:0] data1,
    output wire [418:0] data2,
    output wire [418:0] data3
);
    generate
        if ((BLOCK_ID == 0) && (IS_SPE == 0)) begin : GEN_B0_SPA
            blk0_spa_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk0_spa_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else if ((BLOCK_ID == 0) && (IS_SPE != 0)) begin : GEN_B0_SPE
            blk0_spe_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk0_spe_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else if ((BLOCK_ID == 1) && (IS_SPE == 0)) begin : GEN_B1_SPA
            blk1_spa_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk1_spa_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else if ((BLOCK_ID == 1) && (IS_SPE != 0)) begin : GEN_B1_SPE
            blk1_spe_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk1_spe_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else if (IS_SPE == 0) begin : GEN_B2_SPA
            blk2_spa_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk2_spa_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end else begin : GEN_B2_SPE
            blk2_spe_ssm_lut_rom_256x419_copy01 u_copy01 (
                .clka(clk), .ena(en), .addra(addr0), .douta(data0),
                .clkb(clk), .enb(en), .addrb(addr1), .doutb(data1));
            blk2_spe_ssm_lut_rom_256x419_copy23 u_copy23 (
                .clka(clk), .ena(en), .addra(addr2), .douta(data2),
                .clkb(clk), .enb(en), .addrb(addr3), .doutb(data3));
        end
    endgenerate
endmodule

// Block1/2 SSM cores issue one channel address per clock.  Reuse only copy01
// and disable its second port, avoiding the four-read replication used by
// block0 while keeping exactly the same 419-bit LUT contents.
module mamba_ssm_lut_rom_1r #(
    parameter integer BLOCK_ID = 1,
    parameter integer IS_SPE = 0
)(input wire clk,input wire en,input wire[7:0]addr,output wire[418:0]data);
    wire[418:0]unused_data;
    generate
        if((BLOCK_ID==1)&&(IS_SPE==0))begin:G10
            blk1_spa_ssm_lut_rom_256x419_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(8'd0),.doutb(unused_data));
        end else if((BLOCK_ID==1)&&(IS_SPE!=0))begin:G11
            blk1_spe_ssm_lut_rom_256x419_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(8'd0),.doutb(unused_data));
        end else if(IS_SPE==0)begin:G20
            blk2_spa_ssm_lut_rom_256x419_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(8'd0),.doutb(unused_data));
        end else begin:G21
            blk2_spe_ssm_lut_rom_256x419_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(8'd0),.doutb(unused_data));
        end
    endgenerate
endmodule

module mamba_spa_conv_param_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire[2:0]addr,output wire[511:0]data);
    generate
        if(BLOCK_ID==0)begin:GEN0 blk0_spa_conv_param_rom_8x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else if(BLOCK_ID==1)begin:GEN1 blk1_spa_conv_param_rom_8x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else begin:GEN2 blk2_spa_conv_param_rom_8x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
    endgenerate
endmodule

module mamba_spe_conv_param_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire addr,output wire[511:0]data);
    generate
        if(BLOCK_ID==0)begin:GEN0 blk0_spe_conv_param_rom_2x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else if(BLOCK_ID==1)begin:GEN1 blk1_spe_conv_param_rom_2x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else begin:GEN2 blk2_spe_conv_param_rom_2x512 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
    endgenerate
endmodule

module mamba_spa_dt_param_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire[2:0]addr,output wire[383:0]data);
    generate
        if(BLOCK_ID==0)begin:GEN0 blk0_spa_dt_param_rom_8x384 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else if(BLOCK_ID==1)begin:GEN1 blk1_spa_dt_param_rom_8x384 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else begin:GEN2 blk2_spa_dt_param_rom_8x384 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
    endgenerate
endmodule

module mamba_spe_dt_param_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire addr,output wire[319:0]data);
    generate
        if(BLOCK_ID==0)begin:GEN0 blk0_spe_dt_param_rom_2x320 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else if(BLOCK_ID==1)begin:GEN1 blk1_spe_dt_param_rom_2x320 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
        else begin:GEN2 blk2_spe_dt_param_rom_2x320 u(.clka(clk),.ena(en),.addra(addr),.douta(data));end
    endgenerate
endmodule

module mamba_spa_out_param_rom_4r #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire[4:0]addr0,input wire[4:0]addr1,
  input wire[4:0]addr2,input wire[4:0]addr3,output wire[543:0]data0,
  output wire[543:0]data1,output wire[543:0]data2,output wire[543:0]data3);
    generate
        if(BLOCK_ID==0)begin:GEN0
            blk0_spa_out_param_rom_32x544_copy01 a(.clka(clk),.ena(en),.addra(addr0),.douta(data0),.clkb(clk),.enb(en),.addrb(addr1),.doutb(data1));
            blk0_spa_out_param_rom_32x544_copy23 b(.clka(clk),.ena(en),.addra(addr2),.douta(data2),.clkb(clk),.enb(en),.addrb(addr3),.doutb(data3));
        end else if(BLOCK_ID==1)begin:GEN1
            blk1_spa_out_param_rom_32x544_copy01 a(.clka(clk),.ena(en),.addra(addr0),.douta(data0),.clkb(clk),.enb(en),.addrb(addr1),.doutb(data1));
            blk1_spa_out_param_rom_32x544_copy23 b(.clka(clk),.ena(en),.addra(addr2),.douta(data2),.clkb(clk),.enb(en),.addrb(addr3),.doutb(data3));
        end else begin:GEN2
            blk2_spa_out_param_rom_32x544_copy01 a(.clka(clk),.ena(en),.addra(addr0),.douta(data0),.clkb(clk),.enb(en),.addrb(addr1),.doutb(data1));
            blk2_spa_out_param_rom_32x544_copy23 b(.clka(clk),.ena(en),.addra(addr2),.douta(data2),.clkb(clk),.enb(en),.addrb(addr3),.doutb(data3));
        end
    endgenerate
endmodule

module mamba_spa_out_param_rom_1r #(
    parameter integer BLOCK_ID=1
)(input wire clk,input wire en,input wire[4:0]addr,output wire[543:0]data);
    wire[543:0]unused_data;
    generate
        if(BLOCK_ID==1)begin:G1 blk1_spa_out_param_rom_32x544_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(5'd0),.doutb(unused_data));end
        else begin:G2 blk2_spa_out_param_rom_32x544_copy01 u(.clka(clk),.ena(en),.addra(addr),.douta(data),.clkb(clk),.enb(1'b0),.addrb(5'd0),.doutb(unused_data));end
    endgenerate
endmodule

module mamba_spe_out_param_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,output wire[1279:0]data);
    generate
        if(BLOCK_ID==0)begin:GEN0 blk0_spe_out_param_rom_1x1280 u(.clka(clk),.ena(en),.addra(1'b0),.douta(data));end
        else if(BLOCK_ID==1)begin:GEN1 blk1_spe_out_param_rom_1x1280 u(.clka(clk),.ena(en),.addra(1'b0),.douta(data));end
        else begin:GEN2 blk2_spe_out_param_rom_1x1280 u(.clka(clk),.ena(en),.addra(1'b0),.douta(data));end
    endgenerate
endmodule

// Channel-major stream ROMs used by the STEAM projection datapaths.  A word
// contains every output row for one four-input group.  BLOCK_ID is resolved at
// elaboration, so only one 16-deep vendor ROM is present in each block.
module mamba_spa_x_stream_weight_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire[3:0]addr,
  output wire[1087:0]data);
    generate
        if(BLOCK_ID==0)begin:G0
            blk0_spa_x_stream_weight_rom_16x1088 u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end else if(BLOCK_ID==1)begin:G1
            b1_sx4_rom u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end else begin:G2
            b2_sx4_rom u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end
    endgenerate
endmodule

module mamba_spa_out_stream_weight_rom #(
    parameter integer BLOCK_ID=0
)(input wire clk,input wire en,input wire[3:0]addr,
  output wire[1023:0]data);
    generate
        if(BLOCK_ID==0)begin:G0
            blk0_spa_out_stream_weight_rom_16x1024 u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end else if(BLOCK_ID==1)begin:G1
            b1_so4_rom u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end else begin:G2
            b2_so4_rom u(
                .clka(clk),.ena(en),.addra(addr),.douta(data));
        end
    endgenerate
endmodule

`default_nettype wire
