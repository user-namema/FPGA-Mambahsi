`timescale 1ns/1ps
`default_nettype none

// Exact fixed input requantization ROM for Block1/2 in_proj.
//
// TABLE_ID:
//   0 = Block1 Spa, multiplier=16384, shift=16
//   1 = Block1 Spe, multiplier=32767, shift=17
//   2 = Block2 Spa, multiplier=16384, shift=16
//   3 = Block2 Spe, multiplier=16384, shift=16
//
// Every instance is a single asynchronous 256x8 distributed ROM.  The parent
// instantiates four copies per observer to provide four independent byte reads
// per clock, then registers all lookup outputs before updating its input RAM.
module mamba_input_q8_lut_256x8 #(
    parameter integer TABLE_ID = 0
) (
    input  wire [7:0] address,
    output wire [7:0] data
);
    (* rom_style = "distributed" *) reg [7:0] lut [0:255];

    generate
        if (TABLE_ID == 1) begin : G_BLOCK1_SPE
            initial begin
        lut[8'h00] = 8'h00;
        lut[8'h01] = 8'h00;
        lut[8'h02] = 8'h00;
        lut[8'h03] = 8'h01;
        lut[8'h04] = 8'h01;
        lut[8'h05] = 8'h01;
        lut[8'h06] = 8'h01;
        lut[8'h07] = 8'h02;
        lut[8'h08] = 8'h02;
        lut[8'h09] = 8'h02;
        lut[8'h0a] = 8'h02;
        lut[8'h0b] = 8'h03;
        lut[8'h0c] = 8'h03;
        lut[8'h0d] = 8'h03;
        lut[8'h0e] = 8'h03;
        lut[8'h0f] = 8'h04;
        lut[8'h10] = 8'h04;
        lut[8'h11] = 8'h04;
        lut[8'h12] = 8'h04;
        lut[8'h13] = 8'h05;
        lut[8'h14] = 8'h05;
        lut[8'h15] = 8'h05;
        lut[8'h16] = 8'h05;
        lut[8'h17] = 8'h06;
        lut[8'h18] = 8'h06;
        lut[8'h19] = 8'h06;
        lut[8'h1a] = 8'h06;
        lut[8'h1b] = 8'h07;
        lut[8'h1c] = 8'h07;
        lut[8'h1d] = 8'h07;
        lut[8'h1e] = 8'h07;
        lut[8'h1f] = 8'h08;
        lut[8'h20] = 8'h08;
        lut[8'h21] = 8'h08;
        lut[8'h22] = 8'h08;
        lut[8'h23] = 8'h09;
        lut[8'h24] = 8'h09;
        lut[8'h25] = 8'h09;
        lut[8'h26] = 8'h09;
        lut[8'h27] = 8'h0a;
        lut[8'h28] = 8'h0a;
        lut[8'h29] = 8'h0a;
        lut[8'h2a] = 8'h0a;
        lut[8'h2b] = 8'h0b;
        lut[8'h2c] = 8'h0b;
        lut[8'h2d] = 8'h0b;
        lut[8'h2e] = 8'h0b;
        lut[8'h2f] = 8'h0c;
        lut[8'h30] = 8'h0c;
        lut[8'h31] = 8'h0c;
        lut[8'h32] = 8'h0c;
        lut[8'h33] = 8'h0d;
        lut[8'h34] = 8'h0d;
        lut[8'h35] = 8'h0d;
        lut[8'h36] = 8'h0d;
        lut[8'h37] = 8'h0e;
        lut[8'h38] = 8'h0e;
        lut[8'h39] = 8'h0e;
        lut[8'h3a] = 8'h0e;
        lut[8'h3b] = 8'h0f;
        lut[8'h3c] = 8'h0f;
        lut[8'h3d] = 8'h0f;
        lut[8'h3e] = 8'h0f;
        lut[8'h3f] = 8'h10;
        lut[8'h40] = 8'h10;
        lut[8'h41] = 8'h10;
        lut[8'h42] = 8'h10;
        lut[8'h43] = 8'h11;
        lut[8'h44] = 8'h11;
        lut[8'h45] = 8'h11;
        lut[8'h46] = 8'h11;
        lut[8'h47] = 8'h12;
        lut[8'h48] = 8'h12;
        lut[8'h49] = 8'h12;
        lut[8'h4a] = 8'h12;
        lut[8'h4b] = 8'h13;
        lut[8'h4c] = 8'h13;
        lut[8'h4d] = 8'h13;
        lut[8'h4e] = 8'h13;
        lut[8'h4f] = 8'h14;
        lut[8'h50] = 8'h14;
        lut[8'h51] = 8'h14;
        lut[8'h52] = 8'h14;
        lut[8'h53] = 8'h15;
        lut[8'h54] = 8'h15;
        lut[8'h55] = 8'h15;
        lut[8'h56] = 8'h15;
        lut[8'h57] = 8'h16;
        lut[8'h58] = 8'h16;
        lut[8'h59] = 8'h16;
        lut[8'h5a] = 8'h16;
        lut[8'h5b] = 8'h17;
        lut[8'h5c] = 8'h17;
        lut[8'h5d] = 8'h17;
        lut[8'h5e] = 8'h17;
        lut[8'h5f] = 8'h18;
        lut[8'h60] = 8'h18;
        lut[8'h61] = 8'h18;
        lut[8'h62] = 8'h18;
        lut[8'h63] = 8'h19;
        lut[8'h64] = 8'h19;
        lut[8'h65] = 8'h19;
        lut[8'h66] = 8'h19;
        lut[8'h67] = 8'h1a;
        lut[8'h68] = 8'h1a;
        lut[8'h69] = 8'h1a;
        lut[8'h6a] = 8'h1a;
        lut[8'h6b] = 8'h1b;
        lut[8'h6c] = 8'h1b;
        lut[8'h6d] = 8'h1b;
        lut[8'h6e] = 8'h1b;
        lut[8'h6f] = 8'h1c;
        lut[8'h70] = 8'h1c;
        lut[8'h71] = 8'h1c;
        lut[8'h72] = 8'h1c;
        lut[8'h73] = 8'h1d;
        lut[8'h74] = 8'h1d;
        lut[8'h75] = 8'h1d;
        lut[8'h76] = 8'h1d;
        lut[8'h77] = 8'h1e;
        lut[8'h78] = 8'h1e;
        lut[8'h79] = 8'h1e;
        lut[8'h7a] = 8'h1e;
        lut[8'h7b] = 8'h1f;
        lut[8'h7c] = 8'h1f;
        lut[8'h7d] = 8'h1f;
        lut[8'h7e] = 8'h1f;
        lut[8'h7f] = 8'h20;
        lut[8'h80] = 8'he0;
        lut[8'h81] = 8'he0;
        lut[8'h82] = 8'he1;
        lut[8'h83] = 8'he1;
        lut[8'h84] = 8'he1;
        lut[8'h85] = 8'he1;
        lut[8'h86] = 8'he2;
        lut[8'h87] = 8'he2;
        lut[8'h88] = 8'he2;
        lut[8'h89] = 8'he2;
        lut[8'h8a] = 8'he3;
        lut[8'h8b] = 8'he3;
        lut[8'h8c] = 8'he3;
        lut[8'h8d] = 8'he3;
        lut[8'h8e] = 8'he4;
        lut[8'h8f] = 8'he4;
        lut[8'h90] = 8'he4;
        lut[8'h91] = 8'he4;
        lut[8'h92] = 8'he5;
        lut[8'h93] = 8'he5;
        lut[8'h94] = 8'he5;
        lut[8'h95] = 8'he5;
        lut[8'h96] = 8'he6;
        lut[8'h97] = 8'he6;
        lut[8'h98] = 8'he6;
        lut[8'h99] = 8'he6;
        lut[8'h9a] = 8'he7;
        lut[8'h9b] = 8'he7;
        lut[8'h9c] = 8'he7;
        lut[8'h9d] = 8'he7;
        lut[8'h9e] = 8'he8;
        lut[8'h9f] = 8'he8;
        lut[8'ha0] = 8'he8;
        lut[8'ha1] = 8'he8;
        lut[8'ha2] = 8'he9;
        lut[8'ha3] = 8'he9;
        lut[8'ha4] = 8'he9;
        lut[8'ha5] = 8'he9;
        lut[8'ha6] = 8'hea;
        lut[8'ha7] = 8'hea;
        lut[8'ha8] = 8'hea;
        lut[8'ha9] = 8'hea;
        lut[8'haa] = 8'heb;
        lut[8'hab] = 8'heb;
        lut[8'hac] = 8'heb;
        lut[8'had] = 8'heb;
        lut[8'hae] = 8'hec;
        lut[8'haf] = 8'hec;
        lut[8'hb0] = 8'hec;
        lut[8'hb1] = 8'hec;
        lut[8'hb2] = 8'hed;
        lut[8'hb3] = 8'hed;
        lut[8'hb4] = 8'hed;
        lut[8'hb5] = 8'hed;
        lut[8'hb6] = 8'hee;
        lut[8'hb7] = 8'hee;
        lut[8'hb8] = 8'hee;
        lut[8'hb9] = 8'hee;
        lut[8'hba] = 8'hef;
        lut[8'hbb] = 8'hef;
        lut[8'hbc] = 8'hef;
        lut[8'hbd] = 8'hef;
        lut[8'hbe] = 8'hf0;
        lut[8'hbf] = 8'hf0;
        lut[8'hc0] = 8'hf0;
        lut[8'hc1] = 8'hf0;
        lut[8'hc2] = 8'hf1;
        lut[8'hc3] = 8'hf1;
        lut[8'hc4] = 8'hf1;
        lut[8'hc5] = 8'hf1;
        lut[8'hc6] = 8'hf2;
        lut[8'hc7] = 8'hf2;
        lut[8'hc8] = 8'hf2;
        lut[8'hc9] = 8'hf2;
        lut[8'hca] = 8'hf3;
        lut[8'hcb] = 8'hf3;
        lut[8'hcc] = 8'hf3;
        lut[8'hcd] = 8'hf3;
        lut[8'hce] = 8'hf4;
        lut[8'hcf] = 8'hf4;
        lut[8'hd0] = 8'hf4;
        lut[8'hd1] = 8'hf4;
        lut[8'hd2] = 8'hf5;
        lut[8'hd3] = 8'hf5;
        lut[8'hd4] = 8'hf5;
        lut[8'hd5] = 8'hf5;
        lut[8'hd6] = 8'hf6;
        lut[8'hd7] = 8'hf6;
        lut[8'hd8] = 8'hf6;
        lut[8'hd9] = 8'hf6;
        lut[8'hda] = 8'hf7;
        lut[8'hdb] = 8'hf7;
        lut[8'hdc] = 8'hf7;
        lut[8'hdd] = 8'hf7;
        lut[8'hde] = 8'hf8;
        lut[8'hdf] = 8'hf8;
        lut[8'he0] = 8'hf8;
        lut[8'he1] = 8'hf8;
        lut[8'he2] = 8'hf9;
        lut[8'he3] = 8'hf9;
        lut[8'he4] = 8'hf9;
        lut[8'he5] = 8'hf9;
        lut[8'he6] = 8'hfa;
        lut[8'he7] = 8'hfa;
        lut[8'he8] = 8'hfa;
        lut[8'he9] = 8'hfa;
        lut[8'hea] = 8'hfb;
        lut[8'heb] = 8'hfb;
        lut[8'hec] = 8'hfb;
        lut[8'hed] = 8'hfb;
        lut[8'hee] = 8'hfc;
        lut[8'hef] = 8'hfc;
        lut[8'hf0] = 8'hfc;
        lut[8'hf1] = 8'hfc;
        lut[8'hf2] = 8'hfd;
        lut[8'hf3] = 8'hfd;
        lut[8'hf4] = 8'hfd;
        lut[8'hf5] = 8'hfd;
        lut[8'hf6] = 8'hfe;
        lut[8'hf7] = 8'hfe;
        lut[8'hf8] = 8'hfe;
        lut[8'hf9] = 8'hfe;
        lut[8'hfa] = 8'hff;
        lut[8'hfb] = 8'hff;
        lut[8'hfc] = 8'hff;
        lut[8'hfd] = 8'hff;
        lut[8'hfe] = 8'h00;
        lut[8'hff] = 8'h00;
            end
        end else begin : G_QUARTER_SCALE
            initial begin
        lut[8'h00] = 8'h00;
        lut[8'h01] = 8'h00;
        lut[8'h02] = 8'h01;
        lut[8'h03] = 8'h01;
        lut[8'h04] = 8'h01;
        lut[8'h05] = 8'h01;
        lut[8'h06] = 8'h02;
        lut[8'h07] = 8'h02;
        lut[8'h08] = 8'h02;
        lut[8'h09] = 8'h02;
        lut[8'h0a] = 8'h03;
        lut[8'h0b] = 8'h03;
        lut[8'h0c] = 8'h03;
        lut[8'h0d] = 8'h03;
        lut[8'h0e] = 8'h04;
        lut[8'h0f] = 8'h04;
        lut[8'h10] = 8'h04;
        lut[8'h11] = 8'h04;
        lut[8'h12] = 8'h05;
        lut[8'h13] = 8'h05;
        lut[8'h14] = 8'h05;
        lut[8'h15] = 8'h05;
        lut[8'h16] = 8'h06;
        lut[8'h17] = 8'h06;
        lut[8'h18] = 8'h06;
        lut[8'h19] = 8'h06;
        lut[8'h1a] = 8'h07;
        lut[8'h1b] = 8'h07;
        lut[8'h1c] = 8'h07;
        lut[8'h1d] = 8'h07;
        lut[8'h1e] = 8'h08;
        lut[8'h1f] = 8'h08;
        lut[8'h20] = 8'h08;
        lut[8'h21] = 8'h08;
        lut[8'h22] = 8'h09;
        lut[8'h23] = 8'h09;
        lut[8'h24] = 8'h09;
        lut[8'h25] = 8'h09;
        lut[8'h26] = 8'h0a;
        lut[8'h27] = 8'h0a;
        lut[8'h28] = 8'h0a;
        lut[8'h29] = 8'h0a;
        lut[8'h2a] = 8'h0b;
        lut[8'h2b] = 8'h0b;
        lut[8'h2c] = 8'h0b;
        lut[8'h2d] = 8'h0b;
        lut[8'h2e] = 8'h0c;
        lut[8'h2f] = 8'h0c;
        lut[8'h30] = 8'h0c;
        lut[8'h31] = 8'h0c;
        lut[8'h32] = 8'h0d;
        lut[8'h33] = 8'h0d;
        lut[8'h34] = 8'h0d;
        lut[8'h35] = 8'h0d;
        lut[8'h36] = 8'h0e;
        lut[8'h37] = 8'h0e;
        lut[8'h38] = 8'h0e;
        lut[8'h39] = 8'h0e;
        lut[8'h3a] = 8'h0f;
        lut[8'h3b] = 8'h0f;
        lut[8'h3c] = 8'h0f;
        lut[8'h3d] = 8'h0f;
        lut[8'h3e] = 8'h10;
        lut[8'h3f] = 8'h10;
        lut[8'h40] = 8'h10;
        lut[8'h41] = 8'h10;
        lut[8'h42] = 8'h11;
        lut[8'h43] = 8'h11;
        lut[8'h44] = 8'h11;
        lut[8'h45] = 8'h11;
        lut[8'h46] = 8'h12;
        lut[8'h47] = 8'h12;
        lut[8'h48] = 8'h12;
        lut[8'h49] = 8'h12;
        lut[8'h4a] = 8'h13;
        lut[8'h4b] = 8'h13;
        lut[8'h4c] = 8'h13;
        lut[8'h4d] = 8'h13;
        lut[8'h4e] = 8'h14;
        lut[8'h4f] = 8'h14;
        lut[8'h50] = 8'h14;
        lut[8'h51] = 8'h14;
        lut[8'h52] = 8'h15;
        lut[8'h53] = 8'h15;
        lut[8'h54] = 8'h15;
        lut[8'h55] = 8'h15;
        lut[8'h56] = 8'h16;
        lut[8'h57] = 8'h16;
        lut[8'h58] = 8'h16;
        lut[8'h59] = 8'h16;
        lut[8'h5a] = 8'h17;
        lut[8'h5b] = 8'h17;
        lut[8'h5c] = 8'h17;
        lut[8'h5d] = 8'h17;
        lut[8'h5e] = 8'h18;
        lut[8'h5f] = 8'h18;
        lut[8'h60] = 8'h18;
        lut[8'h61] = 8'h18;
        lut[8'h62] = 8'h19;
        lut[8'h63] = 8'h19;
        lut[8'h64] = 8'h19;
        lut[8'h65] = 8'h19;
        lut[8'h66] = 8'h1a;
        lut[8'h67] = 8'h1a;
        lut[8'h68] = 8'h1a;
        lut[8'h69] = 8'h1a;
        lut[8'h6a] = 8'h1b;
        lut[8'h6b] = 8'h1b;
        lut[8'h6c] = 8'h1b;
        lut[8'h6d] = 8'h1b;
        lut[8'h6e] = 8'h1c;
        lut[8'h6f] = 8'h1c;
        lut[8'h70] = 8'h1c;
        lut[8'h71] = 8'h1c;
        lut[8'h72] = 8'h1d;
        lut[8'h73] = 8'h1d;
        lut[8'h74] = 8'h1d;
        lut[8'h75] = 8'h1d;
        lut[8'h76] = 8'h1e;
        lut[8'h77] = 8'h1e;
        lut[8'h78] = 8'h1e;
        lut[8'h79] = 8'h1e;
        lut[8'h7a] = 8'h1f;
        lut[8'h7b] = 8'h1f;
        lut[8'h7c] = 8'h1f;
        lut[8'h7d] = 8'h1f;
        lut[8'h7e] = 8'h20;
        lut[8'h7f] = 8'h20;
        lut[8'h80] = 8'he0;
        lut[8'h81] = 8'he0;
        lut[8'h82] = 8'he0;
        lut[8'h83] = 8'he1;
        lut[8'h84] = 8'he1;
        lut[8'h85] = 8'he1;
        lut[8'h86] = 8'he1;
        lut[8'h87] = 8'he2;
        lut[8'h88] = 8'he2;
        lut[8'h89] = 8'he2;
        lut[8'h8a] = 8'he2;
        lut[8'h8b] = 8'he3;
        lut[8'h8c] = 8'he3;
        lut[8'h8d] = 8'he3;
        lut[8'h8e] = 8'he3;
        lut[8'h8f] = 8'he4;
        lut[8'h90] = 8'he4;
        lut[8'h91] = 8'he4;
        lut[8'h92] = 8'he4;
        lut[8'h93] = 8'he5;
        lut[8'h94] = 8'he5;
        lut[8'h95] = 8'he5;
        lut[8'h96] = 8'he5;
        lut[8'h97] = 8'he6;
        lut[8'h98] = 8'he6;
        lut[8'h99] = 8'he6;
        lut[8'h9a] = 8'he6;
        lut[8'h9b] = 8'he7;
        lut[8'h9c] = 8'he7;
        lut[8'h9d] = 8'he7;
        lut[8'h9e] = 8'he7;
        lut[8'h9f] = 8'he8;
        lut[8'ha0] = 8'he8;
        lut[8'ha1] = 8'he8;
        lut[8'ha2] = 8'he8;
        lut[8'ha3] = 8'he9;
        lut[8'ha4] = 8'he9;
        lut[8'ha5] = 8'he9;
        lut[8'ha6] = 8'he9;
        lut[8'ha7] = 8'hea;
        lut[8'ha8] = 8'hea;
        lut[8'ha9] = 8'hea;
        lut[8'haa] = 8'hea;
        lut[8'hab] = 8'heb;
        lut[8'hac] = 8'heb;
        lut[8'had] = 8'heb;
        lut[8'hae] = 8'heb;
        lut[8'haf] = 8'hec;
        lut[8'hb0] = 8'hec;
        lut[8'hb1] = 8'hec;
        lut[8'hb2] = 8'hec;
        lut[8'hb3] = 8'hed;
        lut[8'hb4] = 8'hed;
        lut[8'hb5] = 8'hed;
        lut[8'hb6] = 8'hed;
        lut[8'hb7] = 8'hee;
        lut[8'hb8] = 8'hee;
        lut[8'hb9] = 8'hee;
        lut[8'hba] = 8'hee;
        lut[8'hbb] = 8'hef;
        lut[8'hbc] = 8'hef;
        lut[8'hbd] = 8'hef;
        lut[8'hbe] = 8'hef;
        lut[8'hbf] = 8'hf0;
        lut[8'hc0] = 8'hf0;
        lut[8'hc1] = 8'hf0;
        lut[8'hc2] = 8'hf0;
        lut[8'hc3] = 8'hf1;
        lut[8'hc4] = 8'hf1;
        lut[8'hc5] = 8'hf1;
        lut[8'hc6] = 8'hf1;
        lut[8'hc7] = 8'hf2;
        lut[8'hc8] = 8'hf2;
        lut[8'hc9] = 8'hf2;
        lut[8'hca] = 8'hf2;
        lut[8'hcb] = 8'hf3;
        lut[8'hcc] = 8'hf3;
        lut[8'hcd] = 8'hf3;
        lut[8'hce] = 8'hf3;
        lut[8'hcf] = 8'hf4;
        lut[8'hd0] = 8'hf4;
        lut[8'hd1] = 8'hf4;
        lut[8'hd2] = 8'hf4;
        lut[8'hd3] = 8'hf5;
        lut[8'hd4] = 8'hf5;
        lut[8'hd5] = 8'hf5;
        lut[8'hd6] = 8'hf5;
        lut[8'hd7] = 8'hf6;
        lut[8'hd8] = 8'hf6;
        lut[8'hd9] = 8'hf6;
        lut[8'hda] = 8'hf6;
        lut[8'hdb] = 8'hf7;
        lut[8'hdc] = 8'hf7;
        lut[8'hdd] = 8'hf7;
        lut[8'hde] = 8'hf7;
        lut[8'hdf] = 8'hf8;
        lut[8'he0] = 8'hf8;
        lut[8'he1] = 8'hf8;
        lut[8'he2] = 8'hf8;
        lut[8'he3] = 8'hf9;
        lut[8'he4] = 8'hf9;
        lut[8'he5] = 8'hf9;
        lut[8'he6] = 8'hf9;
        lut[8'he7] = 8'hfa;
        lut[8'he8] = 8'hfa;
        lut[8'he9] = 8'hfa;
        lut[8'hea] = 8'hfa;
        lut[8'heb] = 8'hfb;
        lut[8'hec] = 8'hfb;
        lut[8'hed] = 8'hfb;
        lut[8'hee] = 8'hfb;
        lut[8'hef] = 8'hfc;
        lut[8'hf0] = 8'hfc;
        lut[8'hf1] = 8'hfc;
        lut[8'hf2] = 8'hfc;
        lut[8'hf3] = 8'hfd;
        lut[8'hf4] = 8'hfd;
        lut[8'hf5] = 8'hfd;
        lut[8'hf6] = 8'hfd;
        lut[8'hf7] = 8'hfe;
        lut[8'hf8] = 8'hfe;
        lut[8'hf9] = 8'hfe;
        lut[8'hfa] = 8'hfe;
        lut[8'hfb] = 8'hff;
        lut[8'hfc] = 8'hff;
        lut[8'hfd] = 8'hff;
        lut[8'hfe] = 8'hff;
        lut[8'hff] = 8'h00;
            end
        end
    endgenerate

    assign data = lut[address];

`ifndef SYNTHESIS
    initial begin
        if ((TABLE_ID < 0) || (TABLE_ID > 3)) begin
            $error("Unsupported input-q8 LUT TABLE_ID=%0d", TABLE_ID);
            $finish;
        end
    end
`endif
endmodule

// Pool sums use all ten two's-complement bits. Initialization is elaboration
// time only: there is no run-time multiplier and no extra pipeline stage.
module mamba_pool_q8_lut_1024x8 #(
    parameter signed [15:0] MULTIPLIER = 16'sd16409,
    parameter signed [6:0] SHIFT = 7'sd16
) (
    input wire [9:0] address,
    output wire [7:0] data
);
    (* rom_style = "distributed" *) reg [7:0] lut [0:1023];

    function [7:0] make_entry;
        input [9:0] code;
        reg signed [9:0] sample;
        reg signed [25:0] product;
        reg signed [63:0] extended_product, magnitude, rounded;
        begin
            sample = $signed(code);
            product = sample * MULTIPLIER;
            extended_product = {{38{product[25]}},product};
            magnitude = (extended_product<0) ? -extended_product : extended_product;
            if (SHIFT>0)
                rounded = (magnitude+(64'sd1<<<(SHIFT-1))) >>> SHIFT;
            else if (SHIFT<0)
                rounded = magnitude <<< (-SHIFT);
            else
                rounded = magnitude;
            if (extended_product<0) rounded=-rounded;
            if (rounded>127) make_entry=8'h7f;
            else if (rounded< -128) make_entry=8'h80;
            else make_entry=rounded[7:0];
        end
    endfunction

    integer index;
    initial begin
        for (index=0; index<1024; index=index+1)
            lut[index]=make_entry(index[9:0]);
    end
    assign data=lut[address];
endmodule

`default_nettype wire
