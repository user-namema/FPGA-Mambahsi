// D1 frozen out_proj biases, first_tile_layers.
function signed [31:0] spa_out_bias;
    input integer block_id;
    input [4:0] index;
    begin
        case (block_id)
            0: case (index)
                5'd0: spa_out_bias = 32'sh00000255;
                5'd1: spa_out_bias = 32'shfffffe10;
                5'd2: spa_out_bias = 32'sh0000015f;
                5'd3: spa_out_bias = 32'sh0000027a;
                5'd4: spa_out_bias = 32'sh0000001d;
                5'd5: spa_out_bias = 32'shfffffcb0;
                5'd6: spa_out_bias = 32'sh000001e2;
                5'd7: spa_out_bias = 32'shfffffdd5;
                5'd8: spa_out_bias = 32'sh00000080;
                5'd9: spa_out_bias = 32'sh00000068;
                5'd10: spa_out_bias = 32'sh000001ac;
                5'd11: spa_out_bias = 32'sh00000348;
                5'd12: spa_out_bias = 32'shffffffc6;
                5'd13: spa_out_bias = 32'shffffffc5;
                5'd14: spa_out_bias = 32'sh000000f7;
                5'd15: spa_out_bias = 32'sh00000084;
                5'd16: spa_out_bias = 32'sh00000056;
                5'd17: spa_out_bias = 32'sh000001f5;
                5'd18: spa_out_bias = 32'shffffffd0;
                5'd19: spa_out_bias = 32'sh0000021a;
                5'd20: spa_out_bias = 32'shfffffee1;
                5'd21: spa_out_bias = 32'sh00000306;
                5'd22: spa_out_bias = 32'sh000001c9;
                5'd23: spa_out_bias = 32'shfffffec1;
                5'd24: spa_out_bias = 32'shfffffeb2;
                5'd25: spa_out_bias = 32'sh0000016d;
                5'd26: spa_out_bias = 32'shfffffd22;
                5'd27: spa_out_bias = 32'shffffff93;
                5'd28: spa_out_bias = 32'shffffffbe;
                5'd29: spa_out_bias = 32'shfffffeae;
                5'd30: spa_out_bias = 32'shfffffe89;
                5'd31: spa_out_bias = 32'sh00000073;
                default: spa_out_bias = 0;
            endcase
            1: case (index)
                5'd0: spa_out_bias = 32'sh000002df;
                5'd1: spa_out_bias = 32'shfffffe7a;
                5'd2: spa_out_bias = 32'shfffffe49;
                5'd3: spa_out_bias = 32'sh000000ff;
                5'd4: spa_out_bias = 32'sh00000238;
                5'd5: spa_out_bias = 32'sh00000004;
                5'd6: spa_out_bias = 32'shfffffe94;
                5'd7: spa_out_bias = 32'shffffffa7;
                5'd8: spa_out_bias = 32'shfffffefb;
                5'd9: spa_out_bias = 32'shfffffeaa;
                5'd10: spa_out_bias = 32'shffffff86;
                5'd11: spa_out_bias = 32'sh00000003;
                5'd12: spa_out_bias = 32'shfffffd99;
                5'd13: spa_out_bias = 32'sh00000094;
                5'd14: spa_out_bias = 32'sh000000ec;
                5'd15: spa_out_bias = 32'sh00000104;
                5'd16: spa_out_bias = 32'sh00000047;
                5'd17: spa_out_bias = 32'sh0000006d;
                5'd18: spa_out_bias = 32'shffffff64;
                5'd19: spa_out_bias = 32'sh000001ce;
                5'd20: spa_out_bias = 32'sh00000167;
                5'd21: spa_out_bias = 32'shffffffed;
                5'd22: spa_out_bias = 32'sh0000034b;
                5'd23: spa_out_bias = 32'sh0000023e;
                5'd24: spa_out_bias = 32'sh0000025e;
                5'd25: spa_out_bias = 32'sh000000be;
                5'd26: spa_out_bias = 32'sh000000d3;
                5'd27: spa_out_bias = 32'sh0000002e;
                5'd28: spa_out_bias = 32'sh00000047;
                5'd29: spa_out_bias = 32'shfffffe39;
                5'd30: spa_out_bias = 32'sh00000189;
                5'd31: spa_out_bias = 32'shffffffe4;
                default: spa_out_bias = 0;
            endcase
            2: case (index)
                5'd0: spa_out_bias = 32'shffffffee;
                5'd1: spa_out_bias = 32'sh000000e7;
                5'd2: spa_out_bias = 32'sh00000056;
                5'd3: spa_out_bias = 32'shffffffc8;
                5'd4: spa_out_bias = 32'shffffff1f;
                5'd5: spa_out_bias = 32'sh00000169;
                5'd6: spa_out_bias = 32'shffffffde;
                5'd7: spa_out_bias = 32'shffffffe9;
                5'd8: spa_out_bias = 32'shffffffc2;
                5'd9: spa_out_bias = 32'shffffff8d;
                5'd10: spa_out_bias = 32'shffffff5d;
                5'd11: spa_out_bias = 32'sh00000088;
                5'd12: spa_out_bias = 32'sh000000b4;
                5'd13: spa_out_bias = 32'sh000000dc;
                5'd14: spa_out_bias = 32'sh00000124;
                5'd15: spa_out_bias = 32'shfffffff6;
                5'd16: spa_out_bias = 32'sh00000133;
                5'd17: spa_out_bias = 32'shfffffeec;
                5'd18: spa_out_bias = 32'shfffffffb;
                5'd19: spa_out_bias = 32'sh000000a5;
                5'd20: spa_out_bias = 32'sh000001a5;
                5'd21: spa_out_bias = 32'sh000000f0;
                5'd22: spa_out_bias = 32'sh0000003f;
                5'd23: spa_out_bias = 32'sh000000f8;
                5'd24: spa_out_bias = 32'sh000000b2;
                5'd25: spa_out_bias = 32'sh00000066;
                5'd26: spa_out_bias = 32'sh000000dc;
                5'd27: spa_out_bias = 32'sh000000c3;
                5'd28: spa_out_bias = 32'shfffffef1;
                5'd29: spa_out_bias = 32'shffffff6a;
                5'd30: spa_out_bias = 32'sh0000007b;
                5'd31: spa_out_bias = 32'shffffff34;
                default: spa_out_bias = 0;
            endcase
            default: spa_out_bias = 0;
        endcase
    end
endfunction
