`timescale 1ns/1ps
`default_nettype none

// QAT-aligned dense classifier for the final 4x4 feature map:
//   Conv1x1 32->64 (folded BN) -> ReLU -> Conv1x1 64->9.
//
// Two output rows share every signed INT8 activation through the common
// three-cycle (A+D)*B DSP Macro.  The datapath therefore uses 16 packed DSPs
// instead of 32 ordinary INT8 multipliers:
//   head1: two 16-input chunks per output pair (64 issue cycles/pixel),
//   head2: four 16-input chunks per class pair (20 issue cycles/pixel; class
//          nine is a masked dummy paired with the real final class eight).
(* KEEP_HIERARCHY = "yes" *)
module mamba_classifier_head_stream #(
    parameter signed [15:0] CONV1_MULTIPLIER =25465,
    parameter integer CONV1_SHIFT =18,
    parameter signed [15:0] CONV2_MULTIPLIER =25784,
    parameter integer CONV2_SHIFT =22
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[7:0]in_pixel_addr,
    input wire[4:0]in_channel_base,input wire[63:0]in_data,
    output reg out_valid,output reg[3:0]out_pixel_addr,
    output reg[71:0]out_logits,output reg[3:0]out_class,
    output reg done,output wire busy
);
    wire [63:0] d1_head_input;
    reg[191:0]input_collect;
    wire input_wr_en=in_valid&&(in_channel_base==5'd24);
    wire[255:0]input_wr_data={d1_head_input,input_collect};
    reg input_rd_en;reg[3:0]input_rd_addr;wire[255:0]input_rd_data;
    reg[31:0]ready_count;
    mamba_input_bram_256x256 u_input_bram(
        .clka(clk),.ena(input_wr_en),.wea({input_wr_en}),
        .addra({4'd0,in_pixel_addr[3:0]}),.dina(input_wr_data),
        .clkb(clk),.enb(input_rd_en),.addrb({4'd0,input_rd_addr}),
        .doutb(input_rd_data));
    always@(posedge clk)begin
        if(!rst_n)begin input_collect<=0;ready_count<=0;end
        else if(frame_start)begin ready_count<=0;end
        else if(in_valid)begin
            if(in_channel_base!=5'd24)
                input_collect[in_channel_base*8+:64]<=d1_head_input;
            if(in_channel_base==5'd24)ready_count<=ready_count+1'b1;
        end
    end

    genvar head_input_lane;
    generate for (head_input_lane=0; head_input_lane<8; head_input_lane=head_input_lane+1) begin : G_D1_INPUT
        mamba_d1_head_input_lut u_lut (
            .address(in_data[head_input_lane*8 +: 8]),
            .data(d1_head_input[head_input_lane*8 +: 8]));
    end endgenerate
    localparam[2:0]ST_WAIT=0,ST_READ=1,ST_CAPTURE=2,ST_HEAD1=3,
        ST_DRAIN1=4,ST_HEAD2=5,ST_DRAIN2=6;
    reg[2:0]state;reg[31:0]pixel_q;
    reg[5:0]head1_row_base;reg head1_chunk;
    reg[3:0]head2_class_base;reg[1:0]head2_subchunk;
    reg[255:0]input_activation;reg[511:0]head1_activation;

    wire head1_request=state==ST_HEAD1;
    wire head2_request=state==ST_HEAD2;
    wire request_valid=head1_request|head2_request;
    // {layer,pixel,row_base,chunk,00}; row_base always names the low member.
    wire[15:0]request_meta=head2_request
        ?{1'b1,pixel_q[3:0],3'd0,head2_class_base,
          head2_subchunk,2'b00}
        :{1'b0,pixel_q[3:0],1'b0,head1_row_base,
          1'b0,head1_chunk,2'b00};

    // Two independent ROM reads preserve the original service rate while the
    // output pair shares the 16 DSP multipliers.  The last head2 high row is a
    // masked dummy, so its address is kept inside the legal 0..17 range.
    wire[287:0]head1_params_low,head1_params_high;
    head_conv1_param_rom_64x288 u_head1_rom_low(
        .clka(clk),.ena(head1_request),.addra(head1_row_base),
        .douta(head1_params_low));
    head_conv1_param_rom_64x288 u_head1_rom_high(
        .clka(clk),.ena(head1_request),.addra(head1_row_base+1'b1),
        .douta(head1_params_high));

    wire[4:0]head2_rom_addr_low=head2_class_base*2
        +head2_subchunk[1];
    wire[4:0]head2_rom_addr_high=(head2_class_base==4'd8)
        ?5'd0:(head2_class_base+1'b1)*2+head2_subchunk[1];
    wire[287:0]head2_params_low,head2_params_high_raw;
    head_conv2_param_rom_18x288 u_head2_rom_low(
        .clka(clk),.ena(head2_request),.addra(head2_rom_addr_low),
        .douta(head2_params_low));
    head_conv2_param_rom_18x288 u_head2_rom_high(
        .clka(clk),.ena(head2_request),.addra(head2_rom_addr_high),
        .douta(head2_params_high_raw));

    reg request_valid_d;reg[15:0]request_meta_d;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin request_valid_d<=0;request_meta_d<=0;end
        else begin
            request_valid_d<=request_valid;
            if(request_valid)request_meta_d<=request_meta;
        end
    end

    wire request_is_head2=request_meta_d[15];
    wire[1:0]request_chunk=request_meta_d[3:2];
    wire[6:0]request_row_base=request_meta_d[10:4];
    wire request_dummy_high=request_is_head2&&(request_row_base==7'd8);
    wire[127:0]dot_activation=request_is_head2
        ?head1_activation[request_chunk*128+:128]
        :input_activation[request_chunk[0]*128+:128];
    wire[127:0]dot_weight_low=request_is_head2
        ?head2_params_low[request_chunk[0]*128+:128]
        :head1_params_low[request_chunk[0]*128+:128];
    wire[127:0]dot_weight_high=request_dummy_high?128'd0:
        (request_is_head2
          ?head2_params_high_raw[request_chunk[0]*128+:128]
          :head1_params_high[request_chunk[0]*128+:128]);
    wire[31:0]dot_bias_low=request_is_head2
        ?head2_params_low[287:256]:head1_params_low[287:256];
    wire[31:0]dot_bias_high=request_dummy_high?32'd0:
        (request_is_head2?head2_params_high_raw[287:256]
                         :head1_params_high[287:256]);

    wire dot_valid;wire signed[19:0]dot_sum_low,dot_sum_high;
    wire[79:0]dot_meta;
    mamba_dot16_pair_pipeline#(80)u_dot_pair(
        .clk(clk),.rst_n(rst_n),.frame_start(frame_start),
        .in_valid(request_valid_d),.activation(dot_activation),
        .weight_low(dot_weight_low),.weight_high(dot_weight_high),
        .in_meta({dot_bias_high,dot_bias_low,request_meta_d}),
        .out_valid(dot_valid),.out_sum_low(dot_sum_low),
        .out_sum_high(dot_sum_high),.out_meta(dot_meta));

    wire dot_is_head2=dot_meta[15];
    wire[1:0]dot_chunk=dot_meta[3:2];
    wire[6:0]dot_row_base=dot_meta[10:4];
    wire[31:0]dot_bias_low_aligned=dot_meta[47:16];
    wire[31:0]dot_bias_high_aligned=dot_meta[79:48];
    wire dot_final_chunk=dot_is_head2?(dot_chunk==2'd3)
                                      :(dot_chunk==2'd1);
    reg signed[20:0]partial_low,partial_high;
    wire signed[20:0]dot_low_ext={dot_sum_low[19],dot_sum_low};
    wire signed[20:0]dot_high_ext={dot_sum_high[19],dot_sum_high};
    wire signed[20:0]partial_plus_low=partial_low+dot_low_ext;
    wire signed[20:0]partial_plus_high=partial_high+dot_high_ext;
    wire signed[20:0]requant_input_low=partial_plus_low
        +$signed(dot_bias_low_aligned[20:0]);
    wire signed[20:0]requant_input_high=partial_plus_high
        +$signed(dot_bias_high_aligned[20:0]);
    wire requant_valid=dot_valid&&dot_final_chunk;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin partial_low<=0;partial_high<=0;end
        else if(dot_valid)begin
            if(dot_chunk==0)begin
                partial_low<=dot_low_ext;partial_high<=dot_high_ext;
            end else if(!dot_final_chunk)begin
                partial_low<=partial_plus_low;partial_high<=partial_plus_high;
            end
        end
    end

    wire signed[15:0]requant_multiplier=dot_is_head2
        ?CONV2_MULTIPLIER:CONV1_MULTIPLIER;
    wire signed[36:0]requant_product_low,requant_product_high;
    requant_mult_21x16 u_requant_low(
        .CLK(clk),.A(requant_input_low),.B(requant_multiplier),
        .P(requant_product_low));
    requant_mult_21x16 u_requant_high(
        .CLK(clk),.A(requant_input_high),.B(requant_multiplier),
        .P(requant_product_high));

    reg qv0,qv1,qv2;reg[15:0]qm0,qm1,qm2;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin
            qv0<=0;qv1<=0;qv2<=0;qm0<=0;qm1<=0;qm2<=0;
        end else begin
            qv0<=requant_valid;qv1<=qv0;qv2<=qv1;
            if(requant_valid)qm0<=dot_meta[15:0];
            if(qv0)qm1<=qm0;if(qv1)qm2<=qm1;
        end
    end

    function[7:0]quantize_int8;
        input signed[36:0]value;input integer shift_value;
        reg signed[63:0]magnitude,rounded;
        begin
            magnitude=value<0?-$signed({{27{value[36]}},value})
                             :$signed({{27{value[36]}},value});
            rounded=(magnitude+(64'sd1<<<(shift_value-1)))>>>shift_value;
            if(value<0)rounded=-rounded;
            if(rounded>127)quantize_int8=8'h7f;
            else if(rounded< -128)quantize_int8=8'h80;
            else quantize_int8=rounded[7:0];
        end
    endfunction

    wire q_is_head2=qm2[15];wire[3:0]q_pixel=qm2[14:11];
    wire[6:0]q_row_base=qm2[10:4];
    wire[7:0]q_code_low=quantize_int8(
        requant_product_low,q_is_head2?CONV2_SHIFT:CONV1_SHIFT);
    wire[7:0]q_code_high=quantize_int8(
        requant_product_high,q_is_head2?CONV2_SHIFT:CONV1_SHIFT);
    // Kept as named observation points for the existing exact testbench.
    // They are no longer part of the classifier decision path.
    wire[7:0]q_relu_low=q_code_low[7]?8'd0:q_code_low;
    wire[7:0]q_relu_high=q_code_high[7]?8'd0:q_code_high;
    // Register quantize/saturate before any class comparison.  The former
    // qm2 -> quantize -> two running compares -> out_class path contained
    // roughly twenty LUT/CARRY levels.  This boundary keeps the arithmetic
    // result exact and adds latency only, not classifier service II.
    reg result_valid_q;
    reg[15:0]result_meta_q;
    reg[7:0]result_code_low_q,result_code_high_q;
    always@(posedge clk)begin
        if(!rst_n||frame_start)result_valid_q<=0;
        else result_valid_q<=qv2;
        result_meta_q<=qm2;
        result_code_low_q<=q_code_low;
        result_code_high_q<=q_code_high;
    end
    wire result_is_head2=result_meta_q[15];
    wire[3:0]result_pixel=result_meta_q[14:11];
    wire[6:0]result_row_base=result_meta_q[10:4];
    wire[7:0]result_relu_low=result_code_low_q[7]?8'd0:
        result_code_low_q;
    wire[7:0]result_relu_high=result_code_high_q[7]?8'd0:
        result_code_high_q;
    // Compatibility observation used by the exact head1 testbench.  The
    // functional write path below uses the registered commit valid.
    wire head1_result_valid=qv2&&!q_is_head2;
    wire head1_commit_valid=result_valid_q&&!result_is_head2;
    wire head2_result_valid=result_valid_q&&result_is_head2;

    reg[63:0]logit_pack;
    reg final_valid_q;
    reg[71:0]final_logits_q;
    reg[3:0]final_pixel_q;
    reg group_valid_q;
    reg[71:0]group_logits_q;
    reg[3:0]group_pixel_q;
    reg signed[7:0]group0_logit_q,group1_logit_q,group2_logit_q;
    reg[3:0]group0_class_q,group1_class_q,group2_class_q;

    wire signed[7:0]fl0=$signed(final_logits_q[7:0]);
    wire signed[7:0]fl1=$signed(final_logits_q[15:8]);
    wire signed[7:0]fl2=$signed(final_logits_q[23:16]);
    wire signed[7:0]fl3=$signed(final_logits_q[31:24]);
    wire signed[7:0]fl4=$signed(final_logits_q[39:32]);
    wire signed[7:0]fl5=$signed(final_logits_q[47:40]);
    wire signed[7:0]fl6=$signed(final_logits_q[55:48]);
    wire signed[7:0]fl7=$signed(final_logits_q[63:56]);
    wire signed[7:0]fl8=$signed(final_logits_q[71:64]);
    wire g0_take1=fl1>fl0;
    wire signed[7:0]g0_first=g0_take1?fl1:fl0;
    wire[3:0]g0_first_class=g0_take1?4'd1:4'd0;
    wire g0_take2=fl2>g0_first;
    wire signed[7:0]g0_logit=g0_take2?fl2:g0_first;
    wire[3:0]g0_class=g0_take2?4'd2:g0_first_class;
    wire g1_take4=fl4>fl3;
    wire signed[7:0]g1_first=g1_take4?fl4:fl3;
    wire[3:0]g1_first_class=g1_take4?4'd4:4'd3;
    wire g1_take5=fl5>g1_first;
    wire signed[7:0]g1_logit=g1_take5?fl5:g1_first;
    wire[3:0]g1_class=g1_take5?4'd5:g1_first_class;
    wire g2_take7=fl7>fl6;
    wire signed[7:0]g2_first=g2_take7?fl7:fl6;
    wire[3:0]g2_first_class=g2_take7?4'd7:4'd6;
    wire g2_take8=fl8>g2_first;
    wire signed[7:0]g2_logit=g2_take8?fl8:g2_first;
    wire[3:0]g2_class=g2_take8?4'd8:g2_first_class;

    wire final_take1=group1_logit_q>group0_logit_q;
    wire signed[7:0]final_first_logit=final_take1?group1_logit_q:
        group0_logit_q;
    wire[3:0]final_first_class=final_take1?group1_class_q:
        group0_class_q;
    wire final_take2=group2_logit_q>final_first_logit;
    wire[3:0]final_best_class=final_take2?group2_class_q:
        final_first_class;

    always@(posedge clk)begin
        if(!rst_n)begin
            head1_activation<=0;logit_pack<=0;final_valid_q<=0;
            group_valid_q<=0;out_valid<=0;out_pixel_addr<=0;out_logits<=0;
            out_class<=0;done<=0;
        end else if(frame_start)begin
            final_valid_q<=0;group_valid_q<=0;out_valid<=0;
            out_pixel_addr<=0;out_class<=0;done<=0;
        end else begin
            final_valid_q<=head2_result_valid&&(result_row_base==8);
            group_valid_q<=final_valid_q;
            out_valid<=group_valid_q;done<=0;
            if(head1_commit_valid)
                head1_activation[result_row_base*8+:16]
                    <={result_relu_high,result_relu_low};
            if(head2_result_valid)begin
                if(result_row_base<8)
                    logit_pack[result_row_base*8+:16]
                        <={result_code_high_q,result_code_low_q};
                else begin
                    final_logits_q<={result_code_low_q,logit_pack};
                    final_pixel_q<=result_pixel;
                end
            end
            if(final_valid_q)begin
                group0_logit_q<=g0_logit;group0_class_q<=g0_class;
                group1_logit_q<=g1_logit;group1_class_q<=g1_class;
                group2_logit_q<=g2_logit;group2_class_q<=g2_class;
                group_logits_q<=final_logits_q;
                group_pixel_q<=final_pixel_q;
            end
            if(group_valid_q)begin
                out_pixel_addr<=group_pixel_q;
                out_logits<=group_logits_q;
                out_class<=final_best_class;
                if(group_pixel_q==15)done<=1;
            end
        end
    end

    always@(posedge clk)begin
        if(!rst_n)begin
            state<=ST_WAIT;pixel_q<=0;head1_row_base<=0;head1_chunk<=0;
            head2_class_base<=0;head2_subchunk<=0;
            input_activation<=0;input_rd_en<=0;input_rd_addr<=0;
        end else if(frame_start)begin
            state<=ST_WAIT;pixel_q<=0;head1_row_base<=0;head1_chunk<=0;
            head2_class_base<=0;head2_subchunk<=0;
            input_rd_en<=0;input_rd_addr<=0;
        end else begin
            input_rd_en<=0;
            case(state)
                ST_WAIT:begin
                    head1_row_base<=0;head1_chunk<=0;
                    head2_class_base<=0;head2_subchunk<=0;
                    if(pixel_q<ready_count)begin
                        input_rd_en<=1;input_rd_addr<=pixel_q[3:0];
                        state<=ST_READ;
                    end
                end
                ST_READ:state<=ST_CAPTURE;
                ST_CAPTURE:begin
                    input_activation<=input_rd_data;state<=ST_HEAD1;
                end
                ST_HEAD1:begin
                    if(head1_chunk)begin
                        head1_chunk<=0;
                        if(head1_row_base==62)begin
                            head1_row_base<=0;state<=ST_DRAIN1;
                        end else head1_row_base<=head1_row_base+2;
                    end else head1_chunk<=1;
                end
                ST_DRAIN1:if(head1_commit_valid&&(result_row_base==62))begin
                    head2_class_base<=0;head2_subchunk<=0;state<=ST_HEAD2;
                end
                ST_HEAD2:begin
                    if(head2_subchunk==3)begin
                        head2_subchunk<=0;
                        if(head2_class_base==8)begin
                            head2_class_base<=0;state<=ST_DRAIN2;
                        end else head2_class_base<=head2_class_base+2;
                    end else head2_subchunk<=head2_subchunk+1'b1;
                end
                ST_DRAIN2:if(head2_result_valid&&(result_row_base==8))begin
                    pixel_q<=pixel_q+1'b1;state<=ST_WAIT;
                end
                default:state<=ST_WAIT;
            endcase
        end
    end
    assign busy=(state!=ST_WAIT)||(pixel_q<ready_count)||request_valid_d
        ||dot_valid||qv0||qv1||qv2||result_valid_q||final_valid_q
        ||group_valid_q||out_valid;

`ifndef SYNTHESIS
    reg[4:0]sim_expected_pixel;reg[4:0]sim_expected_channel;
    always@(posedge clk)begin
        if(!rst_n||frame_start)begin
            sim_expected_pixel<=0;sim_expected_channel<=0;
        end else if(in_valid)begin
            if(in_pixel_addr[3:0]!=sim_expected_pixel[3:0])
                $error("classifier input pixels are not monotonic");
            if(in_channel_base!=sim_expected_channel)
                $error("classifier input channels are not 0,8,16,24");
            if(in_channel_base==24)begin
                sim_expected_pixel<=sim_expected_pixel+1'b1;
                sim_expected_channel<=0;
            end else sim_expected_channel<=sim_expected_channel+8;
        end
    end
`endif
endmodule

`default_nettype wire
