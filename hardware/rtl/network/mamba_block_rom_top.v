`timescale 1ns / 1ps
`default_nettype none

// Reusable BothMamba block.  Patch embedding and inter-block pooling stay
// outside this module.  All learned tensors are selected by BLOCK_ID ROMs;
// requant constants are compile-time parameters.
module mamba_block_rom_top #(
    parameter integer BLOCK_ID=0,
    parameter integer PIXEL_COUNT=(BLOCK_ID==0)?256:(BLOCK_ID==1)?64:16,
    parameter signed[15:0] SPA_IN_M=(BLOCK_ID==0)?16'sd18546:(BLOCK_ID==1)?16'sd25243:16'sd26470,
    parameter signed[6:0] SPA_IN_S=(BLOCK_ID==0)?7'sd20:(BLOCK_ID==1)?7'sd20:7'sd20,
    parameter signed[15:0] SPE_IN_M=(BLOCK_ID==0)?16'sd22145:(BLOCK_ID==1)?16'sd25227:16'sd19624,
    parameter signed[6:0] SPE_IN_S=(BLOCK_ID==0)?7'sd19:(BLOCK_ID==1)?7'sd19:7'sd18,
    parameter signed[15:0] SPA_CONV_M=(BLOCK_ID==0)?16'sd16653:(BLOCK_ID==1)?16'sd21357:16'sd26570,
    parameter signed[6:0] SPA_CONV_S=(BLOCK_ID==0)?7'sd17:(BLOCK_ID==1)?7'sd17:7'sd17,
    parameter signed[15:0] SPE_CONV_M=(BLOCK_ID==0)?16'sd26479:(BLOCK_ID==1)?16'sd31982:16'sd30685,
    parameter signed[6:0] SPE_CONV_S=(BLOCK_ID==0)?7'sd18:(BLOCK_ID==1)?7'sd17:7'sd17,
    parameter signed[15:0] SPA_X_DT_M=(BLOCK_ID==0)?16'sd19739:(BLOCK_ID==1)?16'sd29845:16'sd29343,
    parameter signed[6:0] SPA_X_DT_S=(BLOCK_ID==0)?7'sd18:(BLOCK_ID==1)?7'sd18:7'sd18,
    parameter signed[15:0] SPA_X_B_M=(BLOCK_ID==0)?16'sd25960:(BLOCK_ID==1)?16'sd32617:16'sd20506,
    parameter signed[6:0] SPA_X_B_S=(BLOCK_ID==0)?7'sd22:(BLOCK_ID==1)?7'sd22:7'sd21,
    parameter signed[15:0] SPA_X_C_M=(BLOCK_ID==0)?16'sd23270:(BLOCK_ID==1)?16'sd16419:16'sd22337,
    parameter signed[6:0] SPA_X_C_S=(BLOCK_ID==0)?7'sd22:(BLOCK_ID==1)?7'sd21:7'sd21,
    parameter signed[15:0] SPE_X_DT_M=(BLOCK_ID==0)?16'sd22135:(BLOCK_ID==1)?16'sd22977:16'sd30831,
    parameter signed[6:0] SPE_X_DT_S=(BLOCK_ID==0)?7'sd16:(BLOCK_ID==1)?7'sd17:7'sd16,
    parameter signed[15:0] SPE_X_B_M=(BLOCK_ID==0)?16'sd16799:(BLOCK_ID==1)?16'sd19420:16'sd28707,
    parameter signed[6:0] SPE_X_B_S=(BLOCK_ID==0)?7'sd19:(BLOCK_ID==1)?7'sd20:7'sd20,
    parameter signed[15:0] SPE_X_C_M=(BLOCK_ID==0)?16'sd28221:(BLOCK_ID==1)?16'sd17749:16'sd29510,
    parameter signed[6:0] SPE_X_C_S=(BLOCK_ID==0)?7'sd20:(BLOCK_ID==1)?7'sd20:7'sd20,
    parameter signed[15:0] SPA_DT_M=(BLOCK_ID==0)?16'sd17223:(BLOCK_ID==1)?16'sd27587:16'sd21797,
    parameter signed[6:0] SPA_DT_S=(BLOCK_ID==0)?7'sd22:(BLOCK_ID==1)?7'sd22:7'sd21,
    parameter signed[15:0] SPE_DT_M=(BLOCK_ID==0)?16'sd27745:(BLOCK_ID==1)?16'sd31109:16'sd29056,
    parameter signed[6:0] SPE_DT_S=(BLOCK_ID==0)?7'sd23:(BLOCK_ID==1)?7'sd22:7'sd22
)(
    input wire clk,input wire rst_n,input wire frame_start,
    input wire in_valid,input wire[7:0]in_pixel_addr,
    input wire[4:0]in_channel_base,input wire[((BLOCK_ID==0)?32:40)-1:0]in_data,
    output wire out_valid,output wire[7:0]out_pixel_addr,
    output wire[4:0]out_channel_base,output wire[63:0]out_data,
    output wire done
);
    // Keep the global frame pulse off the large block datapath.  These
    // equivalent registers are intentionally retained as separate physical
    // roots so placement can keep each copy close to its consumer cluster.
    // Payload registers are not reset by these pulses; only control state is.
    (* KEEP = "TRUE", MAX_FANOUT = 64 *) reg block_frame_start_q;
    (* KEEP = "TRUE", MAX_FANOUT = 64 *) reg spa_conv_frame_start_q;
    (* KEEP = "TRUE", MAX_FANOUT = 64 *) reg spe_conv_frame_start_q;
    always @(posedge clk) begin
        if (!rst_n) begin
            block_frame_start_q <= 1'b0;
            spa_conv_frame_start_q <= 1'b0;
            spe_conv_frame_start_q <= 1'b0;
        end else begin
            block_frame_start_q <= frame_start;
            spa_conv_frame_start_q <= frame_start;
            spe_conv_frame_start_q <= frame_start;
        end
    end

    wire si_v,pi_v;wire[7:0]si_p,pi_p;wire[5:0]si_c;wire[3:0]pi_c;
    wire[1:0]pi_t;wire[63:0]si_d,pi_d;wire si_done,pi_done;
    wire sx_record_credit_ready;
    wire ssm_record_credit_ready;
    wire in_record_launch;
    wire sx_input_record_consumed;
    wire sx_output_record_launch;
    wire ssm_record_consumed;
    reg [2:0] block_record_credits;
    reg [2:0] ssm_record_credits;
    wire block_credit_available=(block_record_credits!=0)
        ||sx_input_record_consumed;
    wire ssm_credit_available=(ssm_record_credits!=0)
        ||ssm_record_consumed;
    wire block_record_credit_ready = (BLOCK_ID==0)
        ? (sx_record_credit_ready&&block_credit_available) : 1'b1;
    // Block0's fused SSM owns a four-record ingress FIFO.  The scaled Block1
    // and Block2 cores already write each arriving record into their BRAM
    // watermarks, so their x_proj may launch as soon as its local result slot
    // is available and does not need a duplicate SSM credit counter.
    wire spa_x_downstream_ready = (BLOCK_ID==0)
        ? (ssm_credit_available&&ssm_record_credit_ready) : 1'b1;
    always @(posedge clk) begin
        if(!rst_n||block_frame_start_q)begin
            block_record_credits<=3'd4;
            ssm_record_credits<=3'd4;
        end
        else begin
            if(BLOCK_ID==0)begin
                case({in_record_launch,sx_input_record_consumed})
                    2'b10: block_record_credits<=block_record_credits-1'b1;
                    2'b01: block_record_credits<=block_record_credits+1'b1;
                    default: block_record_credits<=block_record_credits;
                endcase
                case({sx_output_record_launch,ssm_record_consumed})
                    2'b10: ssm_record_credits<=ssm_record_credits-1'b1;
                    2'b01: ssm_record_credits<=ssm_record_credits+1'b1;
                    default: ssm_record_credits<=ssm_record_credits;
                endcase
            end
        end
    end
`ifndef SYNTHESIS
    // The two counters protect different latency domains.  Keep these checks
    // close to the counters so a future interface edit cannot silently merge
    // them again or wrap a three-bit credit count.
    always @(posedge clk) begin
        if(rst_n&&!block_frame_start_q)begin
            if((BLOCK_ID==0)&&in_record_launch&&(block_record_credits==0)
                    &&!sx_input_record_consumed)
                $fatal(1,"Spa x_proj credit underflow");
            if((BLOCK_ID==0)&&sx_input_record_consumed&&(block_record_credits==4)
                    &&!in_record_launch)
                $fatal(1,"Spa x_proj credit overflow");
            if((BLOCK_ID==0)&&sx_output_record_launch&&(ssm_record_credits==0)
                    &&!ssm_record_consumed)
                $fatal(1,"Block0 SSM credit underflow");
            if((BLOCK_ID==0)&&ssm_record_consumed&&(ssm_record_credits==4)
                    &&!sx_output_record_launch)
                $fatal(1,"Block0 SSM credit overflow");
        end
    end
`endif
    both_in_proj_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),
        .SPA_REQUANT_MULTIPLIER(SPA_IN_M),.SPA_REQUANT_SHIFT(SPA_IN_S),
        .SPE_REQUANT_MULTIPLIER(SPE_IN_M),.SPE_REQUANT_SHIFT(SPE_IN_S)) u_in(
        .clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),
        .record_credit_ready(block_record_credit_ready),
        .record_launch(in_record_launch),
        .patch_valid(in_valid),.patch_pixel_addr(in_pixel_addr),
        .patch_channel_base(in_channel_base),.patch_data(in_data),
        .spa_weight_we(1'b0),.spa_weight_addr(11'd0),.spa_weight_data(8'd0),
        .spa_requant_multiplier(32'd0),.spa_requant_shift(7'd0),
        .spe_weight_we(1'b0),.spe_weight_addr(7'd0),.spe_weight_data(8'd0),
        .spe_requant_multiplier(32'd0),.spe_requant_shift(7'd0),
        .spa_out_valid(si_v),.spa_out_pixel_addr(si_p),.spa_out_channel_base(si_c),
        .spa_out_data(si_d),.spa_done(si_done),.spe_out_valid(pi_v),
        .spe_out_pixel_addr(pi_p),.spe_out_token(pi_t),.spe_out_channel_base(pi_c),
        .spe_out_data(pi_d),.spe_done(pi_done));

    wire sc_v,pc_v;wire[7:0]sc_p,pc_p;wire[5:0]sc_c;wire[3:0]pc_c;
    wire[1:0]pc_t;wire[63:0]sc_raw,sc_d,pc_raw,pc_d;wire sc_done,pc_done;
    both_conv1d_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),
        .SPA_REQUANT_MULTIPLIER(SPA_CONV_M),.SPA_REQUANT_SHIFT(SPA_CONV_S),
        .SPE_REQUANT_MULTIPLIER(SPE_CONV_M),.SPE_REQUANT_SHIFT(SPE_CONV_S)) u_conv(
        .clk(clk),.rst_n(rst_n),
        .spa_frame_start(spa_conv_frame_start_q),
        .spe_frame_start(spe_conv_frame_start_q),
        .spa_in_valid(si_v),.spa_in_pixel_addr(si_p),.spa_in_channel_base(si_c),.spa_in_data(si_d),
        .spe_in_valid(pi_v),.spe_in_pixel_addr(pi_p),.spe_in_token(pi_t),.spe_in_channel_base(pi_c),.spe_in_data(pi_d),
        .spa_weight_we(1'b0),.spa_weight_addr(8'd0),.spa_weight_data(8'd0),
        .spa_bias_we(1'b0),.spa_bias_addr(6'd0),.spa_bias_data(32'd0),
        .spa_requant_multiplier(32'd0),.spa_requant_shift(7'd0),
        .spe_weight_we(1'b0),.spe_weight_addr(6'd0),.spe_weight_data(8'd0),
        .spe_bias_we(1'b0),.spe_bias_addr(4'd0),.spe_bias_data(32'd0),
        .spe_requant_multiplier(32'd0),.spe_requant_shift(7'd0),
        .spa_out_valid(sc_v),.spa_out_pixel_addr(sc_p),.spa_out_channel_base(sc_c),
        .spa_out_raw_data(sc_raw),.spa_out_data(sc_d),.spa_done(sc_done),
        .spe_out_valid(pc_v),.spe_out_pixel_addr(pc_p),.spe_out_token(pc_t),
        .spe_out_channel_base(pc_c),.spe_out_raw_data(pc_raw),.spe_out_data(pc_d),.spe_done(pc_done));

    wire sxdt_v,sb_v,scv_v,sxu_v;wire[7:0]sxdt_p,sb_p,scv_p,sxu_p;wire[17:0]sxdt_d;
    wire[5:0]sxu_c;wire[63:0]sxu_d;
    wire[127:0]sb_d,scv_d;wire sx_done;
    spa_x_proj_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(SPA_X_DT_M),
        .DT_SHIFT(SPA_X_DT_S),.B_MULTIPLIER(SPA_X_B_M),.B_SHIFT(SPA_X_B_S),
        .C_MULTIPLIER(SPA_X_C_M),.C_SHIFT(SPA_X_C_S)) u_sx(
        .clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),
        .downstream_record_ready(spa_x_downstream_ready),.in_valid(sc_v),
        .in_pixel_addr(sc_p),.in_channel_base(sc_c),.in_data(sc_d),
        .cfg_weight_we(1'b0),.cfg_weight_addr(12'd0),.cfg_weight_data(8'd0),
        .cfg_dt_multiplier(32'd0),.cfg_dt_shift(7'd0),.cfg_b_multiplier(32'd0),
        .cfg_b_shift(7'd0),.cfg_c_multiplier(32'd0),.cfg_c_shift(7'd0),
        .dt_valid(sxdt_v),.dt_pixel_addr(sxdt_p),.dt_data(sxdt_d),
        .b_valid(sb_v),.b_pixel_addr(sb_p),.b_data(sb_d),
        .c_valid(scv_v),.c_pixel_addr(scv_p),.c_data(scv_d),
        .u_valid(sxu_v),.u_pixel_addr(sxu_p),.u_channel_base(sxu_c),.u_data(sxu_d),
        .record_credit_ready(sx_record_credit_ready),
        .input_record_consumed(sx_input_record_consumed),
        .output_record_launch(sx_output_record_launch),.done(sx_done));

    wire pxdt_v,pb_v,pcv_v;wire[7:0]pxdt_p,pb_p,pcv_p;wire[1:0]pxdt_t,pb_t,pcv_t;
    wire[8:0]pxdt_d;wire[127:0]pb_d,pcv_d;wire px_done;
    spe_x_proj_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.DT_MULTIPLIER(SPE_X_DT_M),
        .DT_SHIFT(SPE_X_DT_S),.B_MULTIPLIER(SPE_X_B_M),.B_SHIFT(SPE_X_B_S),
        .C_MULTIPLIER(SPE_X_C_M),.C_SHIFT(SPE_X_C_S)) u_px(
        .clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),.in_valid(pc_v),
        .in_pixel_addr(pc_p),.in_token(pc_t),.in_channel_base(pc_c),.in_data(pc_d),
        .cfg_weight_we(1'b0),.cfg_weight_addr(10'd0),.cfg_weight_data(8'd0),
        .cfg_dt_multiplier(32'd0),.cfg_dt_shift(7'd0),.cfg_b_multiplier(32'd0),
        .cfg_b_shift(7'd0),.cfg_c_multiplier(32'd0),.cfg_c_shift(7'd0),
        .dt_valid(pxdt_v),.dt_pixel_addr(pxdt_p),.dt_token(pxdt_t),.dt_data(pxdt_d),
        .b_valid(pb_v),.b_pixel_addr(pb_p),.b_token(pb_t),.b_data(pb_d),
        .c_valid(pcv_v),.c_pixel_addr(pcv_p),.c_token(pcv_t),.c_data(pcv_d),.done(px_done));

    wire sdt_v,pdt_v;wire[7:0]sdt_p,pdt_p;wire[5:0]sdt_c;wire[3:0]pdt_c;
    wire[1:0]pdt_t;wire[63:0]sdt_d,pdt_d;wire sdt_done,pdt_done;
    spa_dt_proj_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.REQUANT_MULTIPLIER(SPA_DT_M),
        .REQUANT_SHIFT(SPA_DT_S)) u_sdt(.clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),
        .in_valid(sxdt_v),.in_pixel_addr(sxdt_p),.in_data(sxdt_d),
        .cfg_weight_we(1'b0),.cfg_weight_addr(7'd0),.cfg_weight_data(8'd0),
        .cfg_bias_we(1'b0),.cfg_bias_addr(6'd0),.cfg_bias_data(32'd0),
        .cfg_multiplier(32'd0),.cfg_shift(7'd0),.out_valid(sdt_v),
        .out_pixel_addr(sdt_p),.out_channel_base(sdt_c),.out_data(sdt_d),.done(sdt_done));
    spe_dt_proj_stream #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT),.REQUANT_MULTIPLIER(SPE_DT_M),
        .REQUANT_SHIFT(SPE_DT_S)) u_pdt(.clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),
        .in_valid(pxdt_v),.in_pixel_addr(pxdt_p),.in_token(pxdt_t),.in_data(pxdt_d),
        .cfg_weight_we(1'b0),.cfg_weight_addr(4'd0),.cfg_weight_data(8'd0),
        .cfg_bias_we(1'b0),.cfg_bias_addr(4'd0),.cfg_bias_data(32'd0),
        .cfg_multiplier(32'd0),.cfg_shift(7'd0),.out_valid(pdt_v),
        .out_pixel_addr(pdt_p),.out_token(pdt_t),.out_channel_base(pdt_c),
        .out_data(pdt_d),.done(pdt_done));

    // Block0's STEAM x_proj delays U in the same four-record credit domain as
    // DT/B/C.  The area-scaled Block1/2 x_proj does not duplicate U storage;
    // their SSM ingress BRAM accepts the earlier Conv stream independently.
    wire tail_spa_u_valid=(BLOCK_ID==0)?sxu_v:sc_v;
    wire[7:0]tail_spa_u_pixel=(BLOCK_ID==0)?sxu_p:sc_p;
    wire[5:0]tail_spa_u_channel=(BLOCK_ID==0)?sxu_c:sc_c;
    wire[63:0]tail_spa_u_data=(BLOCK_ID==0)?sxu_d:sc_d;
    wire tail_spa_u_done=(BLOCK_ID==0)?sx_done:sc_done;

    both_mamba_block0_complete #(.BLOCK_ID(BLOCK_ID),.PIXEL_COUNT(PIXEL_COUNT)) u_tail(
        .clk(clk),.rst_n(rst_n),.frame_start(block_frame_start_q),
        .block_input_valid(in_valid),.block_input_pixel_addr(in_pixel_addr),
        .block_input_channel_base(in_channel_base),.block_input_data(in_data),
        .spa_u_valid(tail_spa_u_valid),.spa_u_pixel_addr(tail_spa_u_pixel),
        .spa_u_channel_base(tail_spa_u_channel),.spa_u_data(tail_spa_u_data),
        .spa_u_done(tail_spa_u_done),
        .spa_dt_valid(sdt_v),.spa_dt_pixel_addr(sdt_p),.spa_dt_channel_base(sdt_c),.spa_dt_data(sdt_d),.spa_dt_done(sdt_done),
        .spa_b_valid(sb_v),.spa_b_pixel_addr(sb_p),.spa_b_data(sb_d),.spa_c_valid(scv_v),.spa_c_pixel_addr(scv_p),.spa_c_data(scv_d),.spa_x_done(sx_done),
        .spe_u_valid(pc_v),.spe_u_pixel_addr(pc_p),.spe_u_token(pc_t),.spe_u_channel_base(pc_c),.spe_u_data(pc_d),.spe_u_done(pc_done),
        .spe_dt_valid(pdt_v),.spe_dt_pixel_addr(pdt_p),.spe_dt_token(pdt_t),.spe_dt_channel_base(pdt_c),.spe_dt_data(pdt_d),.spe_dt_done(pdt_done),
        .spe_b_valid(pb_v),.spe_b_pixel_addr(pb_p),.spe_b_token(pb_t),.spe_b_data(pb_d),
        .spe_c_valid(pcv_v),.spe_c_pixel_addr(pcv_p),.spe_c_token(pcv_t),.spe_c_data(pcv_d),.spe_x_done(px_done),
        .spa_cfg_lut_we(1'b0),.spa_cfg_lut_addr(8'd0),.spa_cfg_lut_data(419'd0),
        .spe_cfg_lut_we(1'b0),.spe_cfg_lut_addr(8'd0),.spe_cfg_lut_data(419'd0),
        .cfg_spa_ssm_multiplier(16'd0),.cfg_spa_ssm_shift(7'd0),.cfg_spe_ssm_multiplier(16'd0),.cfg_spe_ssm_shift(7'd0),
        .cfg_spa_out_weight_we(1'b0),.cfg_spa_out_weight_addr(11'd0),.cfg_spa_out_weight_data(8'd0),.cfg_spa_out_bias_we(1'b0),.cfg_spa_out_bias_addr(5'd0),.cfg_spa_out_bias_data(32'd0),.cfg_spa_out_multiplier(16'd0),.cfg_spa_out_shift(7'd0),
        .cfg_spe_out_weight_we(1'b0),.cfg_spe_out_weight_addr(7'd0),.cfg_spe_out_weight_data(8'd0),.cfg_spe_out_bias_we(1'b0),.cfg_spe_out_bias_addr(3'd0),.cfg_spe_out_bias_data(32'd0),.cfg_spe_out_multiplier(16'd0),.cfg_spe_out_shift(7'd0),
        .cfg_fusion_a_multiplier(16'd0),.cfg_fusion_a_shift(7'd0),.cfg_fusion_b_multiplier(16'd0),.cfg_fusion_b_shift(7'd0),
        .cfg_block_a_multiplier(16'd0),.cfg_block_a_shift(7'd0),.cfg_block_b_multiplier(16'd0),.cfg_block_b_shift(7'd0),
        .out_valid(out_valid),.out_pixel_addr(out_pixel_addr),.out_channel_base(out_channel_base),.out_data(out_data),
        .spa_input_record_ready(ssm_record_credit_ready),
        .spa_input_record_consumed(ssm_record_consumed),.done(done));
endmodule

`default_nettype wire
