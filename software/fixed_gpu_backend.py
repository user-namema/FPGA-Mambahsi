"""Hardware-contract GPU simulation, not native INT8/TensorRT inference.

No floating QAT forward is executed by forward_codes. MACs use exact FP64
integer arithmetic, with the same frozen scales and integer requantization as
the FPGA reference. Each batch lane is an independent tile/SSM state.
"""
import contextlib
import io
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import both_FPGA_single_qat_source as hw
from ssm_error_ablation import SSMNumericConfig


def update_wide_gpu(h, a, kbu, fraction, c):
    """Exact wide sum without materializing a potentially 65-bit numerator.

    Arithmetic right shifts give floor quotient and nonnegative remainder.
    Sum the two quotients/remainders, then implement signed half-away rounding.
    This keeps every tensor on its original device. Bounds are proved at setup.
    """
    f = c.a_fraction_bits
    shift = f + c.state_fraction_bits - fraction
    pa = a * h
    qa, ra = pa >> f, pa & ((1 << f) - 1)
    if shift >= f:
        qu, ru = kbu << (shift - f), torch.zeros_like(kbu)
    else:
        n = f - shift
        qu, ru = kbu >> n, (kbu & ((1 << n) - 1)) << shift
    half = 1 << (f - 1)
    def rounded(q, r):
        return q + ((r > half) | ((r == half) & (q >= 0))).to(torch.int64)
    if c.rounding == 'single':
        rem = ra + ru
        q = qa + qu + (rem >> f)
        raw = rounded(q, rem & ((1 << f) - 1))
    else:
        raw = rounded(qa, ra) + rounded(qu, ru)
    return raw.clamp(c.state_min, c.state_max)


def certify_wide(tables, c):
    f = tables['k_fraction_bits']
    shift = c.a_fraction_bits + c.state_fraction_bits - f
    if not 0 <= shift <= 62:
        raise ValueError('Unsupported common shift for GPU wide decomposition: %s' % shift)
    pa = int(tables['a'].abs().max()) * (1 << (c.state_bits-1))
    kbu = int(tables['k'].abs().max()) * 128 * 128
    quotient = (pa >> c.a_fraction_bits) + ((kbu << shift) >> c.a_fraction_bits) + 4
    if max(pa, kbu, quotient) >= (1 << 63) - 1:
        raise OverflowError('GPU quotient/remainder INT64 container is insufficient')
    return dict(algorithm='floor quotient/remainder, signed half-away',
                pa_abs_bound=pa, kbu_abs_bound=kbu, quotient_abs_bound=quotient,
                common_left_shift=shift, cpu_state_fallback=False)


class FixedGPUModel(nn.Module):
    def __init__(self, net, config):
        super().__init__()
        self.net = net.eval()
        expected = dict(branch_mode='both', fusion_mode='sum', A_mode='shared',
                        use_D=True, use_z=False, activation='relu', norm_path='bn',
                        hidden_dim=32, token_num=4, d_state=16, head_dim=64, skip_scale=2)
        if any(config.get(k) != v for k,v in expected.items()):
            raise ValueError('Fixed GPU backend requires current both/sum/shared-A/D1/ReLU/BN architecture')
        if len(net.mamba) != 5:
            raise ValueError("Expected three Mamba blocks and two pooling layers")
        self.device = next(net.parameters()).device
        self.input_scale = hw.get_activation_scale(net.patch_embedding[0], 1./255, self.device)
        self.layers, self.cores = {}, {}
        self.parity_report = None
        self._benchmark_config = config
        self._benchmark_scan_backend = 'integer_lut_wide_gpu_reference'
        self.numeric_configs = {}

    def act(self, module, fallback):
        return hw.get_activation_scale(module, fallback, self.device)

    def boundary(self, module, fallback):
        return hw.get_boundary_scale(module, fallback, self.device)

    def layer(self, x, s_in, module, s_out, name, bits=8):
        if name not in self.layers:
            sw = hw.scalar_scale(module.lsq_w.s.detach(), self.device).abs().clamp_min(1e-12)
            si = hw.scalar_scale(s_in, self.device).abs().clamp_min(1e-12)
            so = hw.scalar_scale(s_out, self.device).abs().clamp_min(1e-12)
            wb = int(module.nbit_w)
            w = hw.round_half_away_from_zero(module.weight.detach()/sw).clamp(-(1 << (wb-1)), (1 << (wb-1))-1)
            reduction = module.in_features if isinstance(module, nn.Linear) else (module.in_channels//module.groups)*math.prod(module.kernel_size)
            # Include UINT8 input and signed INT10 pool output; all other inputs
            # are constrained by explicit INT8/INT9 boundaries.
            input_bound = max(512, 1 << (getattr(module, 'nbit_a', 8)-1))
            mac_bound = input_bound * int(w.abs().max()) * reduction
            bias = None
            if module.bias is not None:
                raw_bias = torch.round(module.bias.detach()/(si*sw))
                if not torch.isfinite(raw_bias).all() or torch.any(raw_bias.double().abs() > (1 << 31)-1):
                    raise OverflowError(name+': invalid/out-of-range INT32 bias')
                bias = raw_bias.to(torch.int64)
            bias_bound = 0 if bias is None else int(bias.abs().max())
            if mac_bound + bias_bound > (1 << 31)-1:
                raise OverflowError(name+': conservative MAC+bias bound exceeds INT32')
            mi, sh = hw.get_m_int_shift((si*sw)/so, q_bits=16)
            self.layers[name] = (w.double(), bias, so, mi, sh, mac_bound+bias_bound)
        w, bias, so, mi, sh, bound = self.layers[name]
        xd = x.double()
        if isinstance(module, nn.Conv2d):
            accum = F.conv2d(xd,w,None,module.stride,module.padding,module.dilation,module.groups)
        elif isinstance(module, nn.Conv1d):
            accum = F.conv1d(xd,w,None,module.stride,module.padding,module.dilation,module.groups)
        elif isinstance(module, nn.Linear):
            accum = F.linear(xd,w,None)
        else:
            raise TypeError(type(module))
        accum = torch.round(accum).to(torch.int64)
        if bias is not None:
            accum = accum + (bias.view([1,-1]+[1]*(accum.ndim-2)) if isinstance(module,(nn.Conv1d,nn.Conv2d)) else bias)
        codes = hw.apply_multiplier_shift(accum,mi,sh).clamp(-(1 << (bits-1)), (1 << (bits-1))-1)
        return codes.to(torch.int16 if bits > 8 else torch.int8), so

    def core(self, x, si, m, name, so):
        c = SSMNumericConfig.from_dict(getattr(m,'_ssm_numeric_config',None))
        if c.error_source != 'all' or c.resolved_readout_requantization != 'hardware':
            raise ValueError('Fixed timing requires error_source=all and hardware readout')
        batch, length, _ = x.shape
        sx = self.act(m.conv1d,si)
        x,sx = self.layer(x,si,m.in_proj,sx,name+'_in_proj')
        su = self.act(m.x_proj,sx)
        x,su = self.layer(x.transpose(1,2).contiguous(),sx,m.conv1d,su,name+'_conv1d')
        u = F.relu(x[:,:,:length])
        xp = u.transpose(1,2).contiguous()
        sd = self.act(m.dt_proj,su)
        sb = self.boundary(m.b_output_quant,su)
        sc = self.boundary(m.c_output_quant,su)
        xd,_ = self.layer(xp,su,m.x_proj,sd,name+'_x_proj_dt',c.dt_input_bits)
        xb,_ = self.layer(xp,su,m.x_proj,sb,name+'_x_proj_B')
        xc,_ = self.layer(xp,su,m.x_proj,sc,name+'_x_proj_C')
        dt,sdout = self.layer(xd[:,:,:m.dt_rank],sd,m.dt_proj,
                            self.boundary(m.dt_output_quant,sd),name+'_dt_proj',c.dt_output_bits)
        b = xb[:,:,m.dt_rank:m.dt_rank+m.d_state].transpose(1,2).long()
        cr = xc[:,:,m.dt_rank+m.d_state:].transpose(1,2).long()
        if name not in self.cores:
            hw.get_or_build_ssm_luts(m,sdout,sb,su,name,None,self.device)
            tables = m._fpga_ssm_lut_cache['tables']
            proof = certify_wide(tables,c)
            d = hw.get_or_build_d_path(m,su,sc,c,name,None)
            sy = hw.scalar_scale(sc,self.device)*(2.**-c.state_fraction_bits)
            sout = self.act(m.out_proj_linear,sy)
            mi,sh = hw.get_m_int_shift(sy/sout,q_bits=16)
            dproof = None if d is None else hw.certify_d_requant(d,mi.item(),sh)
            if d is None:
                raise ValueError('D1 model has no D table')
            self.cores[name] = dict(a=tables['a'].to(self.device),k=tables['k'].to(self.device),
                fraction=tables['k_fraction_bits'],d=d['coefficient'].to(self.device),
                dshift=d['left_shift'],sy=sy,sout=sout,proof=proof,
                range_certificate=tables['range_certificate'],d_certificate=dproof,
                d_range_certificate=d['range_certificate'])
            self.numeric_configs[name] = c.to_dict()
        cache=self.cores[name]
        h=torch.zeros((batch,m.d_inner,m.d_state),dtype=torch.int64,device=self.device)
        ui=u.long(); dt=dt.long(); ys=[]
        for t in range(length):
            addr=dt[:,t]-c.dt_min
            kbu=cache['k'][addr].unsqueeze(-1)*b[:,:,t].unsqueeze(1)*ui[:,:,t].unsqueeze(-1)
            h=update_wide_gpu(h,cache['a'][addr],kbu,cache['fraction'],c)
            y=(h*cr[:,:,t].unsqueeze(1)).sum(-1)
            y=y+((cache['d'].unsqueeze(0)*ui[:,:,t]) << cache['dshift'])
            ys.append(y)
        readout=torch.stack(ys,dim=1)
        codes=hw.ssm_readout_to_int8(readout,cache['sy'],cache['sout'],c)
        return self.layer(codes,cache['sout'],m.out_proj_linear,so,name+'_out_proj')

    def block(self,x,si,block,index):
        batch,channels,height,width=x.shape
        results=[]
        for branch in ('spa','spe'):
            wrapper=getattr(block,branch+'_mamba'); m=wrapper.mamba
            inp=self.act(m.in_proj,si)
            codes=hw.requantize_int8(x,si,inp)
            residual=self.boundary(getattr(block,branch+'_residual_quant'),si)
            if branch=='spa':
                seq=codes.permute(0,2,3,1).contiguous().view(batch,height*width,channels)
            else:
                if wrapper.channel_num > channels:
                    codes=torch.cat([codes,torch.zeros((batch,wrapper.channel_num-channels,height,width),dtype=codes.dtype,device=self.device)],dim=1)
                seq=codes.permute(0,2,3,1).contiguous().view(batch*height*width,wrapper.token_num,wrapper.group_channel_num)
            out,scale=self.core(seq,inp,m,'blk%d_%s'%(index,branch),residual)
            out=out.reshape(batch,height,width,-1).permute(0,3,1,2).contiguous()[:,:channels]
            if branch=='spe' or getattr(wrapper,'use_proj',True): out=F.relu(out)
            if wrapper.use_residual: out,scale=hw.align_and_add_int8(x,si,out,scale,s_out=residual)
            results.append((out,scale))
        (spa,ss),(spe,sp)=results
        if block.use_att: raise ValueError('Only current sum fusion supported')
        sf=self.boundary(block.fusion_quant,torch.maximum(ss,sp))
        fusion,sf=hw.align_and_add_int8(spa,ss,spe,sp,s_out=sf)
        so=self.boundary(block.block_output_quant,sf)
        if block.use_residual:
            return hw.align_and_add_int8(x,si*block._fpga_skip_scale,fusion,sf,s_out=so)
        return hw.requantize_int8(fusion,sf,so),so

    @torch.inference_mode()
    def forward_codes(self,tiles):
        if tiles.ndim!=4 or tiles.shape[0]<1 or tuple(tiles.shape[-2:])!=(16,16):
            raise ValueError('Expected independent NCHW 16x16 tiles')
        x=hw.round_half_away_from_zero(tiles/self.input_scale).clamp(0,255).to(torch.int16)
        s=self.boundary(self.net.patch_output_quant,self.input_scale)
        x,s=self.layer(x,self.input_scale,self.net.patch_embedding[0],s,'patch_embed')
        x=F.relu(x).to(torch.int8)
        for index in range(3):
            x,s=self.block(x,s,self.net.mamba[index*2],index)
            if index<2: x,s=hw.simulate_avg_pool_codes(x,s,self.net.mamba[index*2+1],self.device)
        sh=self.act(self.net.cls_head[0],s)
        x=hw.requantize_int8(x,s,sh)
        x,s=self.layer(x,sh,self.net.cls_head[0],self.act(self.net.cls_head[3],sh),'head_conv1')
        x=F.relu(x).to(torch.int8)
        return self.layer(x,s,self.net.cls_head[3],self.boundary(self.net.logits_quant,s),'head_conv2')

    def forward(self,tiles):
        codes,scale=self.forward_codes(tiles)
        return codes.float()*scale

    @torch.inference_mode()
    def verify(self,tiles):
        """Untimed strict parity: independent reference tiles AND batch execution."""
        singles=[]
        with contextlib.redirect_stdout(io.StringIO()):
            for tile in tiles.split(1):
                _,_,expected,es=hw.simulate_tile_dual_int8(self.net,tile,self.input_scale,None,self.device)
                actual,scale=self.forward_codes(tile)
                if not torch.equal(actual,expected) or not torch.equal(scale,es):
                    raise RuntimeError('Fixed GPU integer codes/scale disagree with FPGA reference')
                singles.append(actual)
            batched,scale=self.forward_codes(tiles)
            if not torch.equal(batched,torch.cat(singles)):
                raise RuntimeError('Batch lanes are not independent integer tile results')
        self.parity_report=dict(tiles=len(tiles),integer_codes_exact=True,scale_exact=True,batch_exact=True)
        return self.parity_report

    @torch.inference_mode()
    def verify_batch_lanes(self, tiles):
        batched, scale = self.forward_codes(tiles)
        singles = [self.forward_codes(x)[0] for x in tiles.split(1)]
        if not torch.equal(batched, torch.cat(singles)):
            raise RuntimeError("Actual timing batch differs from independent single-tile codes")
        self.parity_report.setdefault("actual_batch_sizes_checked", []).append(len(tiles))

    def manifest(self):
        return dict(execution_type='hardware-contract simulation: FP64 integer MAC + INT64 fixed-point tensors',
                    native_int8_tensor_cores=False,cpu_state_fallback=False,
                    includes_runtime_range_checks=True,numeric_config=self.numeric_configs,
                    parity=self.parity_report,
                    cores={k:{key:v[key] for key in ('fraction','proof','range_certificate','d_certificate','d_range_certificate')} for k,v in self.cores.items()})
