# UP D1 input, quantization and PCIe protocol

## Model and input identity

The hardware parameters implement one UP D1 checkpoint. Its exported input and
integer-reference labels have the following fingerprints:

| Item | SHA256 |
|---|---|
| Training checkpoint | `d113c6123eb3234ea2a6b8fe13640c02b82e235e469e4337ec01c291844daf33` |
| Full-scene input | `5e830f01222de08bbffd59d391f73d8130972deee529c75f7506071edf7c752f` |
| Compact integer-reference label bytes | `28f8b8f34b34f2f58ba67a2fe1fa0de20fd3fae80fb8209a5ebb6e44e5481354` |
| Bitstream used for the board result | `ac91f07dbc13bd6d7000c54e0a8104880f6b2fdb8b9d7e6ac08494634a3732f1` |

The scene is [scene_tiles_uint8.bin](../models/up_d1/scene/scene_tiles_uint8.bin).
The integer label array is
[n3_prediction_rtl_integer.npy](../../evidence/completion_20260923/nonlinear_D1/n3_prediction_rtl_integer.npy).
The label fingerprint above is computed from the compact row-major UINT8 values,
excluding the NPY header. Vivado generates a bitstream for each implementation
run; placement and routing can change its byte-level hash.

## Input layout

The FPGA receives PCA/preprocessed, quantized **UINT8** samples with 16 channels:

```text
scene raster tiles: tile_y, tile_x
within a tile:      y=0..15, x=0..15, channel=0..15
linear byte index: ((y * 16 + x) * 16 + channel)
Patch word[8*c +: 8] = channel c of one pixel
```

One tile is 4096 bytes, transferred as 256 × 128-bit words. UP has 610×340
valid pixels and uses 22 tile columns × 39 tile rows = 858 padded tiles
(3,514,368 bytes). Each boundary tile retains the complete 4096-byte layout
and the padding in the exported scene. The MEM regression vectors contain the
first 16 different tiles and their method-specific expected logits.

## Two-bank ownership and address map

| AXI-MM address | Function |
|---|---|
| `0x00000000..0x0000FFFF` | Input bank A, 16 tiles |
| `0x00010000..0x0001FFFF` | Input bank B, 16 tiles |
| `0x00020000..0x00020FFF` | Current 4096-byte DDR label readback page |

The application exposes a 4-KiB C2H readback page within the BD address decode
range. Host control uses the XDMA user interface and GPIO at AXI-Lite base
`0x40000000`. [send_scene_v2.py](../host/send_scene_v2.py) implements the E2
control protocol.

The host fills and commits both banks before GO. Each bank returns to host
ownership after the loader consumes its committed tiles, allowing the next batch
to be written while the other bank is active. Continuous input sustains a
4096-cycle tile initiation interval; host scheduling and DMA delays can increase
the interval. `--require-ii4096` checks the FPGA interval counters.
`--verify-source` adds input readback traffic for diagnostics.

## Network output and post-processing

Each tile emits 4×4×9 signed INT8 logits (144 bytes). Integer bilinear
interpolation produces a 16×16 score grid before argmax. For p=0..15:

```text
base = min(p // 5, 2)
fraction = p - 5 * base
```

The two-dimensional interpolation weights sum to 25. Scores remain in the wide
integer domain through interpolation, without division, rounding or saturation.
Argmax updates on a strictly larger score, so ties select the lowest class ID.
Padded scene borders are cropped after tile reconstruction.

DDR stores class IDs 0..8 with a 384-byte row pitch. The host removes row padding
to return 610×340 = 207400 bytes. The supplied integer reference uses this exact
post-processing. The floating-point interpolation reference differs at 437
pixels because of near-ties.

Palette for IDs 0 through 8:

```text
FF0000, 00FF00, 0000FF, FFFF00, FF00FF, 00FFFF, C86400, 00C864, 6400C8
```

## D branch and parameter export

The D path adds `D × U` at SSM readout. C reduction and D are aligned, added
in the wide integer domain, and requantized once. Absolute sequence counters
are 32-bit; physical RAM addresses use the widths required by their memory depths.

Parameters are initialized at build time from the exported ROM images in
[`models/up_d1`](../models/up_d1). The training checkpoint is used to produce
parameter exports. A different checkpoint requires a
consistent export of projection, convolution, fusion and head weights; scales;
A/K ROM; folded D coefficients; and the head input contract. The host interface
transfers input tiles and labels; it has no runtime weight-loading command.
