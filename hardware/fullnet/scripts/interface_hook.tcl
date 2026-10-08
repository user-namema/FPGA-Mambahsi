# Common *assumed fabric* interface. No board pin/setup/hold claim.
set inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
set outputs [get_ports -filter {DIRECTION == OUT}]
set ref [get_pins -quiet rst_q_reg/C]
if {[llength $ref]!=1} {error "Missing registered fabric clock reference rst_q_reg/C"}
set cl [get_clocks core_clk]
set_input_delay -clock $cl -reference_pin $ref -min 0.200 $inputs
set_input_delay -clock $cl -reference_pin $ref -max 1.000 $inputs
set_output_delay -clock $cl -reference_pin $ref -min -0.100 $outputs
set_output_delay -clock $cl -reference_pin $ref -max 1.000 $outputs
# No false paths, no multicycle exceptions, no invented clock latency.
