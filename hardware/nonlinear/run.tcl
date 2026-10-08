# Vivado batch: -tclargs sim|d1sim|build N0|N1|N2|N3 blk0_spa NEW_DIR 0|1
if {[llength $argv]!=5} {error "Expected action method core new_output_dir with_D_readout"}
lassign $argv action nl_method nl_core nl_out nl_with_d1
if {$nl_with_d1 ni {0 1}} {error "with_D_readout must be 0 or 1"}
set root [file dirname [file normalize [info script]]]
set nl_clocked 1
switch $action {
    sim {source [file join $root scripts run_sim.tcl]}
    d1sim {source [file join $root scripts run_d1_sim.tcl]}
    build {source [file join $root scripts build_ooc.tcl]}
    default {error "Unknown action"}
}
