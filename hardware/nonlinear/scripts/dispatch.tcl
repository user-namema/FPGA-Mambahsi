# Batch entry: <sim|d-sim|build> <N0|N1|N2|N3> <core> <0|1> new_out
if {$argc != 5} {error {Expected action method core with_D [new_out]}}
lassign $argv action nl_method nl_core nl_with_d1 nl_out
if {$action ni {sim d-sim build}} {error "Unknown action $action"}
if {$nl_method ni {N0 N1 N2 N3}} {error "Unknown method $nl_method"}
if {$nl_with_d1 ni {0 1}} {error "with_D must be 0 or 1"}
if {$nl_out eq ""} {error "new_out must name a new short ASCII output directory"}
set entry [dict get [dict create sim run_sim.tcl d-sim run_d1_sim.tcl build build_ooc.tcl] $action]
source [file join [file dirname [file normalize [info script]]] $entry]
