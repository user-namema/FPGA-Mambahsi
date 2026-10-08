# argv: action core method out_directory
if {[llength $argv]!=4} {error "Expected: sim|build core N0|N1|N2|N3 new_output_dir"}
lassign $argv nl_action nl_core nl_method nl_out
set entry_dir [file dirname [file normalize [info script]]]
if {$nl_action eq "sim"} {
    source [file join $entry_dir run_sim.tcl]
} elseif {$nl_action eq "build"} {
    source [file join $entry_dir build_ooc.tcl]
} else {error "Unknown action $nl_action"}
