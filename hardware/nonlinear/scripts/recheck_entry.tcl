if {[llength $argv] != 2} {error "Expected: routed_checkpoint NEW_report_directory"}
lassign $argv nl_dcp nl_out
source [file join [file dirname [file normalize [info script]]] recheck_checkpoint.tcl]
