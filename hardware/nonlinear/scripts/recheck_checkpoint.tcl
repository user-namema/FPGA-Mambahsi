# Fast read-only timing re-analysis of the old physical implementation.
# Does NOT synthesize, optimize, place, route, or overwrite any input DCP.
# New-session Tcl example:
# set nl_dcp E:/nl_compare/clocked_N3_blk0_spa_21736_24087/post_route.dcp
# set nl_out E:/nl_compare/recheck_N3_v2_01
# source .../scripts/recheck_checkpoint.tcl
if {[llength [get_projects -quiet]]} {error "Use a NEW Vivado session"}
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir interface_timing.tcl]
source [file join $script_dir report_split_timing.tcl]
if {![info exists nl_dcp] || ![file isfile $nl_dcp]} {error "Set nl_dcp to an existing routed clocked DCP"}
if {![info exists nl_out]} {error "Set nl_out to a NEW report directory"}
set nl_dcp [file normalize $nl_dcp]
set nl_out [file normalize $nl_out]
if {[file exists $nl_out]} {error "Output exists; preserving $nl_out"}
file mkdir $nl_out
set_param general.maxThreads 2
open_checkpoint $nl_dcp
set f [open [file join $nl_out recheck_metadata.txt] w]
puts $f "run_kind=read_only_reconstraint_no_reroute\ninput_checkpoint=$nl_dcp\nvivado=[version -short]"
close $f
# Keep the old reports so the constraint-only delta is directly inspectable.
report_timing_summary -delay_type min_max -report_unconstrained \
    -file [file join $nl_out original_constraints_timing.rpt]
nl_apply_interface_timing $nl_out
nl_report_split_timing $nl_out
report_route_status -file [file join $nl_out route_status.rpt]
set f [open [file join $nl_out timing_scope_status.txt] r]
puts [read $f]
close $f
puts "RECHECK COMPLETE. Old placement/routing is unchanged; original DCP was NOT written."
puts "Exit code 0 means reports generated, NOT that setup/hold/coverage passed."
