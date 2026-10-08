# In a NEW Vivado window:
# set nl_method N0; set nl_core blk0_spa
# set nl_out E:/nl_compare/n0_b0spa_v1
# source .../scripts/build_ooc.tcl
source [file join [file dirname [file normalize [info script]]] common.tcl]
source [file join $nl_root scripts interface_timing.tcl]
source [file join $nl_root scripts report_split_timing.tcl]
if {[llength [get_projects -quiet]]} {error "Use a new Vivado session without an open project"}
if {![info exists nl_method]} {set nl_method N3}
if {![info exists nl_core]} {set nl_core blk0_spa}
if {![info exists nl_clocked]} {set nl_clocked 1}
if {!$nl_clocked} {error "timing_v2 requires the clocked shell; use an old package for bare OOC"}
nl_interface_defaults
if {$nl_method ni {N0 N1 N2 N3}} {error "nl_method must be N0, N1, N2 or N3"}
nl_check_core $nl_core
if {![info exists nl_out]} {
    error "Set nl_out to a NEW short ASCII build directory"
}
set nl_out [nl_new_directory $nl_out]
set nl_previous_pwd [pwd]
cd $nl_out
nl_copy_data $nl_out
set nl_top nl_[string tolower $nl_method]_${nl_core}
if {[info exists nl_clocked] && $nl_clocked} {set nl_top nl_clocked_[string tolower $nl_method]_${nl_core}}
if {![info exists nl_with_d1]} {set nl_with_d1 0}
if {$nl_with_d1} {set nl_top nl_clocked_d1_[string tolower $nl_method]_${nl_core}}
create_project -in_memory -part xcku060-ffva1156-2-i
set_param general.maxThreads 2
set_property include_dirs [list [file join $nl_root rtl]] [current_fileset]
read_verilog -sv [nl_sources]
synth_design -top $nl_top -part xcku060-ffva1156-2-i -mode out_of_context \
    -flatten_hierarchy rebuilt
nl_apply_interface_timing $nl_out
check_timing -verbose -file check_timing_synth.rpt
report_utilization -hierarchical -file utilization_synth.rpt
write_checkpoint post_synth.dcp
opt_design
place_design
phys_opt_design
route_design
write_checkpoint post_route.dcp
report_utilization -file utilization_routed.rpt
report_utilization -hierarchical -file utilization_hierarchical_routed.rpt
nl_report_split_timing $nl_out
report_route_status -file route_status.rpt
report_drc -file drc.rpt
set sf [open run_metadata.txt w]
puts $sf "top=$nl_top\nmethod=$nl_method\ncore=$nl_core\npart=xcku060-ffva1156-2-i\nclock_period_ns=$nl_period_ns"
puts $sf "timing_model=shared_fabric_clock_reference_v2\nrun_kind=rebuilt_and_routed"
puts $sf "scope=D1 OOC coefficient generator; with_common_D_readout=$nl_with_d1; NOT complete SSM/network/board"
puts $sf "D_adapter=17-cycle C alignment plus 1-cycle C+D sum; external C reduction input; separate stream"
puts $sf "vivado=[version -short]\nsource_root=$nl_root"
close $sf
set setup_path [get_timing_paths -delay_type max -max_paths 1]
set hold_path [get_timing_paths -delay_type min -max_paths 1]
set nl_wns [get_property SLACK $setup_path]
set nl_whs [get_property SLACK $hold_path]
puts "OOC COMPLETE: $nl_top WNS=$nl_wns WHS=$nl_whs; reports: $nl_out"
if {$nl_wns < 0 || $nl_whs < 0} {
    puts "WARNING: global setup/hold NOT MET. Inspect split reports; do not report full timing closure."
}
puts "Coverage review is independent of slack: timing_scope_status.txt full_signoff=NOT_CLAIMED"
cd $nl_previous_pwd
# Leave routed design open so Report Utilization/Timing and Device view are available.
