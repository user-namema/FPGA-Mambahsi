# New Vivado session, no project open. nl_core defaults to blk0_spa.
source [file join [file dirname [file normalize [info script]]] common.tcl]
if {[llength [get_projects -quiet]]} {error "Use a new Vivado session without an open project"}
if {![info exists nl_core]} {set nl_core blk0_spa}
nl_check_core $nl_core
if {![info exists nl_out]} {
    error "Set nl_out to a NEW short ASCII build directory"
}
set nl_out [nl_new_directory $nl_out]
create_project nl_sim $nl_out -part xcku060-ffva1156-2-i
set_property target_language Verilog [current_project]
add_files -norecurse [nl_sources]
add_files -norecurse [glob [file join $nl_root vectors *.mem]]
set_property include_dirs [list [file join $nl_root rtl]] [get_filesets sources_1]
set_property include_dirs [list [file join $nl_root rtl]] [get_filesets sim_1]
add_files -fileset sim_1 -norecurse [list [file join $nl_root sim nl_tb.sv] \
    [file join $nl_root sim generated_tb_tops.sv]]
if {[info exists nl_method] && $nl_method eq "N2"} {
    add_files -fileset sim_1 -norecurse [list [file join $nl_root sim nl_n2_tb.sv] [file join $nl_root sim generated_n2_tb_tops.sv]]
    set_property top tb_n2_${nl_core} [get_filesets sim_1]
} else {set_property top tb_${nl_core} [get_filesets sim_1]}
set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
set_property xsim.elaborate.mt_level 2 [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value {-nosignalhandlers} -objects [get_filesets sim_1]
set nl_sim_dir [file join $nl_out nl_sim.sim sim_1 behav xsim]
nl_copy_data $nl_sim_dir
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -mode behavioral
run all
set nl_marker [file join $nl_sim_dir ${nl_core}_pass.txt]
if {![file exists $nl_marker]} {
    error "Simulation did not reach PASS; inspect Tcl Console and $nl_sim_dir/simulate.log"
}
set mf [open $nl_marker r]
puts [read $mf]
close $mf
set nl_csv ${nl_core}_coefficients.csv
if {[info exists nl_method] && $nl_method eq "N2"} {set nl_csv ${nl_core}_n2_coefficients.csv}
puts "NUMERIC CSV: [file join $nl_sim_dir $nl_csv]"
puts "Do not issue run all again after PASS. Simulation remains open for waveform inspection."
