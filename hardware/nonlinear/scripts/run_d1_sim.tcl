# Common D1 readout regression; independent from N0--N3 coefficient tests.
source [file join [file dirname [file normalize [info script]]] common.tcl]
if {[llength [get_projects -quiet]]} {error "Use a new Vivado window without an open project"}
if {![info exists nl_core]} {set nl_core blk0_spa}
nl_check_core $nl_core
if {![info exists nl_out]} {
    error "Set nl_out to a NEW short ASCII build directory"
}
set nl_out [nl_new_directory $nl_out]
create_project nl_sim $nl_out -part xcku060-ffva1156-2-i
add_files -norecurse [nl_sources]
add_files -norecurse [glob [file join $nl_root vectors *.mem]]
add_files -fileset sim_1 -norecurse [list [file join $nl_root sim nl_d1_tb.sv] [file join $nl_root sim generated_d1_tb_tops.sv]]
set_property include_dirs [list [file join $nl_root rtl]] [get_filesets sources_1]
set_property top tb_d1_${nl_core} [get_filesets sim_1]
set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
set_property xsim.elaborate.mt_level 2 [get_filesets sim_1]
set_property -name xsim.simulate.xsim.more_options -value {-nosignalhandlers} -objects [get_filesets sim_1]
set directory [file join $nl_out nl_sim.sim sim_1 behav xsim]
nl_copy_data $directory
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
launch_simulation -mode behavioral
run all
set marker [file join $directory ${nl_core}_d1_pass.txt]
if {![file exists $marker]} {error "D1 regression did not reach PASS: $directory"}
set f [open $marker r]
puts [read $f]
close $f
puts "Do not run all again after PASS."
