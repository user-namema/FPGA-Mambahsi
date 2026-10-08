# External simulation driver. Frozen package/scripts/sim.tcl remains untouched.
nf_load_build
source [file join $driver_dir saif_register.tcl]
proc stage {message} {
    puts "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}] $message"
    flush stdout
}
set nf_probe [expr {$action eq "probe"}]
set nf_tiles 16
set stamp [clock format [clock seconds] -format %Y%m%d_%H%M%S]
set nf_kind [expr {$nf_probe ? "saifprobe" : "postroute"}]
set nf_simset sim_${nf_kind}_16_${stamp}
catch {close_sim -force}
create_fileset -simset $nf_simset
set fs [get_filesets $nf_simset]
current_fileset -simset $fs
set_property SOURCE_SET sources_1 $fs
add_files -fileset $nf_simset -norecurse [file join $nf_package fullnet sim tb_nf.sv]
# Fresh vectors per run, no edits to existing vectors or old PASS files.
set vector_dir [file join $output vectors]
file mkdir $vector_dir
foreach {source_name dest_name count} {tiles.mem tiles.mem 4096 n3_gold.mem gold.mem 2304} {
    set lines [split [string trim [file_text [file join $nf_package models up_d1 vectors $source_name]]] \n]
    if {[llength $lines]<$count} {error "Short vector file: $source_name"}
    write_new [file join $vector_dir $dest_name] [join [lrange $lines 0 [expr {$count-1}]] \n]
}
add_files -fileset $nf_simset -norecurse [glob [file join $vector_dir *.mem]]
set_property top tb_nf $fs
set_property generic [list TILE_COUNT=16 PERIOD_NS=$nf_period_ns] $fs
set_property verilog_define [get_property verilog_define [get_filesets sources_1]] $fs
set_property xsim.simulate.runtime 0ns $fs
set_property xsim.simulate.log_all_signals false $fs
set_property xsim.elaborate.debug_level all $fs
set_property -name xsim.simulate.xsim.more_options -value {-nosignalhandlers} -objects $fs
update_compile_order -fileset $nf_simset
set nf_simdir [file join $nf_out nf_fullnet.sim $nf_simset impl func xsim]
stage "LAUNCH_BEGIN (compile/elaborate may take tens of minutes; no warmup yet)"
launch_simulation -simset $nf_simset -mode post-implementation -type functional
stage "LAUNCH_END"
if {![file isdirectory $nf_simdir]} {error "Unexpected XSim directory: $nf_simdir"}
set marker [file join $nf_simdir PASS.txt]
if {[file exists $marker]} {error "Unexpected pre-existing PASS in fresh simulation"}
set nf_saif [file join $output [expr {$nf_probe ? "probe_NOT_FOR_POWER.saif" : "activity.saif"}]]
if {!$nf_probe} {
    # Identical to the original unified protocol. Registration does not advance time.
    stage "WARMUP_BEGIN cycles=9216"
    run 92160ns
    stage "WARMUP_END"
    if {[file exists $marker]} {error "Premature finish before power window"}
}
stage "OPEN_SAIF_BEGIN file=$nf_saif"
open_saif $nf_saif
stage "OPEN_SAIF_END"
set registration [saifreg::register /tb_nf/dut [file join $output registration.log] 1024 3]
if {$nf_probe} {
    stage "PROBE_RUN_BEGIN duration=100ns (reset-only; NOT power activity)"
    run 100ns
    stage "PROBE_RUN_END"
} else {
    stage "CAPTURE_BEGIN cycles=32768 duration=327680ns"
    # Same continuous window, split into eight runs for visible progress only.
    for {set part 1} {$part<=8} {incr part} {
        run 40960ns
        if {[file exists $marker]} {error "Simulation ended inside the activity window"}
        stage "CAPTURE_PROGRESS part=$part/8 cycles=[expr {$part*4096}]"
    }
    stage "CAPTURE_END"
}
stage "CLOSE_SAIF_BEGIN"
close_saif
stage "CLOSE_SAIF_END"
if {$nf_probe} {
    if {[file exists $marker]} {error "Unexpected full-network PASS during 100ns probe"}
    if {![file exists $nf_saif] || [file size $nf_saif]<100} {error "Probe SAIF missing/empty"}
    stage "PROBE_REGISTRATION_PASS (not functional PASS, not coverage/power approval)"
} else {
    stage "DRAIN_BEGIN: wait for 16-tile bit-exact PASS"
    set remaining [expr {(1024+15*4096+17000-9216-32768)*$nf_period_ns}]
    run ${remaining}ns
    if {![file exists $marker]} {error "No full-network PASS: inspect $nf_simdir/simulate.log"}
    set passed [file_text $marker]
    if {![string match {*tiles=16 bytes=2304 II=4096*} $passed]} {error "Unexpected PASS protocol"}
    puts $passed
    stage "DRAIN_END"
    set nf_activity_hash $nf_source_hash
    set nf_activity_simdir $nf_simdir
}
