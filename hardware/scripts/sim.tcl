if {[llength $argv]!=1} {error "Expected build directory"}
set nf_out [lindex $argv 0]
source [file join [file dirname [info script]] common.tcl]
nf_load_build
catch {close_sim -force}
set fsname sim_release_[clock format [clock seconds] -format %Y%m%d_%H%M%S]
create_fileset -simset $fsname
set fs [get_filesets $fsname]
current_fileset -simset $fs
set_property SOURCE_SET sources_1 $fs
set vectors [file join $nf_out $fsname]
file mkdir $vectors
if {$hw_profile eq "board"} {
    set top tb_d1_network_ddr_hdmi
    add_files -fileset $fsname -norecurse [file join $nf_package board sim ${top}.sv]
    foreach p [glob [file join $nf_package board sim *.mem]] {file copy $p $vectors}
    set_property generic {REAL_FIVE=0} $fs
    set expect {PASS: D1 repeated-first-tile bank loader -> network -> DDR -> HDMI -> C2H.}
} else {
    set top tb_nf
    add_files -fileset $fsname -norecurse [file join $nf_package fullnet sim tb_nf.sv]
    file copy [file join $nf_package models up_d1 vectors tiles.mem] [file join $vectors tiles.mem]
    file copy [file join $nf_package models up_d1 vectors [string tolower $nf_method]_gold.mem] [file join $vectors gold.mem]
    set_property generic {TILE_COUNT=16 PERIOD_NS=10.0} $fs
    set expect {PASS independent method-specific software logits; tiles=16 bytes=2304 II=4096}
}
add_files -fileset $fsname -norecurse [glob [file join $vectors *.mem]]
set_property top $top $fs
set_property verilog_define [get_property verilog_define [get_filesets sources_1]] $fs
set_property xsim.simulate.runtime 0ns $fs
set_property xsim.simulate.log_all_signals false $fs
set_property xsim.elaborate.debug_level typical $fs
set_property -name xsim.simulate.xsim.more_options -value {-nosignalhandlers} -objects $fs
update_compile_order -fileset $fsname
launch_simulation -simset $fsname -mode behavioral
run all
set project [get_property NAME [current_project]]
set directory [file join $nf_out ${project}.sim $fsname behav xsim]
set proof [file join $directory [expr {$hw_profile eq "board" ? "simulate.log" : "PASS.txt"}]]
if {![file exists $proof]} {error "Missing PASS evidence: $proof"}
set f [open $proof r];set text [read $f];close $f
if {[string first $expect $text]<0} {error "Simulation did not reach the required PASS; do not build"}
nf_write [file join $nf_out sim_ok.txt] "$nf_source_hash\n$directory\n$expect"
puts "SIM VERIFIED: $directory; no second run all after finish"
