# Shared portable full-network/board state. No fixed workstation paths.
set nf_package [file dirname [file dirname [file normalize [info script]]]]
proc nf_write {path text} {
    set f [open $path w]; puts $f $text; close $f
}
proc nf_hash {} {
    return [string trim [exec $::nf_python -I [file join $::nf_package tools verify.py]]]
}
proc nf_load_build {} {
    global nf_out nf_package hw_profile
    set nf_out [file normalize $nf_out]
    if {[info exists ::nf_python]} {set selected_python $::nf_python}
    uplevel #0 [list source [file join $nf_out config.tcl]]
    if {[info exists selected_python]} {set ::nf_python $selected_python}
    set nf_package [file join $nf_out package]
    if {[nf_hash] ne $::nf_source_hash} {error "Frozen package changed; prepare a new build"}
    set name [expr {$hw_profile eq "board" ? "pcie_network_hdmi" : "nf_fullnet"}]
    if {![llength [get_projects -quiet]]} {open_project [file join $nf_out ${name}.xpr]}
    if {[file normalize [get_property DIRECTORY [current_project]]] ne $nf_out} {error "Another project is open"}
    set prefix [expr {$hw_profile eq "board" ? "d1" : "nf"}]
    current_run -synthesis [get_runs ${prefix}_synth]
    current_run -implementation [get_runs ${prefix}_impl]
}
set_param general.maxThreads 2
