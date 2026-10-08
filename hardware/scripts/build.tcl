if {[llength $argv]!=1} {error "Expected build directory"}
set nf_out [lindex $argv 0]
source [file join [file dirname [info script]] common.tcl]
nf_load_build
set proof [file join $nf_out sim_ok.txt]
if {![file exists $proof]} {error "Run simulation first"}
set f [open $proof r];set passed [read $f];close $f
if {[lindex [split $passed \n] 0] ne $nf_source_hash} {error "Stale simulation"}
set prefix [expr {$hw_profile eq "board" ? "d1" : "nf"}]
if {[get_property STATUS [get_runs ${prefix}_synth]] ne "Not started"} {error "Run already started; preserve results, use a new build or explicitly resume in GUI"}
launch_runs ${prefix}_synth -jobs 2
wait_on_run ${prefix}_synth
open_run ${prefix}_synth
file mkdir [file join $nf_out synth_reports]
report_utilization -file [file join $nf_out synth_reports utilization.rpt]
report_utilization -hierarchical -file [file join $nf_out synth_reports hierarchy.rpt]
write_checkpoint [file join $nf_out network_synth.dcp]
set dsp [llength [get_cells -hier -filter {REF_NAME =~ DSP48*}]]
set br36 [llength [get_cells -hier -filter {REF_NAME =~ RAMB36*}]]
set br18 [llength [get_cells -hier -filter {REF_NAME =~ RAMB18*}]]
nf_write [file join $nf_out capacity.txt] "DSP=$dsp /2760; BRAM36_equiv=[expr {$br36+$br18/2.0}] /1080; LUT usage: synth_reports/utilization.rpt"
if {$dsp>2760 || $br36+$br18/2.0>1080} {
    nf_write [file join $nf_out build_status.txt] "DOES_NOT_FIT_KU060\nSynthesis report/DCP retained; no routed frequency/power"
    error "Capacity exceeded; this is an experimental result, not a missing report"
}
close_design
launch_runs ${prefix}_impl -to_step route_design -jobs 2
wait_on_run ${prefix}_impl
open_run ${prefix}_impl
if {$hw_profile eq "board"} {phys_opt_design -directive Explore}
write_checkpoint [file join $nf_out network_routed.dcp]
set reports [file join $nf_out routed_reports]
file mkdir $reports
report_route_status -file [file join $reports route_status.rpt]
if {$hw_profile ne "board"} {
    source [file join $nf_package fullnet scripts report.tcl]
    set met [nf_reports $reports]
} else {
    report_utilization -file [file join $reports utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $reports timing_summary.rpt]
    check_timing -verbose -file [file join $reports check_timing.rpt]
    report_cdc -file [file join $reports cdc.rpt]
    report_drc -file [file join $reports drc.rpt]
    set met 1
    foreach type {max min} {
        set p [get_timing_paths -delay_type $type -max_paths 1]
        if {![llength $p] || [get_property SLACK $p]<0} {set met 0}
    }
}
nf_write [file join $nf_out build_status.txt] "ROUTED\nobserved_setup_hold_met=$met\ncoverage_review=PENDING"
if {$hw_profile eq "board"} {
    if {!$met} {error "Timing failed; bitstream withheld"}
    write_bitstream [file join $nf_out up_pcie_network_hdmi_D1.bit]
}
puts "DONE: reports and DCP retained at $nf_out. Review coverage/DRC; no Fmax inferred from slack."
