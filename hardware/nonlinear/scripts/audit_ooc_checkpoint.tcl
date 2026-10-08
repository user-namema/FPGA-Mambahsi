# Read-only analysis of an existing routed checkpoint. No opt/place/route,
# no false paths, no modification or overwrite of the input checkpoint.
if {[llength $argv] != 2} {error "Usage: <routed.dcp> <NEW report directory>"}
lassign $argv audit_dcp audit_out
if {[file exists $audit_out]} {error "Audit output exists; select a new directory"}
file mkdir $audit_out
set_param general.maxThreads 2
open_checkpoint $audit_dcp
set audit_f [open [file join $audit_out properties.txt] w]
puts $audit_f "checkpoint=[file normalize $audit_dcp]"
puts $audit_f "version=[version -short]"
puts $audit_f "clk_properties:"
puts $audit_f [report_property -all -return_string [get_ports clk]]
set audit_dsps [get_cells -hier -filter {REF_NAME == DSP48E2}]
puts $audit_f "DSP48E2 count=[llength $audit_dsps]"
if {[llength $audit_dsps]} {
    set audit_dsp [lindex $audit_dsps 0]
    puts $audit_f [report_property -all -return_string $audit_dsp]
    set audit_bp [get_pins -of_objects $audit_dsp -filter {REF_PIN_NAME == B[0]}]
    puts $audit_f "B0=$audit_bp"
    if {[llength $audit_bp]} {
        puts $audit_f [report_property -all -return_string $audit_bp]
        set audit_net [get_nets -of_objects $audit_bp]
        puts $audit_f [report_property -all -return_string $audit_net]
    }
}
close $audit_f
check_timing -verbose -file [file join $audit_out check_timing.rpt]
report_clocks -file [file join $audit_out clocks.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -max_paths 1200 -file [file join $audit_out timing_summary.rpt]
report_route_status -file [file join $audit_out route_status.rpt]
report_exceptions -file [file join $audit_out exceptions.rpt]
set audit_rf [open [file join $audit_out timing_summary.rpt] r]
set audit_text [read $audit_rf]
close $audit_rf
set audit_sources {}
foreach {whole source} [regexp -all -inline {Slack:\s+inf\s+Source:\s+(\S+)} $audit_text] {
    lappend audit_sources $source
}
set audit_f [open [file join $audit_out dsp_startpoints.txt] w]
puts $audit_f "Reported infinite-slack paths=[llength $audit_sources]"
set audit_unproven 0
foreach audit_p [lsort -unique $audit_sources] {
    if {![regexp {^(.*)/DSP_A_B_DATA_INST/(A2_DATA|B2_DATA|ACOUT|BCOUT)\[([0-9]+)\]$} $audit_p -> parent internal bit]} {
        puts $audit_f "UNPROVEN $audit_p (not A/B data/cascade)"
        incr audit_unproven
        continue
    }
    set cell [get_cells -quiet -hier -filter [format {NAME == "%s"} $parent]]
    set side [string index $internal 0]
    set reg_property ${side}REG
    if {$internal in {ACOUT BCOUT}} {set reg_property ${side}CASCREG}
    set bp [get_pins -quiet -of_objects $cell -filter [format {REF_PIN_NAME == "%s[%d]"} $side $bit]]
    set nets [get_nets -quiet -of_objects $bp]
    set types [get_property TYPE $nets]
    set breg [get_property $reg_property $cell]
    set mode [get_property ${side}_INPUT $cell]
    if {[llength $nets] == 1 && $types in {GROUND POWER} && $breg == 0 && $mode eq "DIRECT"} {
        puts $audit_f "CONSTANT_BYPASS $audit_p $reg_property=$breg ${side}_INPUT=$mode net_type=$types"
    } else {
        incr audit_unproven
        puts $audit_f "UNPROVEN $audit_p $reg_property=$breg ${side}_INPUT=$mode net_type=$types"
    }
}
puts $audit_f "unique_sources=[llength [lsort -unique $audit_sources]] unproven=$audit_unproven"
close $audit_f
write_verilog -force [file join $audit_out routed_netlist.v]
puts "AUDIT COMPLETE (original DCP preserved): $audit_out"
