# Source after synthesis or open_checkpoint. No timing exceptions are added.
proc nl_csv {value} {
    return "\"[string map [list \" \"\"] $value]\""
}

proc nl_path_stats {options delay_type} {
    set paths [get_timing_paths {*}$options -delay_type $delay_type -max_paths 1 -nworst 1]
    if {![llength $paths]} {return [list NO_PATH {} {} {}]}
    set path [lindex $paths 0]
    set slack [get_property SLACK $path]
    set start [get_property STARTPOINT_PIN $path]
    set end [get_property ENDPOINT_PIN $path]
    if {![string is double -strict $slack] ||
        [string match -nocase *inf* $slack] || [string match -nocase *nan* $slack]} {
        return [list UNCONSTRAINED $slack $start $end]
    }
    return [list [expr {$slack >= 0 ? "PASS" : "FAIL"}] $slack $start $end]
}

proc nl_report_split_timing {directory} {
    set clocks [get_clocks core_clk]
    set regs [all_registers -clock $clocks]
    set inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set outputs [get_ports -filter {DIRECTION == OUT}]
    if {![llength $regs]} {error "No sequential cells on core_clk; cannot report REG2REG"}

    # Clock-to-clock selectors alone also include clock-constrained I/O paths.
    # Explicit sequential cell collections keep port paths out of REG2REG.
    set scopes [list \
        [list REG2REG [list -from $regs -to $regs] 1] \
        [list PORT2REG [list -from $inputs -to $regs] 1] \
        [list REG2PORT [list -from $regs -to $outputs] 1] \
        [list PORT2PORT [list -from $inputs -to $outputs] 0]]
    set f [open [file join $directory timing_groups.csv] w]
    puts $f "scope,analysis,status,slack_ns,startpoint,endpoint"
    set sf [open [file join $directory timing_scope_status.txt] w]
    puts $sf "timing_model=shared_fabric_clock_reference_v2"
    puts $sf "register_cells=[llength $regs]\ninput_ports=[llength $inputs]\noutput_ports=[llength $outputs]"
    set observed_ok 1
    foreach spec $scopes {
        lassign $spec scope options required
        foreach {type label} {max setup min hold} {
            set row [nl_path_stats $options $type]
            lassign $row status slack start end
            if {$status eq "NO_PATH" && $required} {set status MISSING_REQUIRED_PATH}
            if {$status ni {PASS NO_PATH}} {set observed_ok 0}
            puts $sf "${scope}_${label}=$status slack_ns=$slack"
            set values [list $scope $label $status $slack $start $end]
            set escaped {}
            foreach value $values {lappend escaped [nl_csv $value]}
            puts $f [join $escaped ,]
            set filename [file join $directory ${scope}_${label}.rpt]
            if {$status in {NO_PATH MISSING_REQUIRED_PATH}} {
                set empty [open $filename w]
                puts $empty "$scope ${label}: $status"
                close $empty
            } else {
                report_timing {*}$options -delay_type $type -max_paths 20 -nworst 1 \
                    -path_type full_clock_expanded -input_pins -file $filename
            }
        }
    }
    puts $sf "observed_split_setup_hold_met=$observed_ok"
    puts $sf "coverage_review=PENDING_REVIEW"
    puts $sf "full_signoff=NOT_CLAIMED"
    puts $sf "A positive finite worst slack is not proof that all unconstrained DSP bypass paths are constants."
    close $sf
    close $f
    # Preserve full unfiltered reports: split reports are NOT exceptions.
    report_timing_summary -delay_type min_max -report_unconstrained \
        -max_paths 100 -file [file join $directory timing_summary_routed.rpt]
    report_timing -delay_type max -max_paths 20 -path_type full_clock_expanded \
        -file [file join $directory setup_paths.rpt]
    report_timing -delay_type min -max_paths 20 -path_type full_clock_expanded \
        -file [file join $directory hold_paths.rpt]
    report_exceptions -file [file join $directory exceptions.rpt]
    report_clocks -file [file join $directory clocks.rpt]
    report_clock_utilization -file [file join $directory clock_utilization.rpt]
    check_timing -verbose -file [file join $directory check_timing_routed.rpt]
    puts "SPLIT TIMING REPORTS: $directory; coverage remains PENDING_REVIEW"
}
