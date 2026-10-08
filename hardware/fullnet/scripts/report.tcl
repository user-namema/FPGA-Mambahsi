proc nf_reports {directory} {
    file mkdir $directory
    report_utilization -file [file join $directory utilization.rpt]
    report_utilization -hierarchical -file [file join $directory hierarchy.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained \
        -max_paths 100 -file [file join $directory timing_summary.rpt]
    check_timing -verbose -file [file join $directory check_timing.rpt]
    report_clocks -file [file join $directory clocks.rpt]
    report_exceptions -file [file join $directory exceptions.rpt]
    report_drc -file [file join $directory drc.rpt]
    write_xdc -force [file join $directory effective.xdc]
    set regs [all_registers -clock [get_clocks core_clk]]
    if {![llength $regs]} {error "No sequential cells on core_clk"}
    set ins [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set outs [get_ports -filter {DIRECTION == OUT}]
    set f [open [file join $directory split_timing.csv] w]
    puts $f "scope,analysis,status,slack_ns"
    set met 1
    foreach {scope from to} [list REG2REG $regs $regs PORT2REG $ins $regs REG2PORT $regs $outs PORT2PORT $ins $outs] {
        foreach {type label} {max setup min hold} {
            set paths [get_timing_paths -from $from -to $to -delay_type $type -max_paths 1]
            set status NO_PATH; set slack {}
            if {[llength $paths]} {
                set slack [get_property SLACK [lindex $paths 0]]
                if {![string is double -strict $slack] || [string match -nocase *inf* $slack]} {
                    set status UNCONSTRAINED
                } else {set status [expr {$slack>=0 ? "PASS" : "FAIL"}]}
                report_timing -from $from -to $to -delay_type $type -max_paths 20 \
                    -path_type full_clock_expanded -file [file join $directory ${scope}_${label}.rpt]
            } elseif {$scope ne "PORT2PORT"} {set status MISSING_REQUIRED_PATH}
            if {$status ni {PASS NO_PATH}} {set met 0}
            puts $f "$scope,$label,$status,$slack"
        }
    }
    close $f
    set f [open [file join $directory timing_status.txt] w]
    puts $f "observed_setup_hold_met=$met"
    puts $f "frequency_test_mhz=[expr {1000.0/[get_property PERIOD [get_clocks core_clk]]}]"
    puts $f "coverage_review=PENDING; inspect check_timing, unconstrained, DRC and exceptions"
    puts $f "fmax=NOT_INFERRED_FROM_WNS; highest passing sweep point is only a tested lower bound"
    puts $f "scope=whole inference network, assumed registered fabric boundary; not board IO"
    close $f
    return $met
}
