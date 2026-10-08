# OOC fabric-interface contract, version 2.
# These are assumed integration budgets, NOT board device/PCB timing numbers.
# Use the same values for N0/N1/N2/N3. Never derive budgets from measured slack.
proc nl_interface_defaults {} {
    foreach {name value} {
        nl_period_ns 10.000
        nl_input_min_ns 0.200
        nl_input_max_ns 1.000
        nl_output_hold_ns 0.100
        nl_output_setup_ns 1.000
    } {
        if {![info exists ::$name]} {set ::$name $value}
        if {![string is double -strict [set ::$name]] ||
            [string match -nocase *inf* [set ::$name]] ||
            [string match -nocase *nan* [set ::$name]]} {
            error "$name must be a finite numeric value"
        }
    }
    if {$::nl_period_ns <= 0 || $::nl_input_min_ns < 0 ||
        $::nl_input_max_ns < $::nl_input_min_ns ||
        $::nl_output_hold_ns < 0 || $::nl_output_setup_ns < 0} {
        error "Invalid OOC interface delay interval"
    }
}

proc nl_apply_interface_timing {directory} {
    nl_interface_defaults
    set clock_port [get_ports -quiet clk]
    # Reference the complete shared clock distribution, not only BUFG.O.
    # BUFG.O precedes the ~3 ns routed distribution to register clock pins.
    # rst_q_reg is an existing synchronous boundary FF in all four shells.
    set reference [get_pins -quiet rst_q_reg/C]
    if {[llength $clock_port] != 1 || [llength $reference] != 1} {
        error "Expected clocked shell with clk and rst_q_reg/C. Bare OOC is not supported by timing_v2."
    }
    set buffer [get_cells -quiet u_clk]
    if {[get_property REF_NAME $buffer] ne "BUFGCE"} {
        error "u_clk must be the real BUFGCE shared by shell and core"
    }
    set inputs [get_ports -filter {DIRECTION == IN && NAME != clk}]
    set outputs [get_ports -filter {DIRECTION == OUT}]
    set expected_in 10
    set expected_out 428
    if {[info exists ::nl_with_d1] && $::nl_with_d1} {
        set expected_in 73
        set expected_out 477
    }
    if {[llength $inputs] != $expected_in || [llength $outputs] != $expected_out} {
        error "Unexpected interface: inputs=[llength $inputs], outputs=[llength $outputs]"
    }
    set clocks [get_clocks -quiet core_clk]
    if {![llength $clocks]} {
        create_clock -name core_clk -period $::nl_period_ns $clock_port
        set clocks [get_clocks core_clk]
    } elseif {[llength $clocks] != 1 ||
              abs([get_property PERIOD $clocks] - $::nl_period_ns) > 0.000001} {
        error "Existing core_clk period differs from requested nl_period_ns; do not silently retime a checkpoint"
    }

    # Explicitly replace BOTH old min and max delays. With no -add_delay,
    # set_input_delay/set_output_delay replace the old constraint of that type.
    # Vivado does not provide the Synopsys remove_input_delay command.
    # The reference clock stays at clk, so physical BUFG clock routing is kept.
    # reference_pin relocates only the hypothetical external launch/capture
    # timing reference to a representative physical fabric-clock leaf.
    set_input_delay -clock $clocks -reference_pin $reference \
        -min $::nl_input_min_ns $inputs
    set_input_delay -clock $clocks -reference_pin $reference \
        -max $::nl_input_max_ns $inputs
    # Output min is NEGATIVE receiver hold time, assuming zero additional skew.
    set_output_delay -clock $clocks -reference_pin $reference \
        -min [expr {-$::nl_output_hold_ns}] $outputs
    set_output_delay -clock $clocks -reference_pin $reference \
        -max $::nl_output_setup_ns $outputs

    set f [open [file join $directory interface_model.txt] w]
    puts $f "model=shared_fabric_clock_reference_v2"
    puts $f "scope=assumed OOC fabric integration; NOT board I/O signoff"
    puts $f "clock_origin=clk\nport_delay_reference=rst_q_reg/C"
    puts $f "interface_clock_assumption=external fabric launch/capture aligned to this physical clock leaf"
    puts $f "period_ns=$::nl_period_ns"
    puts $f "input_min_ns=$::nl_input_min_ns\ninput_max_ns=$::nl_input_max_ns"
    puts $f "output_min_ns=[expr {-$::nl_output_hold_ns}]\noutput_max_ns=$::nl_output_setup_ns"
    puts $f "reset_model=synchronous data through rst_q; no reset false path"
    puts $f "false_paths_added=0\nmulticycle_paths_added=0\nclock_latency_override=0"
    puts $f "Budgets must be fixed before comparisons, not selected to pass measured slack."
    puts $f "For a real board interface, use actual external clock/CQ/setup/hold/PCB data instead."
    close $f
    write_xdc [file join $directory effective_constraints.xdc]
}
