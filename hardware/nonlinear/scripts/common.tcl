# Independent OOC comparison package. Never opens or modifies the board project.
set nl_root [file dirname [file dirname [file normalize [info script]]]]
proc nl_check_core {core} {
    if {$core ni {blk0_spa blk0_spe blk1_spa blk1_spe blk2_spa blk2_spe}} {
        error "Unknown core: $core"
    }
}
proc nl_new_directory {path} {
    set path [file normalize $path]
    if {[file exists $path]} {error "Output exists; choose a NEW nl_out. Preserving $path"}
    if {![string is ascii $path] || [string length $path] > 75} {
        error "Use a short ASCII output path, e.g. E:/nl_compare/n0_b0spa_v1"
    }
    file mkdir $path
    return $path
}
proc nl_copy_data {destination} {
    global nl_root
    file mkdir $destination
    foreach p [glob [file join $nl_root vectors *.mem]] {
        file copy $p [file join $destination [file tail $p]]
    }
    set snapshot [file join $destination D1_source_snapshot]
    file mkdir $snapshot
    foreach relative {rtl sim scripts vectors} {
        file copy [file join $nl_root $relative] $snapshot
    }
    # Unified packages carry checkpoint identity and the exact generated
    # constants beside each new run. Old runs/packages remain untouched.
    if {[file exists [file join $nl_root unified_manifest.json]]} {
        set snapshot [file join $destination unified_source_snapshot]
        file mkdir $snapshot
        foreach relative {unified_manifest.json n2_config.json n2_numeric_report.json TIMING_V2.md rtl sim scripts vectors provenance} {
            file copy [file join $nl_root $relative] $snapshot
        }
        puts "UNIFIED CONSTANTS: [file join $snapshot unified_manifest.json]"
    }
}
proc nl_sources {} {
    global nl_root
    return [list [file join $nl_root rtl nl_math.sv] \
        [file join $nl_root rtl nl_coeff.sv] \
        [file join $nl_root rtl generated_tops.sv] [file join $nl_root rtl nl_n2.sv] \
        [file join $nl_root rtl generated_n2_tops.sv] [file join $nl_root rtl clocked_tops.sv] \
        [file join $nl_root rtl mamba_d_path_lut.v] [file join $nl_root rtl nl_d1_readout.sv] \
        [file join $nl_root rtl generated_d1_tops.sv]]
}
foreach p [concat [nl_sources] [list [file join $nl_root vectors manifest.json] \
        [file join $nl_root rtl math_constants.vh]]] {
    if {![file exists $p]} {error "Missing package file $p; restore the corresponding file under hardware/nonlinear from the complete hardware package"}
}
