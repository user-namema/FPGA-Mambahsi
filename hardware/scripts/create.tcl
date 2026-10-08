if {[llength $argv]!=1} {error "Expected staged build directory"}
if {[llength [get_projects -quiet]]} {error "Use a new empty Vivado session"}
source [file join [lindex $argv 0] config.tcl]
if {[version -short] ne "2025.2"} {error "This release is pinned to Vivado 2025.2; regenerate/revalidate for another version"}
source [file join $nf_package scripts common.tcl]
if {[nf_hash] ne $nf_source_hash} {error "Frozen source mismatch"}
set name [expr {$hw_profile eq "board" ? "pcie_network_hdmi" : "nf_fullnet"}]
if {[file exists [file join $nf_out ${name}.xpr]]} {error "Project already exists; preserving it"}
create_project $name $nf_out -part xcku060-ffva1156-2-i
set_property target_language Verilog [current_project]
set_property XPM_LIBRARIES {XPM_CDC XPM_FIFO XPM_MEMORY} [current_project]
add_files -norecurse $hw_rtl
set inc {}
foreach f $hw_rtl {lappend inc [file dirname $f]}
set_property include_dirs [lsort -unique $inc] [get_filesets sources_1]
set defs {MAMBA_D_PATH_D1}
if {$hw_profile ne "board"} {lappend defs NF_METHOD=[string index $hw_profile 1]}
set_property verilog_define $defs [get_filesets sources_1]
foreach f $hw_ips {read_ip $f}
# No encrypted/generated vendor HDL is redistributed. Vivado regenerates IP here.
generate_target all [get_ips]
if {$hw_profile eq "board"} {
    set top up_pcie_network_hdmi_top
    set prefix d1
    update_compile_order -fileset sources_1
    add_files -norecurse $hw_bd
    open_bd_design $hw_bd
    # Refresh the authored loader module reference before BD generation.
    foreach cell [get_bd_cells -quiet -hier -filter {TYPE == module}] {
        update_module_reference $cell
    }
    validate_bd_design
    save_bd_design
    generate_target all [get_files $hw_bd]
    add_files -norecurse [make_wrapper -files [get_files $hw_bd] -top]
    add_files -fileset constrs_1 -norecurse $hw_xdc
    set_property PROCESSING_ORDER LATE [get_files */hdmi_internal_timing.xdc]
} else {
    set top nl_fullnet_shell
    set prefix nf
    set xdc [file join $nf_out network.xdc]
    nf_write $xdc {create_clock -name core_clk -period 10.0 [get_ports clk]}
    add_files -fileset constrs_1 -norecurse $xdc
    # Potential inactive ROM references retain their D1 images for elaboration.
    add_files -norecurse [glob [file join $nf_package nonlinear vectors *_n3.mem]]
}
set_property top $top [get_filesets sources_1]
create_run ${prefix}_synth -flow {Vivado Synthesis 2025} -constrset constrs_1
create_run ${prefix}_impl -parent_run ${prefix}_synth -flow {Vivado Implementation 2025} -constrset constrs_1
set_property STRATEGY Performance_Explore [get_runs ${prefix}_impl]
if {$hw_profile ne "board"} {
    set_property STEPS.OPT_DESIGN.TCL.PRE [file join $nf_package fullnet scripts interface_hook.tcl] [get_runs nf_impl]
}
current_run -synthesis [get_runs ${prefix}_synth]
current_run -implementation [get_runs ${prefix}_impl]
update_compile_order -fileset sources_1
report_compile_order -used_in synthesis -file [file join $nf_out compile_order.rpt]
report_ip_status -file [file join $nf_out ip_status.rpt]
nf_write [file join $nf_out PREPARED.txt] "Prepared from $nf_source_hash; no simulation/synthesis/route run"
puts "PREPARED: $nf_out/${name}.xpr; next run hardware/run.py sim $nf_out"
