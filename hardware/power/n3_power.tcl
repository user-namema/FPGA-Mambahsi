# Independent driver: never writes frozen package, RTL, constraints or DCPs.
if {[llength $argv] ni {5 7}} {error "Expected mode build output capture python ?repaired_saif sha256?"}
lassign $argv action requested_build output capture_dir requested_python
if {[llength $argv]==7} {
    set repaired_saif [file normalize [lindex $argv 5]]
    set repaired_sha [lindex $argv 6]
    if {$action ne "report"} {error "Repaired SAIF is report-only"}
}
if {$action ni {probe capture report}} {error "Unsupported action"}
if {[llength [get_projects -quiet]]} {error "Use a NEW empty Vivado process"}
set driver_dir [file dirname [file normalize [info script]]]
set nf_out [file normalize $requested_build]
set output [file normalize $output]
set capture_dir [file normalize $capture_dir]
proc file_text {path} {
    set f [open $path r]; set s [read $f]; close $f; return $s
}
proc hash_file {path} {
    return [string trim [exec $::nf_python -I [file join $::driver_dir n3_power.py] --sha256 $path]]
}
proc write_new {path text} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f $text
    close $f
}
source [file join $nf_out package scripts common.tcl]
source [file join $nf_out config.tcl]
set nf_python $requested_python
if {$nf_method ne "N3" || abs($nf_period_ns-10.0)>0.000001} {error "Requires N3 at 100 MHz"}
if {[nf_hash] ne $nf_source_hash} {error "Frozen package changed"}
set checkpoint [file join $nf_out network_routed.dcp]
set dcp_hash [hash_file $checkpoint]
if {$action in {probe capture}} {
    set registration_hash [hash_file [file join $driver_dir saif_register.tcl]]
    set simulation_driver_hash [hash_file [file join $driver_dir sim_capture_v2.tcl]]
    if {$action eq "capture"} {
        set receipt [file join $capture_dir PROBE_OK.tcl]
        if {![file exists $receipt]} {error "Run probe first; no PROBE_OK.tcl"}
        source $receipt
        if {$probe_source_hash ne $nf_source_hash || $probe_dcp_hash ne $dcp_hash ||
            [file normalize $probe_build] ne $nf_out ||
            $probe_registration_hash ne $registration_hash ||
            $probe_simulation_driver_hash ne $simulation_driver_hash} {
            error "Probe does not match this build, DCP or registration scripts"
        }
    }
    puts "N3 $action: external SAIF v2 driver; frozen package and DCP preserved"
    flush stdout
    source [file join $driver_dir sim_capture_v2.tcl]
    if {[hash_file $checkpoint] ne $dcp_hash || [nf_hash] ne $nf_source_hash} {
        error "DCP or frozen package changed during simulation"
    }
    if {$action eq "probe"} {
        set metadata {}
        foreach {name value} [list probe_build $nf_out probe_source_hash $nf_source_hash \
            probe_dcp_hash $dcp_hash probe_registration_hash $registration_hash \
            probe_simulation_driver_hash $simulation_driver_hash \
            probe_objects [dict get $registration objects] \
            probe_scopes [dict get $registration scopes] probe_simdir $nf_simdir] {
            append metadata [list set $name $value] \n
        }
        write_new [file join $output PROBE_OK.tcl] $metadata
        catch {close_sim -force}
        close_project
        puts "PROBE COMPLETE: $output/PROBE_OK.tcl; NOT a power result"
        return
    }
    if {[dict get $registration objects]!=$probe_objects ||
        [dict get $registration scopes]!=$probe_scopes} {
        error "Probe/capture enumeration counts differ; review coverage before using SAIF"
    }
    if {$nf_activity_hash ne $nf_source_hash} {error "Activity source hash mismatch"}
    if {![file exists $nf_saif] || [file size $nf_saif]<100} {error "Missing/empty SAIF"}
    set passfile [file join $nf_activity_simdir PASS.txt]
    if {![file exists $passfile]} {error "Missing final 16-tile post-route PASS"}
    if {[hash_file $checkpoint] ne $dcp_hash} {error "Routed checkpoint changed during capture"}
    set metadata {}
    foreach {name value} [list capture_build $nf_out capture_source_hash $nf_source_hash \
        capture_dcp_hash $dcp_hash capture_saif $nf_saif capture_saif_hash [hash_file $nf_saif] \
        capture_simdir $nf_activity_simdir capture_pass_hash [hash_file $passfile] \
        capture_registration_hash $registration_hash capture_simulation_driver_hash $simulation_driver_hash \
        capture_registration $registration capture_probe_dir $capture_dir] {
        append metadata [list set $name $value] \n
    }
    write_new [file join $output postroute_pass.txt] [file_text $passfile]
    # Written only after golden verification and SAIF close both succeeded.
    write_new [file join $output CAPTURE_OK.tcl] $metadata
    catch {close_sim -force}
    close_project
    puts "CAPTURE COMPLETE: $output/CAPTURE_OK.tcl"
} else {
    set completed [file join $capture_dir CAPTURE_OK.tcl]
    if {![file exists $completed]} {error "No successful capture metadata"}
    source $completed
    if {[file normalize $capture_build] ne $nf_out || $capture_source_hash ne $nf_source_hash} {
        error "Capture belongs to another build/source"
    }
    if {$capture_dcp_hash ne $dcp_hash || [hash_file $capture_saif] ne $capture_saif_hash} {
        error "DCP/SAIF hash mismatch"
    }
    if {[hash_file [file join $capture_simdir PASS.txt]] ne $capture_pass_hash} {error "Post-route PASS changed"}
    set report_saif $capture_saif
    set saif_derivation original
    if {[info exists repaired_saif]} {
        if {![file isfile $repaired_saif] || [hash_file $repaired_saif] ne $repaired_sha} {
            error "Repaired SAIF missing or hash mismatch"
        }
        set report_saif $repaired_saif
        set saif_derivation xsim-null-net-name-v1
    }
    open_checkpoint $checkpoint
    set_operating_conditions -process typical -junction_temp 50
    puts "N3 REPORT: reading SAIF without a per-net unmatched dump"
    flush stdout
    # The previous diagnostic dump grew beyond 2.8 GB from simulation-only
    # primitive nets. Vivado still reports annotation statistics in its log.
    read_saif -strip_path tb_nf/dut $report_saif
    puts "N3 REPORT: SAIF import completed; generating the main power report"
    flush stdout
    # -all must be paired with -type; do not reuse the old bare -all command.
    # Main report first, so optional diagnostic failures do not lose the estimate.
    report_power -hier all -hierarchical_depth 12 -file [file join $output power_saif.rpt]
    puts "N3 REPORT: main power report completed"
    flush stdout
    report_power -advisory -file [file join $output power_advisory.rpt]
    report_operating_conditions -file [file join $output operating_conditions.rpt]
    set diagnostics {}
    foreach kind {register lut lut_ram dsp bram_enable bram_wr_enable} {
        if {[catch {
            report_switching_activity -type $kind -all -file [file join $output activity_${kind}.rpt]
        } reason]} {append diagnostics "$kind: $reason\n"}
    }
    write_new [file join $output diagnostic_errors.txt] $diagnostics
    write_new [file join $output protocol.txt] \
        "method=N3\nfrequency_mhz=100\ninput_II=4096\ntiles=16 first DIFFERENT scene tiles\nwindow_start_cycle=9216\nwindow_cycles=32768\nwindow_ns=327680\nprocess=typical\njunction_c=50\nsource_sha256=$nf_source_hash\nrouted_dcp_sha256=$dcp_hash\noriginal_saif=$capture_saif\nreport_saif=$report_saif\nsaif_derivation=$saif_derivation\nsimulation=$capture_simdir\nactivity_source=post-route FUNCTIONAL; no SDF glitch timing\npower_kind=Vivado estimate; NOT board measurement\nunmatched_dump=disabled; simulation-only primitive nets produced a multi-GB dump\ncoverage_review=PENDING; inspect Vivado annotation statistics, confidence and activity reports\nscope=full D1 inference fabric; excludes PCIe/DDR/HDMI\n"
    write_new [file join $output REPORT_COMPLETE.txt] "Reports generated; activity coverage still requires review."
    close_design
    puts "N3 POWER REPORT: $output/power_saif.rpt"
}
