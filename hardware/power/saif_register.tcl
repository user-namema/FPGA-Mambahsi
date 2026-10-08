# No signal/type filters: root locals + disjoint child subtrees cover the DUT.
# All XSim API errors are fatal. Do not use -quiet or silently skip scopes.
namespace eval saifreg {
    variable version saif-partitions-v2
    variable stream
    variable objects 0
    variable batches 0
    variable seen
}

proc saifreg::note {message} {
    variable stream
    set line "[clock format [clock seconds] -format {%Y-%m-%d %H:%M:%S}] $message"
    puts $line
    flush stdout
    puts $stream $line
    flush $stream
}

proc saifreg::register_objects {recursive scope batch_size} {
    variable objects
    variable batches
    set start [clock milliseconds]
    note "ENUM_BEGIN recursive=$recursive scope=$scope"
    if {$recursive} {
        set selected [get_objects -r *]
    } else {
        set selected [get_objects *]
    }
    set count [llength $selected]
    note "ENUM_END count=$count elapsed_ms=[expr {[clock milliseconds]-$start}] scope=$scope"
    for {set offset 0} {$offset<$count} {incr offset $batch_size} {
        set chunk [lrange $selected $offset [expr {$offset+$batch_size-1}]]
        incr batches
        set start [clock milliseconds]
        note "LOG_BEGIN batch=$batches offset=$offset count=[llength $chunk] scope=$scope"
        log_saif $chunk
        incr objects [llength $chunk]
        note "LOG_END batch=$batches total_objects=$objects elapsed_ms=[expr {[clock milliseconds]-$start}]"
    }
}

proc saifreg::walk {scope depth partition_depth batch_size} {
    variable seen
    # Pass the scope handle itself to current_scope: names can contain escaped
    # identifiers and [] generate indices. Never re-interpret them as globs.
    current_scope $scope
    set canonical [current_scope]
    if {[dict exists $seen $canonical]} {error "Duplicate/self scope: $canonical"}
    dict set seen $canonical 1
    if {$depth >= $partition_depth} {
        register_objects 1 $canonical $batch_size
        return
    }
    register_objects 0 $canonical $batch_size
    set start [clock milliseconds]
    note "SCOPES_BEGIN depth=$depth scope=$canonical"
    set children [get_scopes *]
    note "SCOPES_END count=[llength $children] elapsed_ms=[expr {[clock milliseconds]-$start}] scope=$canonical"
    foreach child $children {
        walk $child [expr {$depth+1}] $partition_depth $batch_size
        current_scope $scope
    }
}

proc saifreg::register {root logfile {batch_size 1024} {partition_depth 3}} {
    variable version
    variable stream
    variable objects
    variable batches
    variable seen
    if {![string is integer -strict $batch_size] || $batch_size<1 ||
        ![string is integer -strict $partition_depth] || $partition_depth<0} {
        error "Invalid registration batch/depth"
    }
    set stream [open $logfile {WRONLY CREAT EXCL}]
    set saved [current_scope]
    set objects 0
    set batches 0
    set seen [dict create]
    set started [clock milliseconds]
    set code [catch {
        note "REGISTER_BEGIN version=$version root=$root batch_size=$batch_size partition_depth=$partition_depth"
        walk $root 0 $partition_depth $batch_size
        if {$objects==0} {error "DUT enumeration returned zero objects"}
        note "REGISTER_END scopes=[dict size $seen] objects=$objects batches=$batches elapsed_ms=[expr {[clock milliseconds]-$started}]"
    } message options]
    # Restore scope even on failure, but never convert a registration error to PASS.
    set restore_code [catch {current_scope $saved} restore_message]
    if {$code} {note "REGISTER_FAILED: $message"}
    close $stream
    if {$code} {return -options $options $message}
    if {$restore_code} {error "Scope restore failed: $restore_message"}
    return [dict create version $version scopes [dict size $seen] objects $objects batches $batches]
}
