"""Host-only Tcl mocks. Does NOT launch Vivado, XSim, synthesis or routing."""
import ast
import tempfile
import tkinter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def interpreter(directory, prefix='/tb'):
    t = tkinter.Tcl()
    t.eval('''
        set where /tb
        set logged {}
        set injected ""
        set events {}
        proc current_scope {args} {
            if {[llength $args]} {set ::where [lindex $args 0]}
            return $::where
        }
        proc descend {scope} {
            set result [dict get $::local_objects $scope]
            foreach child [dict get $::children $scope] {
                set result [concat $result [descend $child]]
            }
            return $result
        }
        proc get_objects {args} {
            if {$::injected eq "enum"} {error "INJECTED_ENUM_FAILURE"}
            if {[lindex $args 0] eq "-r"} {return [descend $::where]}
            return [dict get $::local_objects $::where]
        }
        proc get_scopes {pattern} {return [dict get $::children $::where]}
        proc log_saif {chunk} {
            if {$::injected eq "log"} {error "INJECTED_LOG_FAILURE"}
            if {[llength $chunk]>$::allowed_batch} {error "Oversize batch"}
            set ::logged [concat $::logged $chunk]
        }
    ''')
    graph = {
        '/tb/dut': ('root a', 'root b'),
        '/tb/dut/u_network': ('network a',),
        '/tb/dut/u_network/u_features': ('feature a',),
        '/tb/dut/u_network/u_features/blk0': ('block0 a', 'block0 b'),
        '/tb/dut/u_network/u_features/blk0/\\G[1].u ': ('leaf a',),
        '/tb/dut/u_head': ('head a', 'head b'),
    }
    graph = {key.replace('/tb/', prefix + '/', 1): value for key, value in graph.items()}
    scopes = list(graph)
    tree = {s: [] for s in scopes}
    for s in scopes[1:]:
        tree[s.rsplit('/', 1)[0]].append(s)
    for variable, data in [('local_objects', graph), ('children', tree)]:
        pairs = []
        for key, value in data.items():
            pairs.extend((key, tuple(value)))
        t.setvar(variable, t.call('dict', 'create', *pairs))
    t.setvar('allowed_batch', 1024)
    t.call('source', (ROOT / 'saif_register.tcl').as_posix())
    return t, tuple(x for values in graph.values() for x in values)


def register_tests(directory):
    for depth in (0, 1, 3, 8):
        t, expected = interpreter(directory)
        t.setvar('allowed_batch', 3)
        result = t.call('saifreg::register', '/tb/dut',
                        (directory / f'depth{depth}.log').as_posix(), 3, depth)
        assert sorted(t.splitlist(t.getvar('logged'))) == sorted(expected)
        assert int(t.call('dict', 'get', result, 'objects')) == len(expected)
        assert t.getvar('where') == '/tb'
    for failure in ('enum', 'log'):
        t, _ = interpreter(directory)
        t.setvar('injected', failure)
        try:
            t.call('saifreg::register', '/tb/dut',
                   (directory / f'fail{failure}.log').as_posix())
        except tkinter.TclError as exc:
            assert 'INJECTED_' in str(exc)
        else:
            raise AssertionError('Registration failure was hidden')
        assert t.getvar('where') == '/tb'


def protocol_test(directory, probe):
    directory.mkdir()
    t, _ = interpreter(directory, '/tb_nf')
    for name, value in {
        'nf_out': directory.as_posix(), 'output': directory.as_posix(),
        'driver_dir': ROOT.as_posix(), 'nf_package': directory.as_posix(),
        'nf_source_hash': 'test-hash', 'nf_period_ns': 10,
        'action': 'probe' if probe else 'capture',
    }.items():
        t.setvar(name, value)
    t.eval(r'''
        proc nf_load_build {} {}
        proc create_fileset {args} {}
        proc current_fileset {args} {}
        proc get_filesets {args} {return test_fs}
        proc set_property {args} {}
        proc get_property {args} {return {NF_METHOD=3 MAMBA_D_PATH_D1}}
        proc add_files {args} {}
        proc update_compile_order {args} {}
        proc close_sim {args} {}
        proc write_new {path text} {
            set f [open $path {WRONLY CREAT EXCL}]; puts $f $text; close $f
        }
        proc file_text {path} {
            if {[file exists $path]} {
                set f [open $path r]; set s [read $f]; close $f; return $s
            }
            if {[file tail $path] eq "tiles.mem"} {return [join [lrepeat 4096 00] \n]}
            return [join [lrepeat 2304 00] \n]
        }
        proc launch_simulation {args} {file mkdir $::nf_simdir; set ::elapsed 0}
        proc run {duration} {
            lappend ::events [list run $duration]
            incr ::elapsed [string range $duration 0 end-2]
            if {$::elapsed>=680000} {
                write_new $::marker "PASS independent method-specific software logits; tiles=16 bytes=2304 II=4096"
            }
        }
        proc open_saif {path} {
            lappend ::events [list open $::elapsed]
            set ::saifpath $path
        }
        proc close_saif {} {
            lappend ::events [list close $::elapsed]
            write_new $::saifpath [string repeat x 101]
        }
    ''')
    t.call('source', (ROOT / 'sim_capture_v2.tcl').as_posix())
    events = [t.splitlist(x) for x in t.splitlist(t.getvar('events'))]
    if probe:
        assert events == [('open', 0), ('run', '100ns'), ('close', 100)], events
        assert not (directory / 'CAPTURE_OK.tcl').exists()
    else:
        assert ('open', 92160) in events, events
        assert ('close', 419840) in events, events
        assert sum(1 for event in events if event == ('run', '40960ns')) == 8
        assert t.getvar('nf_activity_hash') == 'test-hash'


def main():
    t = tkinter.Tcl()
    for p in ROOT.glob('*.py'):
        ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
    for p in ROOT.glob('*.tcl'):
        assert int(t.call('info', 'complete', p.read_text(encoding='utf-8'))) == 1, p
    with tempfile.TemporaryDirectory(prefix='saif_v2_mock_') as name:
        directory = Path(name)
        register_tests(directory)
        protocol_test(directory / 'probe', True)
        protocol_test(directory / 'capture', False)
    print('PASS HOST MOCKS: disjoint full enumeration at 4 depths; escaped scope; batch bound;')
    print('error propagation/scope restoration; probe=100ns; capture=92160..419840ns; golden drain.')
    print('No Vivado or FPGA simulation was run. Actual registration speed/coverage remain unverified.')


if __name__ == '__main__':
    main()
