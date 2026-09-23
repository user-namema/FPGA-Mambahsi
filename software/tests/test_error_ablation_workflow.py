"""Small synthetic end-to-end workflow; no scientific accuracy claims."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import train_mambahsi_spatial_split_dense_qat as qat

try:
    from scipy.io import savemat
except ImportError:
    savemat=None


@unittest.skipIf(savemat is None,'scipy required only for synthetic .mat dataset integration')
class WorkflowTests(unittest.TestCase):
    def test_qat_sim_capture_replay_and_qat_dynamics(self):
        self.run_workflow(False)

    def test_d1_shared_u_qat_sim_replay_and_dynamics(self):
        self.run_workflow(True)

    def run_workflow(self, use_d):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);data=root/'data'/'UP';data.mkdir(parents=True)
            fp=root/'fp32';run=fp/'run_seed0';run.mkdir(parents=True)
            rng=np.random.default_rng(1)
            raw=rng.random((48,64,16),dtype=np.float32)
            gt=(np.arange(48*64).reshape(48,64)%9+1).astype(np.int16)
            savemat(data/'PaviaU.mat',{'paviaU':raw});savemat(data/'PaviaU_gt.mat',{'paviaU_gt':gt})
            masks=[]
            for i in range(3):
                m=np.zeros((48,64),bool);m[i*16:(i+1)*16]=True;masks.append(m)
            np.savez(fp/'spatial_split_masks.npz',train_region=masks[0],val_region=masks[1],test_region=masks[2])
            np.savez(run/'sample_indices.npz',train_indices=np.flatnonzero(masks[0]),
                     val_indices=np.flatnonzero(masks[1]),test_indices=np.flatnonzero(masks[2]))
            np.savez(fp/'train_only_preprocess.npz',pca_mean=np.zeros(16),pca_components=np.eye(16),
                     explained_variance=np.ones(16),channel_min=np.zeros(16),channel_max=np.ones(16))
            blocks=[dict(top=i*16,bottom=(i+1)*16,left=j*16,right=(j+1)*16,split=['train','validation','test'][i])
                    for i in range(3) for j in range(4)]
            (fp/'spatial_split.json').write_text(json.dumps(dict(dataset='UP',strategy='blocks',block_size=16,effective_tile_size=16,blocks=blocks)))
            config=dict(qat.DEFAULT_MODEL_CONFIG,in_channels=16,num_classes=9,use_D=use_d)
            (run/'model_config.json').write_text(json.dumps(config))
            torch.manual_seed(18)
            model=qat.build_configured_model(16,9,config)
            torch.save(model.state_dict(),run/'best_model.pth')
            env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',MPLBACKEND='Agg')
            def execute(script,*args):
                result=subprocess.run([sys.executable,script,*map(str,args)],capture_output=True,text=True,env=env)
                if result.returncode:
                    self.fail(result.stdout[-3000:]+'\n'+result.stderr[-4000:])
            stability_args = (['--freeze_bn_epoch','1','--freeze_lsq_epoch','1',
                               '--include_calibrated_baseline','--training_diagnostics',
                               '--lr_schedule','cosine','--eval_interval','1'] if use_d else [])
            execute('train_mambahsi_spatial_split_dense_qat.py','--dataset','UP',
                    '--data_set_path',root/'data','--fp32_dir',fp,'--work_dir',root/'results',
                    '--exp_name','qa','--run_tag','fixture','--device','cpu','--seeds','0',
                    '--max_epoch',('2' if use_d else '1'),'--calibration_steps','1','--batch_size','2','--eval_batch_size',('2' if use_d else '1'),
                    '--dt-input-bits','8','--dt-output-bits','7','--use_D',str(use_d),'--ssm-u-quantization','shared',*stability_args)
            qat_run=next((root/'results').glob('**/run_seed0'))
            training=json.loads((qat_run/'result.json').read_text())
            self.assertTrue(training['best_validation_recheck_match'])
            if use_d:
                self.assertGreaterEqual(training['best_val_score'],training['calibrated_baseline_validation']['mAcc'])
                records=[json.loads(line) for line in (qat_run/'qat_training_diagnostics.jsonl').read_text().splitlines()]
                self.assertEqual([r['epoch'] for r in records],[0,1,2])
                self.assertTrue(all(r['max_abs_parameter_changes']['lsq_scales']==0 for r in records))
                self.assertTrue(all(r['bn_running_max_abs_change']==0 for r in records))
                history=[json.loads(line) for line in (qat_run/'qat_training_history.jsonl').read_text().splitlines()]
                self.assertAlmostEqual(history[-1]['learning_rates']['weights_and_bn'],1e-6)
                # Instrumentation must not change the training trajectory.
                execute('train_mambahsi_spatial_split_dense_qat.py','--dataset','UP',
                        '--data_set_path',root/'data','--fp32_dir',fp,'--work_dir',root/'no_probe',
                        '--exp_name','qa','--run_tag','fixture','--device','cpu','--seeds','0',
                        '--max_epoch','2','--calibration_steps','1','--batch_size','2','--eval_batch_size','2',
                        '--dt-input-bits','8','--dt-output-bits','7','--use_D','true',
                        '--ssm-u-quantization','shared',*[v for v in stability_args if v!='--training_diagnostics'])
                other_run=next((root/'no_probe').glob('**/run_seed0'))
                probed=torch.load(qat_run/'best_qat_foldaware.pth',map_location='cpu')
                plain=torch.load(other_run/'best_qat_foldaware.pth',map_location='cpu')
                self.assertTrue(all(torch.equal(v,plain[k]) for k,v in probed.items()))
                # New instrumentation must also preserve calibration and training RNG.
                execute('train_mambahsi_spatial_split_dense_qat.py','--dataset','UP',
                        '--data_set_path',root/'data','--fp32_dir',fp,'--work_dir',root/'forensic_probe',
                        '--exp_name','qa','--run_tag','fixture','--device','cpu','--seeds','0',
                        '--max_epoch','2','--calibration_steps','1','--batch_size','2','--eval_batch_size','2',
                        '--dt-input-bits','8','--dt-output-bits','7','--use_D','true',
                        '--ssm-u-quantization','shared',*stability_args,
                        '--diagnose_initial_quantization','--capture_oa_drop_pp','0.00001','--capture_start_epoch','1')
                forensic_run=next((root/'forensic_probe').glob('**/run_seed0'))
                forensic_state=torch.load(forensic_run/'best_qat_foldaware.pth',map_location='cpu')
                self.assertTrue(all(torch.equal(v,forensic_state[k]) for k,v in plain.items()))
                diagnosis=json.loads((forensic_run/'initial_quantization_diagnosis/initial_quantization.json').read_text())
                self.assertTrue(diagnosis['model_state_unchanged'])
                self.assertTrue(diagnosis['conversion_within_tolerance'])
                self.assertEqual(diagnosis['cases']['all_repeat']['output_vs_all']['changed'],0)
                self.assertEqual(len(diagnosis['cases']),15)
                execute('train_mambahsi_spatial_split_dense_qat.py','--dataset','UP',
                        '--data_set_path',root/'data','--fp32_dir',fp,'--work_dir',root/'initial_only',
                        '--exp_name','qa','--run_tag','fixture','--device','cpu','--seeds','0',
                        '--calibration_steps','1','--batch_size','2','--eval_batch_size','2',
                        '--use_D','true','--initial_diagnostics_only')
                self.assertFalse(list((root/'initial_only').rglob('best_qat_foldaware.pth')))
                self.assertFalse(list((root/'initial_only').rglob('result.json')))
                self.assertEqual(len(list((root/'initial_only').rglob('initial_quantization.json'))),1)
                execute('train_mambahsi_spatial_split_dense_qat.py','--dataset','UP',
                        '--data_set_path',root/'data','--fp32_dir',fp,'--work_dir',root/'weight_init',
                        '--exp_name','qa','--run_tag','fixture','--device','cpu','--seeds','0',
                        '--calibration_steps','1','--batch_size','2','--eval_batch_size','2',
                        '--use_D','true','--weight_init_validation_only')
                init_path=next((root/'weight_init').rglob('weight_init_validation.json'))
                init=json.loads(init_path.read_text())
                self.assertTrue(init['model_state_unchanged'])
                self.assertEqual(list(init['cases']),['mean','patch_max','all_weight_max'])
                self.assertEqual([len(v['changed_scales']) for v in init['cases'].values()],[0,1,33])
                self.assertEqual(init['cases']['mean']['validation'],
                    json.loads(next((root/'initial_only').rglob('initial_quantization.json')).read_text())['cases']['all']['validation'])
                self.assertFalse(list((root/'weight_init').rglob('best_qat_foldaware.pth')))
                self.assertFalse(list((root/'weight_init').rglob('qat_training_history.jsonl')))
                self.assertFalse(list((root/'weight_init').rglob('result.json')))
                execute('run_max_init_qat_fpga.py','--fp32-dir',fp,'--data-path',root/'data',
                        '--output-dir',root/'max_pipeline','--device','cpu','--seeds','0',
                        '--max-epoch','1','--calibration-steps','1','--batch-size','2','--eval-batch-size','2')
                pipeline=json.loads((root/'max_pipeline/pipeline_results.json').read_text())
                self.assertEqual(len(pipeline),4)
                timestamps={str(p):p.stat().st_mtime_ns for p in (root/'max_pipeline').rglob('*.pth')}
                execute('run_max_init_qat_fpga.py','--fp32-dir',fp,'--data-path',root/'data',
                        '--output-dir',root/'max_pipeline','--device','cpu','--seeds','0',
                        '--max-epoch','1','--calibration-steps','1','--batch-size','2','--eval-batch-size','2','--resume')
                self.assertEqual(timestamps,{str(p):p.stat().st_mtime_ns for p in (root/'max_pipeline').rglob('*.pth')})
                self.assertEqual({(r['stage'],r['policy']) for r in pipeline},
                    {(s,p) for s in ['calibrated','qat'] for p in ['patch_max','all_weight_max']})
                for row in pipeline:
                    self.assertGreaterEqual(row['best_val_mAcc'],row['initial_validation']['mAcc'])
                    self.assertTrue(Path(row['fpga_result_path']).exists())
                    case=init['cases'][row['policy']]['validation']
                    self.assertEqual(row['initial_validation'],case)
                    if row['stage']=='calibrated':
                        self.assertEqual(row['optimization_steps'],0)
                        self.assertEqual(row['best_epoch'],0)
                        self.assertFalse(Path(row['model_result']).with_name('qat_train_loss_curve.png').exists())
                    else:
                        self.assertGreater(row['optimization_steps'],0)
                execute('run_qat_stability_sweep.py','--fp32-dir',fp,'--data-path',root/'data',
                        '--dataset','UP','--device','cpu','--seeds','0','--variants','bn1_fixed_scales',
                        '--max-epoch','1','--calibration-steps','1','--batch-size','2','--eval-batch-size','2',
                        '--output-dir',root/'stability_runner')
                runner=json.loads((root/'stability_runner/stability_manifest.json').read_text())
                self.assertEqual(runner['jobs'][0]['status'],'complete')
                self.assertTrue((root/'stability_runner/validation_summary.csv').exists())
            numeric=json.loads((qat_run/'numeric_config.json').read_text())
            self.assertEqual(numeric['dt_output_bits'],7)
            execute('both_FPGA_single_qat_source.py','--qat-run-dir',qat_run,'--dataset','UP',
                    '--data-path',root/'data','--fp32-dir',fp,'--device','cpu','--output-dir',root/'sim',
                    '--ssm-state-bits','28','--ssm-state-fraction-bits','20',
                    '--report-ssm-errors','--capture-ssm-inputs','--ssm-capture-tiles','2')
            result=json.loads((root/'sim'/'fpga_simulation_result.json').read_text())
            self.assertEqual(result['numeric_config']['dt_output_bits'],7)
            self.assertFalse(result['rtl_baseline_compatible'])
            self.assertEqual(result['model_config']['use_D'],use_d)
            self.assertEqual(result['qat_ssm_contract']['u_quantization'],'shared')
            self.assertEqual(result['d_path_rom_bits'],6480 if use_d else 0)
            manifest=json.loads((root/'sim'/'replay_inputs_manifest.json').read_text())
            self.assertEqual(len(manifest['tiles']),2)
            self.assertTrue(all(t['block']['split']=='test' for t in manifest['tiles']))
            execute('run_ssm_error_ablation.py','--inputs-glob',root/'sim/replay_tiles/*/ssm_replay_inputs/*.npz',
                    '--output-dir',root/'replay','--suite','sources')
            self.assertTrue((root/'replay'/'none_error_by_position.csv').exists())
            if use_d: self.assertTrue((root/'replay'/'d-only_error_by_position.csv').exists())
            execute('analyze_alog_dynamics.py','--run-dir',qat_run,'--artifact-dir',fp,
                    '--data-path',root/'data','--device','cpu','--max-tiles','2',
                    '--no-plots','--output-dir',root/'dynamics')
            summary=json.loads((root/'dynamics'/'analysis_summary.json').read_text())
            self.assertEqual(summary['checkpoint_kind'],'QAT')
            self.assertEqual(summary['analyzed_tiles'],2)
            self.assertTrue((root/'dynamics'/'discrete_retention_stats.csv').exists())
            execute('run_fpga_error_sweep.py','--qat-run-dir',qat_run,'--data-path',root/'data',
                    '--output-dir',root/'sweep_plan','--dry-run')
            planned=json.loads((root/'sweep_plan'/'sweep_manifest.json').read_text())
            self.assertEqual(len(planned['jobs']),7 if use_d else 6)
            execute('run_fpga_error_sweep.py','--qat-run-dir',qat_run,'--data-path',root/'data',
                    '--fp32-dir',fp,'--output-dir',root/'sweep_actual','--suite','sources')
            completed=json.loads((root/'sweep_actual'/'sweep_manifest.json').read_text())
            self.assertTrue(all(j.get('completed') for j in completed['jobs']))
            self.assertTrue((root/'sweep_actual'/'sweep_metrics.csv').exists())
            if use_d:
                execute('diagnose_qat_batch.py','--qat-run-dir',qat_run,'--fp32-dir',fp,
                        '--data-path',root/'data','--device','cpu','--output-dir',root/'batch_trace',
                        '--modes','native','reference-tf32-off','--save-traces')
                diagnosis=json.loads((root/'batch_trace/batch_diagnosis.json').read_text())
                self.assertTrue(diagnosis['model_state_unchanged'])
                self.assertEqual(diagnosis['actual_batch_size'],2)
                self.assertIsNone(diagnosis['modes']['native']['batch_repeat']['first_exact'])
                # Deliberately altered synthetic prediction fixture tests auto-selection,
                # not a claim that this tiny model has real batch sensitivity.
                fixture=root/'prediction_selection_fixture';fixture.mkdir()
                (fixture/'fpga_simulation_result.json').write_text(json.dumps(result))
                saved_prediction=np.load(root/'sim/qat_saved_batch_reproduced_prediction.npy')
                one=saved_prediction.copy();one.ravel()[np.flatnonzero(masks[2])[0]]=(int(one.ravel()[np.flatnonzero(masks[2])[0]])+1)%9
                np.save(fixture/'qat_saved_batch_reproduced_prediction.npy',saved_prediction)
                np.save(fixture/'qat_direct_prediction.npy',one)
                execute('diagnose_qat_batch.py','--qat-run-dir',qat_run,'--fp32-dir',fp,
                        '--data-path',root/'data','--device','cpu','--output-dir',root/'batch_auto',
                        '--prediction-dir',fixture,'--modes','native')
                automatic=json.loads((root/'batch_auto/batch_diagnosis.json').read_text())
                self.assertEqual((automatic['group_index'],automatic['target_index']),(0,0))
                self.assertTrue(automatic['selection'].startswith('first_mismatched'))
                execute('run_fpga_error_sweep.py','--qat-run-dir',qat_run,'--fp32-dir',fp,
                        '--data-path',root/'data','--device','cpu','--suite','requant','--output-dir',root/'requant')
                pairs=json.loads((root/'requant/requant_comparison.json').read_text())
                self.assertEqual(len(pairs),1)
                planned=json.loads((root/'requant/sweep_manifest.json').read_text())
                hw=Path(next(j['output'] for j in planned['jobs'] if j['variant']=='all_hardware'))
                old=Path(next(j['output'] for j in completed['jobs'] if j['variant']=='all'))
                self.assertTrue(np.array_equal(np.load(hw/'int8_hw_prediction.npy'),np.load(old/'int8_hw_prediction.npy')))
                execute('run_ssm_error_ablation.py','--inputs-glob',root/'sim/replay_tiles/*/ssm_replay_inputs/*.npz',
                        '--suite','k-precision','--k-bits-grid','19','21','--k-fraction-grid','24','26',
                        '--output-dir',root/'k_replay')
                k=json.loads((root/'k_replay/replay_manifest.json').read_text())
                self.assertEqual(len(k['configurations']),4)
                self.assertTrue(all('k_fraction_bits' in r for r in k['range_certificates']))
                execute('run_fpga_error_sweep.py','--qat-run-dir',qat_run,'--fp32-dir',fp,
                        '--data-path',root/'data','--device','cpu','--suite','k-precision',
                        '--k-bits-grid','19','21','--k-fraction-grid','26','--output-dir',root/'k_sweep')
                k=json.loads((root/'k_sweep/sweep_manifest.json').read_text())
                self.assertEqual(len(k['jobs']),2)
                self.assertTrue(all(j['completed'] for j in k['jobs']))


if __name__=='__main__': unittest.main()
