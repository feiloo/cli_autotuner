import logging
import sys
from pathlib import Path
import subprocess
import shutil
from time import time, sleep

import optuna
import signal
import psutil
import json


script_templ = r'''#!/bin/bash
set -euo pipefail

mkdir -p out
mkdir -p tmp
workflow_tuner/build_gatk/gatk/gatk MarkDuplicatesSpark \
    --input input_bam_var \
    --java-options "-Djava.io.tmpdir=tmp -Xmsjava_memory_varG -Xmxjava_memory_varG" \
    --conf 'spark.local.dir=tmp' \
    --conf 'spark.executor.cores=cpus_number_var' \
    --conf 'spark.executor.memory=executor_memory_varg' \
    --conf 'spark.driver.memory=driver_memory_varg' \
    --tmp-dir tmp \
    --spark-master local[cpus_number_var] \
    --output output_target_var
'''
def tup_to_dic(x):
    return dict(zip(x._fields, x))

def join_dicts(dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            assert k not in res.keys()
            res[k] = v
    return res

def get_metrics(psproc):
    c = psproc
    res = [
        {'time': time()},
        {'pid': c.pid},
        {'ppid': c.ppid()},
        {'cpu_percent': c.cpu_percent(interval=1.)}, # float
        {'comandline': c.cmdline()}, # list
    ]
    res += [ 
        tup_to_dic(x) for x in [
            c.memory_info(), # tuple
            c.cpu_times(), # tuple
            c.io_counters(), # tuple
            c.num_ctx_switches(), # tuple
            ]
    ]
    
    metrics = join_dicts(res)
    return metrics

def run_workflow(trial_number):

    #cmd = 'NEXTFLOW_MODULES=modules/ nextflow run modules/ --samplesheet samplesheet.csv -c nextflow_conf.config --workflow_variation sequence_alignment --bwa_tool bwa -resume'

    cmd = '''
    podman run --rm -ti \
            --mount type=bind,src=wgs_bam/,dst=/root/wgs_bam \
            --mount type=bind,src=wgs_bam_dedup_test/,dst=/root/wgs_bam_dedup_test \
            quay.io/biocontainers/gatk4:4.4.0.0--py36hdfd78af_0 \
            /bin/bash -c 'mkdir -p container_trial_path_var && cd container_trial_path_var && chmod u+x script.sh && ./script.sh'
    '''

    cmd = '''mkdir -p container_trial_path_var && cd container_trial_path_var && chmod u+x script.sh && ./script.sh
    '''

    container_trial_path = 'wgs_bam_dedup_test/' + str(trial_number)
    opath = 'wgs_bam_dedup_test/' + str(trial_number)
    cmd = cmd.replace('container_trial_path_var', container_trial_path)

    st = time()
    timeout = 60*60*30 # 30h
    #proc = subprocess.run(cmd, text=True, shell=True, capture_output=True, timeout=timeout)
    (Path(opath) / 'stdout.txt').touch()
    (Path(opath) / 'stderr.txt').touch()
    (Path(opath) / 'res_log.txt').touch()


    with (Path(opath) / 'stdout.txt').open('r+') as sdo, (Path(opath) / 'stderr.txt').open('r+') as sde, (Path(opath) / 'res_log.txt').open('r+') as log:
        proc = subprocess.Popen(['/bin/bash', '-c', cmd], text=True, stderr=sde, stdout=sdo)
        pid = proc.pid
        psup = psutil.Process(pid=pid)

        while proc.poll() is None:
            try:
                sleep(1)
                chil = psup.children(recursive=True) 
                
                #res_log = {'childs':}
                for c in chil:
                    m = get_metrics(c)
                    log.write(json.dumps(m))
                
                #print([x.cpu_percent(interval=1.) for x in chil])
                #print([x.memory_info() for x in chil])
                #out, err = proc.communicate()

                '''
                # offset 100 from end of file
                sde.seek(max(sde.tell() - 100, 0))
                sdo.seek(max(sdo.tell() - 100, 0))
                print(str(sde.read()))#.decode('utf-8')), file=sys.stderr)
                print(str(sdo.read()))#.decode('utf-8')), file=sys.stderr)

                #sdo.write(out)
                #sde.write(err)
                '''

                '''
                except KeyboardInterrupt:
                    proc.send_signal(signal.SIGINT)
                    proc.wait()
                    #proc.kill()
                    out, err = proc.communicate()

                    #print(str(out), file=sys.stdout)
                    #print(str(err), file=sys.stderr)

                    sdo.write(out)
                    sde.write(err)
                '''
            except KeyboardInterrupt:
                proc.send_signal(signal.SIGINT)
                proc.wait()
                proc.kill()
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                print(e)

        proc.kill()
    #proc.wait()

    #proc.check_returncode()
    return time() - st

def get_hparams(trial):
    '''
    emem = '50'
    dmem = '4'
    jmem = '56'
    '''
    emem = str(trial.suggest_int('emem', low=16, high=50, step=8))
    dmem = str(trial.suggest_int('dmem', low=2, high=16, step=7))
    jmem = str(trial.suggest_int('jmem', low=16, high=56, step=8))

    return {'emem':emem, 'dmem':dmem, 'jmem':jmem}



def gen_script(hparams, trial_number):
    input_bam_path = '00000-24_T_1.bam'

    target_path = 'wgs_bam_dedup_test/'
    cpus = '12'

    global script_templ 

    emem = hparams['emem']
    dmem = hparams['dmem']
    jmem = hparams['jmem']
    

    script = script_templ
    script = script.replace('input_bam_var', input_bam_path)
    script = script.replace('cpus_number_var', cpus)
    script = script.replace('output_target_var', target_path + 'out.bam')
    script = script.replace('executor_memory_var', emem)
    script = script.replace('driver_memory_var', dmem)
    script = script.replace('java_memory_var', jmem)
    
    stp = Path('wgs_bam_dedup_test/')
    spath = Path('wgs_bam_dedup_test/') / str(trial_number)
    # cleanup 
    for sp in stp.iterdir():
        if (sp / 'tmp').exists():
            shutil.rmtree(str(sp / 'tmp'))
        if (sp / 'out').exists():
            shutil.rmtree(str(sp / 'out'))

    spath.mkdir()

    with (spath/'script.sh').open('w') as f:
            f.write(script)

    print(script)


def objective(trial):
    hparams = get_hparams(trial)
    gen_script(hparams, trial.number) 
    try:
        tim = run_workflow(trial.number)
    except KeyboardInterrupt:
        trial.study.stop()
        raise KeyboardInterrupt()
    except Exception as e:
        print(e)
        tim = 99999999

    return tim


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "workflow_tune_2"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
search_space = {
        'emem': [12,24,32,48],
        'dmem': [2,6,12],
        'jmem': [16,24,32,48],
        }

study = optuna.create_study(
    study_name=study_name, storage=storage_name, 
    #sampler=optuna.samplers.BruteForceSampler(), 
    sampler=optuna.samplers.GridSampler(search_space=search_space),
    load_if_exists=True, direction='minimize')

#study.optimize(objective, n_trials=30)
#study.optimize(objective, n_trials=None)#30)
study.optimize(objective)
df = study.trials_dataframe()
print(df)
