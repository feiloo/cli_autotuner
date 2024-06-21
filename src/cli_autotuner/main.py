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

def run_trial(cmd, outpath, tmpath, trial_number):
    '''
    podman run --rm -ti \
            --mount type=bind,src=wgs_bam/,dst=/root/wgs_bam \
            --mount type=bind,src=wgs_bam_dedup_test/,dst=/root/wgs_bam_dedup_test \
            quay.io/biocontainers/gatk4:4.4.0.0--py36hdfd78af_0 \
            /bin/bash -c 'mkdir -p container_trial_path_var && cd container_trial_path_var && chmod u+x script.sh && ./script.sh'
    '''

    #cmd = '''mkdir -p container_trial_path_var && cd container_trial_path_var && chmod u+x script.sh && ./script.sh'''

    opath = 'wgs_bam_dedup_test/' + str(trial_number)
    outpath = Path(outpath)

    st = time()
    timeout = 60*60*30 # 30h
    #proc = subprocess.run(cmd, text=True, shell=True, capture_output=True, timeout=timeout)
    (outpath / 'stdout.txt').touch()
    (outpath / 'stderr.txt').touch()
    (outpath / 'res_log.txt').touch()


    with (outpath / 'stdout.txt').open('r+') as sdo, (outpath / 'stderr.txt').open('r+') as sde, (outpath / 'res_log.txt').open('r+') as log:
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


'''
type in 
int 
float 
categorical
create_param(name, type_, regex, files=None, param_args)
param_args = [low, high, step], [category1, category2]

name,int,lbalb,[f1,f2,f3],[

{'param_type': [int,float, categorical]
    'param_name': 
    'regex':
    'files':
    'args'

'''

def make_param(trial, p_spec: dict):
    pt = p_spec['param_type']
    pn = p_spec['param_name']
    print(p_spec)
    regex = p_spec['regex']
    assert isinstance(pn, str)

    p_args = p_spec['args']
    if pt == 'int':
        low = p_args['low']
        high = p_args['high']
        step = p_args['step']
        param = trial.suggest_int(pn, low=low, high=high, step=step)
    elif pt == 'float':
        low = p_args['low']
        high = p_args['high']
        step = p_args['step']
        param = trial.suggest_float(pn, low=low, high=high, step=step)
    elif pt == 'categorical':
        categories = p_args
        param = trial.suggest_categorical(pn, choices=p_args)
    else:
        raise RuntimeError()
    files = [Path(f) for f in p_spec['files']]
    return [param, regex, files]


def get_params(trial, spec):
    params = {}
    for p_spec in spec['params']:
        params[p_spec['param_name']] = make_param(trial, p_spec)

    return params

def get_search_space(spec):
    search_space = {}
    for p in spec['params']:
        p_args = p['args']
        if p['param_type'] in ['int','float']:
            grid_steps = list(range(p_args['low'], p_args['high'], p_args['step']))
        else:
            grid_steps = p_args
        search_space[p['param_name']] = grid_steps


    return search_space

def gen_param_cmd(progdir, param):
    param_val = param[0]
    regex = param[1]
    files = param[2]

    if files == []:
        files = list(Path(progdir).iterdir())

    print(f'sed -i s/{regex}/{param_val} {files[0]}')
    
    

def prepare_trial(progdir, workdir, hparams, trial_number):
    # trial path
    tpath = Path(workdir) / str(trial_number)
    if tpath.exists():
        raise RuntimeError()

    tpath.mkdir(parents=True)
    tmpath = tpath / 'tmp'
    outpath = tpath / 'out'
    if tmpath.exists():
        shutil.rmtree(str(tmpath))
    if outpath.exists():
        shutil.rmtree(str(outpath))
    outpath.mkdir()
    tmpath.mkdir()

    '''
    # cleanup 
    for sp in stp.iterdir():
        if (sp / 'tmp').exists():
            shutil.rmtree(str(sp / 'tmp'))
        if (sp / 'out').exists():
            shutil.rmtree(str(sp / 'out'))
    '''

    global script_templ
    script = script_templ
    with (tpath/'script.sh').open('w') as f:
            f.write(script)

    shutil.copytree(progdir, tmpath, dirs_exist_ok=True)
    for param_name, trip in hparams.items():
        gen_param_cmd(progdir, trip)

    return outpath, tmpath


def objective_fn(trial,spec, cmd):
    workdir = './workdir'
    progdir = './progdir'

    hparams = get_params(trial, spec)
    outpath, tmpath = prepare_trial(progdir, workdir, hparams, trial.number) 
    try:
        tim = run_trial(cmd, outpath, tmpath, trial.number)
    except KeyboardInterrupt:
        trial.study.stop()
        raise KeyboardInterrupt()
    except Exception as e:
        print(e)
        tim = 99999999

    return tim


# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "study_name"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"

test_spec = {
        'study_name':'study1',
        'sampler':'s1',
        'pruner':'p1',
        'params':[
            {'param_name':'param1',
                'param_type': 'int',
                'regex':'r',
                'files':[],
                'args':{'low':1, 'high':2,'step':1}
                },
            {'param_name':'param2',
                'param_type': 'categorical',
                'regex':'r',
                'files':[],
                'args':['cat1', 'cat2'],
                }
            ]
        }

spec = test_spec
search_space = get_search_space(spec)
print(search_space)
sampler = optuna.samplers.GridSampler(search_space=search_space)
#sampler=optuna.samplers.BruteForceSampler(), 

study = optuna.create_study(
    study_name=spec['study_name'], 
    storage=storage_name, 
    sampler=sampler,
    load_if_exists=True, 
    direction='minimize')

#study.optimize(objective, n_trials=30)
cmd = 'echo hello'

def objective(trial):
    return objective_fn(trial, spec, cmd)

study.optimize(objective)
df = study.trials_dataframe()
print(df)

# optimize
# optimize --continue

# workdir/trial

# workdir/trial/score.txt

workdir = './workdir'
progdir = './progdir'

def cli():
    pass

if __name__ == '__main__':
    cli()
