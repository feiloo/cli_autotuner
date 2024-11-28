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
import os

import click


script_templ = r'''#!/bin/bash
set -euo pipefail
'''

def tup_to_dic(x):
    ''' tuple to dictionary '''
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



def run_trial(cmd, outpath, tmpath, trial_number, trial, timeout):
    outpath = Path(outpath)

    st = time()
    #proc = subprocess.run(cmd, text=True, shell=True, capture_output=True, timeout=timeout)
    (outpath / 'stdout.txt').touch()
    (outpath / 'stderr.txt').touch()
    (outpath / 'res_log.txt').touch()


    with (outpath / 'stdout.txt').open('r+') as sdo, \
        (outpath / 'stderr.txt').open('r+') as sde, \
        (outpath / 'res_log.txt').open('r+') as log:

        proc = subprocess.Popen(['/bin/bash', '-c', cmd], text=True, stderr=sde, stdout=sdo, cwd=str(tmpath))
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
                trial.study.stop()
            except subprocess.TimeoutExpired:
                pass
            # catch exceptions on psutil subprocess analysis
            except Exception as e:
                print(e)

        if proc.returncode != 0:
            raise RuntimeError("None zero exit-code")

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

def list_files(tmpath):
    fi = []
    for dirpath, dirnames, filenames in os.walk(str(tmpath)):
        for f in filenames:
            pf = (Path(dirpath) / f)
            if pf.is_file():
                relpath = Path('tmp') / str(pf).removeprefix(str(tmpath)+'/')
                fi.append(str(relpath))
    return fi

def gen_param_cmd(tmpath, param):
    param_val = param[0]
    regex = param[1]
    files = param[2]

    if files == []:
        files = list_files(tmpath)

    files_s = ' '.join(files)
    cmd = f"sed -i -e 's/{regex}/{param_val}/g' {files_s}"
    return cmd
    
    

def prepare_trial(progdir, workdir, hparams, trial_number, setup_cmd):
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

    shutil.copytree(progdir, tmpath, dirs_exist_ok=True)
    cmds = []
    for param_name, trip in hparams.items():
        cmds.append(gen_param_cmd(tmpath, trip))

    global script_templ
    script = script_templ
    script += '\n'.join(cmds)

    with (tpath/'script.sh').open('w') as f:
            f.write(script)

    cmd = f'cd {tpath} && chmod u+x ./script.sh && ./script.sh'
    proc = subprocess.run(cmd, text=True, shell=True, capture_output=True, timeout=60)
    proc.check_returncode()

    if setup_cmd is not None:
        proc = subprocess.run(setup_cmd, text=True, shell=True, capture_output=True, timeout=60, cwd=tmpath)
        proc.check_returncode()

    return outpath, tmpath


def objective_fn(trial, spec, cmd, setup_cmd, timeout):
    workdir = './workdir'
    progdir = './progdir'

    print(f'starting trial: {trial.number}\n')
    hparams = get_params(trial, spec)
    print(f'''running trial with params: 
          {hparams}
          ''')
    outpath, tmpath = prepare_trial(progdir, workdir, hparams, trial.number, setup_cmd) 
    try:
        tim = run_trial(cmd, outpath, tmpath, trial.number, trial, timeout)
        # write score to workdir, for ease-of-use
        with (Path(outpath) / 'score.txt').open('w') as f:
            f.write(str(tim))
        return tim
    except KeyboardInterrupt:
        trial.study.stop()
        raise KeyboardInterrupt()
    #except Exception as e:
        #print(e)
        #raise e



def tune_program(progdir, workdir, specfile, cmd, setup_cmd, n_trials, overwrite, timeout):
    if overwrite and Path(workdir).exists():
        shutil.rmtree(str(Path(workdir)))

    Path(workdir).mkdir()

    # Add stream handler of stdout to show the messages
    lg = optuna.logging.get_logger("optuna")
    # remove default logging handler
    lg.handlers = []
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "study_name"  # Unique identifier of the study.
    db_path = str(Path(workdir) / study_name)
    storage_name = f"sqlite:///{db_path}.db"

    print(f'''tuning study can be viewed with optuna-dashboard: 
    optuna-dashboard {storage_name}
    ''')

    with Path(specfile).open('r') as f:
        spec = json.loads(f.read())

    search_space = get_search_space(spec)
    print(f'''tuning on search space:
    {search_space}''')

    sampler_name = spec['sampler']
    pruner_name = spec['pruner']
    if pruner_name is not None:
        raise NotImplementedError('pruners arent implemented yet')

    seed = spec['seed']

    # for guidance on what sampler/pruner, see: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
    if sampler_name == 'random':
        sampler = optuna.samplers.RandomSampler(seed)
    elif sampler_name == 'grid':
        sampler = optuna.samplers.GridSampler(search_space=search_space, seed=seed)
    elif sampler_name == 'tpe':
        sampler = optuna.samplers.TPESamper(seed=seed)
    else: 
        raise NotImplementedError("sampler is not implemented")

    study = optuna.create_study(
        study_name=spec['study_name'], 
        storage=storage_name, 
        sampler=sampler,
        load_if_exists=True, 
        direction='minimize')

    def objective(trial):
        return objective_fn(trial, spec, cmd, setup_cmd, timeout)

    study.optimize(objective,n_trials=n_trials, catch=[Exception])
    df = study.trials_dataframe()
    print(f'''results-table:
    {df}
    ''')
    print(f'''best params:
    {study.best_params}
    achieved a score of:
    {study.best_value}
    ''')



#@click.group()
@click.command()
@click.argument('workdir', type=click.Path(exists=False, dir_okay=True, path_type=Path))
@click.argument('progdir', type=click.Path(exists=True, dir_okay=True, path_type=Path))
@click.argument('specfile', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument('cmd', type=str)
#@click.option('--dev', is_flag=True, default=False)
@click.option('--setup_cmd', default=None, type=str, help='run a setup command before each trial')
@click.option('--n_trials', default=16, type=int, help='default=16, number of parameter configuration to try')
@click.option('--timeout', default=30*60*60, type=int, help='default=108000 (30 hours), time after which a trial will be considered failed')
@click.option('--overwrite', is_flag=True, help='overwrite existing workdir')
#@click.pass_context
def cli(workdir, progdir, specfile, cmd, setup_cmd,n_trials, timeout, overwrite):
    tune_program(progdir, workdir, specfile, cmd, setup_cmd,n_trials, timeout, overwrite)
         

'''
if __name__ == '__main__':
    cli()
'''
