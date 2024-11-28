# cli_autotuner

this is a simple tool for optimizing linux programs

is uses bash, sed, python, optuna

### how it works

for now, this tool minimizes the program execution time.

it assumes that the files of the optimizable program are in PROGDIR.

the directory will be copied to WORKDIR 

the parameter search space can be defined in a file like test_spec.json

the files for each parameter can be specified in the SPECFILE

optimizable parameters will be replaced with a sed regex.

the CMD specifies how to invoke the optimizable program

### usage

```
Usage: cli_autotuner [OPTIONS] WORKDIR PROGDIR SPECFILE CMD

Options:
  --setup_cmd TEXT    run a setup command before each trial
  --n_trials INTEGER  default=16, number of parameter configuration to try
  --timeout INTEGER   default=108000 (30 hours), time after which a trial will
                      be considered failed
  --overwrite         overwrite existing workdir
  --help              Show this message and exit.
```

### example

```
mkdir progdir
echo 'sleep $((param1 + $RANDOM % 3))' > progdir/program.sh
chmod u+x progdir/program.sh

cli_autotuner workdir progdir test_spec.json ./program.sh
```
