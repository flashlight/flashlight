import pathlib
import subprocess
import os
import stat
import shutil

############## Global Variable ##############
USERNAME = os.environ.get('USER')
LOG_ROOT = f'/checkpoint/{USERNAME}/fl_experiments/'


############## Templates ##############
SBATCH_TEMPLATE = """#!/bin/bash

{slrum_params}

# 1. Load modules
module purge
module load cuda/11.0
module load cudnn/v8.0.3.33-cuda.11.0
module load NCCL/2.7.8-1-cuda.11.0
module load intel/mkl/2020.3.279
module load kenlm/010421/gcc.9.3.0

# 2. Signal handler
trap_handler () {{
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
       # exit 0
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_JOB_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}}

trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

# 3. Your job
/usr/mpi/gcc/openmpi-4.0.4rc3/bin/orterun {job_script}

# 4. sleep 60 days in the background
sleep 5184000 &
wait $!
"""

JOB_TEMPLATE = """#!/bin/bash
{binary} {flags}
"""


############## Run ##############
def _format_extra_flags(extra):
    flags = extra.strip().split()
    res = ""
    for flag in flags:
        if len(flag) == 0:
            continue
        sp = flag.split('=')
        k = sp[0][2:]
        v = sp[1]
        res += f"_{k}{v}"
    return res


def _format_slrum_params(slrum_params):
    res = []
    for k, v in slrum_params.items():
        res.append(f"#SBATCH --{k}={v}")
    return '\n'.join(res)


def _get_slrum_params(partition='learnfair',
                      comment='flashlight',
                      cpus_per_task=10,
                      gpus=16,
                      gpu32=True,
                      mem_per_gpu=55,
                      hours=72):
    assert gpus > 0
    assert gpus <= 8 or gpus % 8 == 0, gpus

    slrum_params = {}
    ntasks = min(8, gpus)
    slrum_params['cpus-per-task'] = cpus_per_task
    slrum_params['partition'] = partition
    slrum_params['gres'] = f'gpu:volta:{ntasks}'
    if gpu32:
        slrum_params['constraint'] = 'volta32gb,bldg2'
    slrum_params['ntasks-per-node'] = ntasks
    slrum_params['nodes'] = max(1, gpus // 8)
    slrum_params['time'] = f'{hours}:00:00'
    slrum_params['mem'] = f'{mem_per_gpu * max(1, ntasks)}GB'
    slrum_params['signal'] = 'B:USR1@200'
    slrum_params['comment'] = 'flashlight_h2_release'
    slrum_params['open-mode'] = 'append'

    return slrum_params


def main(binary, mode, config, model_path, extra, partition, comment, ngpu, gpu16, cpus_per_task, mem_per_gpu, hours, local):
    assert binary.exists(), binary
    assert config.exists(), config

    if local:
        ngpu = 1

    #####################
    ### 1. Parse Mode ###
    #####################

    flags = ""
    with open(config) as f:
        for line in f:
            flags += line.strip() + " "
    
    for flag in extra.strip().split():
        flags += flag.strip() + " "
    
    flags += f"--logtostderr "
    if ngpu == 1:
        flags += "--distributed_enable=false "

    # Train mode
    if mode == 'train':
        exp_name = config.parent.name
        exp_id, _ = config.name.rsplit('.', 1)
        exp_id += _format_extra_flags(extra)
        log_dir = os.path.join(LOG_ROOT, exp_name)

        if local:
            log_dir = '/tmp'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        sub_log_dir = os.path.join(log_dir, exp_id)
        if not os.path.exists(sub_log_dir):
            oldmask = os.umask(000)
            os.makedirs(sub_log_dir, mode=0o777, exist_ok=True)
            os.umask(oldmask)

        checkpoint_path = os.path.join(sub_log_dir, 'model_0')
        if os.path.exists(checkpoint_path):
            print('Exists:', checkpoint_path)
            print('Skipping')
            return

        model_path = os.path.join(log_dir, exp_id)
        model_path = os.path.join(model_path, "model_")
        flags += f" --exp_checkpoint_path={model_path}"

    # Continue mode
    elif mode == 'continue':
        print("Continue mode not supported yet")
        return

    else:
        print("Not supported mode: ", mode)
        return

    ############################
    ### 2. Create Job Script ###
    ############################

    job_script = JOB_TEMPLATE.format(
        prerequesits='',
        comment=comment,
        binary=binary,
        mode=mode,
        model_path=os.path.join(log_dir, exp_id),
        flags=flags,
        postprocess=''
    )
    job_script_path = os.path.join(log_dir, f"{exp_id}/{mode}_{USERNAME}.sh")
    with open(job_script_path, 'w') as f:
        f.write(job_script)
    st = os.stat(job_script_path)
    os.chmod(job_script_path, st.st_mode | stat.S_IEXEC)

    ###############################
    ### 3. Create Sbatch Script ###
    ###############################

    gpu32 = not gpu16
    slrum_params = _get_slrum_params(
        partition=partition, gpus=ngpu, gpu32=gpu32, hours=hours)
    slrum_params.update({
        'job-name': f'{exp_name}:{exp_id}',
        'output': f'{log_dir}/{exp_id}/{USERNAME}.out',
        'error': f'{log_dir}/{exp_id}/{USERNAME}.err',
    })

    sbatch_cmd = SBATCH_TEMPLATE.format(
        slrum_params=_format_slrum_params(slrum_params),
        job_script=job_script_path)
    sbatch_path = os.path.join(log_dir, f"{exp_id}/run_{USERNAME}.sh")
    with open(sbatch_path, 'w') as f:
        f.write(sbatch_cmd)
    st = os.stat(sbatch_path)
    os.chmod(sbatch_path, st.st_mode | stat.S_IEXEC)

    ##############
    ### 4. Run ###
    ##############

    if local:
        os.system(job_script)
    else:
        os.system("sbatch " + sbatch_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    ## Training
    parser.add_argument('--binary', type=pathlib.Path)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--config', type=pathlib.Path, default='')
    parser.add_argument('--model_path', type=pathlib.Path, default='')
    parser.add_argument('--extra', type=str, default='')
    ## sbatch
    parser.add_argument('--partition', type=str, default='learnfair')
    parser.add_argument('--comment', type=str, default='wav2letter')
    parser.add_argument('--ngpu', type=int, default=8)
    parser.add_argument('--gpu16', action='store_true')
    parser.add_argument('--cpus_per_task', type=int, default=10)
    parser.add_argument('--mem_per_gpu', type=int, default=55)
    parser.add_argument('--hours', type=int, default=72)
    ## local
    parser.add_argument('--local', action='store_true')

    main(**vars(parser.parse_args()))
