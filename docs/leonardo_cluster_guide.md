# Leonardo Cluster Guide

Reference guide for running MLP4CS experiments on the EuroHPC Leonardo cluster.

---

## 1. Overview
- **Project account:** `EUHPC_D34_189`
- **Username:** `gxydias0`
- **Budget:** 144,000 CPU-hours, valid until March 2027
- **Login:** SSH with password + 2FA via Google Authenticator
- **Work modes:**
  - **Interactive session:** fast debugging, max 30 min, immediate output
  - **Batch job:** submit and forget, runs for hours/days on compute node
- **Hardware partitions:**
  - **Booster** (A100 GPUs): 3456 nodes, 32 cores, 4× A100 64GB, 512GB RAM, used for our experiments
  - **DCGP** (CPU only): 1536 nodes, 112 cores, 512GB RAM
- **Main file system paths:**
  - `$HOME` = `/leonardo/home/userexternal/gxydias0` (50GB, usually for code, conda)
  - `$WORK` = `/leonardo_work/EUHPC_D34_189` (1TB, usually for datasets, outputs)
  - `$FAST` = `/leonardo_scratch/fast/EUHPC_D34_189` (1TB for fast I/O, auto-cleaned after 40 days)

---

## 2. Daily Connection
Run from local PowerShell (as Administrator):

**Step 1: Authenticate with step:**
```bash
step ssh login gxydias.ece@gmail.com --provisioner cineca-hpc
```
- Enter password when asked
- Approve the 2FA prompt in Google Authenticator

**Step 2: Clear old host key** (only if "connection key error"):
```bash
ssh-keygen -R login.leonardo.cineca.it
```

**Step 3: SSH into Leonardo:**
```bash
ssh gxydias0@login.leonardo.cineca.it
```

Once connected, activate conda and move to the project:
```bash
conda activate mlp4cs
cd $WORK/mlp4cs
```

---

## 3. Budget Monitoring
On the cluster:
```bash
saldo -b gxydias0 # Personal budget
saldo -b EUHPC_D34_189 # Project budget
cindata # Storage usage
cinQuota # Quota limits
```

**Budget formula:** `Cost = Hours × Nodes × GPU_fraction × CPU_cores`
- 1 GPU job for 10h = `10 × 1 × 0.25 × 32` = 80 CPU-hours
- 4 GPU job for 10h = `10 × 1 × 1.0 × 32` = 320 CPU-hours
- Available single-GPU jobs: 144,000 ÷ 80 = ~1,800 (the maximum of the resources requested per node is determined by the GPUs)

---

## 4. Data Transfer
Run from **local PowerShell** (re-run `step ssh login gxydias.ece@gmail.com --provisioner cineca-hpc` first if the session has expired).

**Upload file:**
```bash
scp <local_path> gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/<remote_path>/
```

**Upload folder:**
```bash
scp -r <local_folder> gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/<remote_path>/
```

**Download file:**
```bash
scp gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/<remote_path> <local_path>
```

**Download folder (results/logs):**
```bash
scp -r gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/results C:\Users\giorg\Downloads\
scp -r gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/logs C:\Users\giorg\Downloads\
```

---

## 5. First-Time Setup 
### Clean + install conda
```bash
# Clean existing (if needed)
ls -a
rm -f $HOME/.condarc
rm -rf $HOME/.conda

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash $HOME/Miniconda3-latest-Linux-x86_64.sh
exec bash
```

### Configure conda-forge (avoids CINECA network issues)
```bash
conda config --set auto_activate_base false
conda config --set channel_priority strict
conda config --add channels conda-forge
```

Then comment out default channels in both `.condarc` files:
```bash
nano $HOME/.condarc
nano $HOME/miniconda3/.condarc
# Add # before any line with "defaults" or "repo.anaconda.com"
```

### Create directory structure
```bash
cd $WORK
mkdir mlp4cs
```

### Create environment
```bash
conda create -n mlp4cs python=3.10 -y
conda activate mlp4cs
cd $WORK/mlp4cs
pip install -r requirements.txt
```

**Working environment:**
 - Python 3.10, Torch 2.7.0+cu126, Transformers 4.52.4, Unsloth 2025.11.1
 - A100-SXM-64GB, CUDA Toolkit 12.6, Bfloat16 TRUE

### Upload project (from local PowerShell)
```bash
cd C:\Users\giorg\Projects\PycharmProjects\mlp4cs
scp -r data gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/
scp -r src gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/
scp -r scripts gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/
scp requirements.txt gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/
```

---

## 6. Interactive GPU Session (for testing)
Use for quick validation before submitting batch jobs. Max 30 min.

```bash
srun -N 1 --ntasks-per-node=1 --gres=gpu:1 --partition=boost_usr_prod --qos=boost_qos_dbg --account=EUHPC_D34_189 --time=00:30:00 --pty /bin/bash
```

**Flag breakdown:**
- `-N 1`: 1 node
- `--ntasks-per-node=1`: 1 task
- `--gres=gpu:1`: 1 GPU
- `--partition=boost_usr_prod`: A100 partition
- `--qos=boost_qos_dbg`: debug queue (max 30 min)
- `--time=00:30:00`: 30 min timeout

**Notes on partition and QoS:**
- **Partition** = a pool of nodes with similar hardware. List all: `sinfo`
  - `boost_usr_prod`: A100 GPU nodes (use for our experiments)
  - `dcgp_usr_prod`: CPU-only nodes, 112 cores each (for heavy CPU workloads, not used here)
- **QoS** (Quality of Service) = a "ticket type" that sets job limits (max time, max GPUs, priority). List all: `sacctmgr show qos format=Name%30,MaxWall%15,MaxTRES%30`
  - `boost_qos_dbg`: for debug, max 30 min (use for interactive testing)
  - `boost_qos_lprod`: for long production, up to 4 days (use for batch jobs)

Once on the compute node:
```bash
conda activate mlp4cs
cd $WORK/mlp4cs
nvidia-smi # verify GPU
python -c "import torch; print(torch.cuda.is_available())" # confirm PyTorch detects the GPU
python -m src.main # run experiment (with MAX_DIALOGUES=1 or 5 in config.py)
```

**Test scripts:**
```bash
python scripts/test_gpu.py
python scripts/test_all_models.py
```

---

## 7. Batch Jobs (full runs)

### Submit and monitor
```bash
sbatch scripts/slurm/<job_name>.sh # Submit (returns job ID)
squeue -u gxydias0 # Check status: PD=pending, R=running, CF=configuring
tail -f logs/<job_name>_<JOB_ID>.out # Watch live stdout (print statements)
tail -f logs/<job_name>_<JOB_ID>.err # Watch live progress (tqdm writes here)
cat logs/<job_name>_<JOB_ID>.out # Check stdout
cat logs/<job_name>_<JOB_ID>.err # Check errors
scancel <JOB_ID> # Cancel job
sacct -j <JOB_ID> # Job details after completion
```

### Template of a normal experiment workflow
**Step 1: Prepare SLURM script** (locally):
- Create `scripts/slurm/<exp_name>.sh` following the template in existing scripts
- Upload:
  ```bash
  scp scripts/slurm/<exp_name>.sh gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/scripts/slurm/
  ```

**Step 2: Update config on cluster:**
```bash
nano src/config.py
# Edit EXPX_CONFIGS to only include the active config
# Ensure MAX_DIALOGUES = None for full run
```

**Step 3: Quick test first** (interactive session, 1-5 dialogues):
```bash
# Start interactive session
srun -N 1 --ntasks-per-node=1 --gres=gpu:1 partition=boost_usr_prod --qos=boost_qos_dbg --account=EUHPC_D34_189 --time=00:30:00 --pty /bin/bash

# On compute node
conda activate mlp4cs
cd $WORK/mlp4cs
python -m src.main

# Exit session when done
exit
```
Verify `results/<exp>/<exp>_tomiinek_input.json` exists and remove `results/` for the next full run.

**Step 4: Submit full batch job:**
```bash
sbatch scripts/slurm/<exp_name>.sh
```

**Step 5: Monitor and collect:**
```bash
tail -f logs/<exp_name>_<JOB_ID>.err
# When done:
scp -r gxydias0@login.leonardo.cineca.it:/leonardo_work/EUHPC_D34_189/mlp4cs/results C:\Users\giorg\Downloads\
```

**Step 6: Local Tomiinek evaluation:**
```bash
# From local project root (run locally)
python src/evaluation/tomiinek_local.py
```

---

## 8. SLURM Script Template

```bash
#!/bin/bash
#SBATCH --job-name=<exp_name>
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --output=logs/<exp_name>_%j.out
#SBATCH --error=logs/<exp_name>_%j.err
#SBATCH --account=EUHPC_D34_189

cd $WORK/mlp4cs
source ~/.bashrc
conda activate mlp4cs

# Offline mode (no internet on compute nodes)
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m src.main
```

---

## 9. References

**Official documentation:**
- [CINECA HPC Documentation](https://docs.hpc.cineca.it/index.html) 

**Account & budget management:**
- [UserDB](https://userdb.hpc.cineca.it/index?destination=node/20742) 
- [CINECA Keycloak](https://sso.hpc.cineca.it/realms/CINECA-HPC/account/#/) 

**Project applications:**
- [EuroHPC JU Application Portal](https://access.eurohpc-ju.europa.eu/applications) 

---
