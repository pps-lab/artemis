# Prerequisites and Setup

Everything below is what actually worked for the Oct 2026 runs (eu-west-1, stock Ubuntu). Check it before
`make run`; each item cost a debugging round the first time.

## Required Configuration

### Environment Variables

`.envrc` lives in `doe-suite-config/` (not in `doe-suite/`) and is git-ignored. Load it from `doe-suite/`:

```bash
cd doe-suite
source ../doe-suite-config/.envrc
export PYTHONPATH=$DOES_PROJECT_DIR/doe-suite-config   # custom ETL steps (does_etl_custom) are imported from here
```

Current contents (Oct 2026):

```bash
export DOES_PROJECT_DIR="/Users/hidde/Postdoc-newpc/artemis"
export DOES_PROJECT_ID_SUFFIX="hly"        # project id artemis_hly -> VPC artemis_hly_vpc_base
export DOES_CLOUD="aws"
export DOES_SSH_KEY_NAME="hidde-mbp"       # EC2 key pair in eu-west-1 = ~/.ssh/id_ed25519
export DOES_AWS_USER="ubuntu"              # stock Ubuntu AMI
```

The region is **not** taken from `.envrc`: it is `aws_region: eu-west-1` in `doe-suite-config/group_vars/all/main.yml`
(together with the key pair name and the git branch the instances clone, `git_remote_repository.version`).
Push the branch before launching.

### SSH

- The EC2 key pair must exist in the launch region and match a local private key. `aws ec2 import-key-pair`
  needs permissions this account's CLI user does not have: import the public key in the EC2 console
  (Key pairs → Actions → Import) and set `DOES_SSH_KEY_NAME` to the name you gave it.
- Ansible talks to fresh instances non-interactively, and `ansible_ssh_common_args` in `doe-suite/ansible.cfg`
  sits under `[defaults]` where it has no effect, so the first contact fails with
  "Host key verification failed" unless `~/.ssh/config` handles it:

  ```
  Host *.compute.amazonaws.com *.compute-1.amazonaws.com
      User ubuntu
      IdentityFile ~/.ssh/id_ed25519
      IdentitiesOnly yes
      StrictHostKeyChecking accept-new
      ServerAliveInterval 60
      ServerAliveCountMax 10
  ```

- A stale `SSH_AUTH_SOCK` (agent socket from an old terminal) makes every ssh hang; `unset SSH_AUTH_SOCK` or
  kill the stale agent.

### AWS Credentials

Default profile, account 226594459110. Verify with `aws sts get-caller-identity`. The vCPU quota is 896; the
eu-west-1 **VPC limit is 5** and each `DOES_PROJECT_ID_SUFFIX` creates its own VPC, so `VpcLimitExceeded` means
delete an old `artemis_*_vpc_base` (VPC console; the suffix's own VPC is reused across suites).

## Machine Configurations

Instance types in `doe-suite-config/group_vars/<size>/main.yml`:

| Size   | Instance Type | vCPUs | RAM   | Volume | Hourly Cost* |
|--------|---------------|-------|-------|--------|--------------|
| Small  | r6i.8xlarge   | 32    | 256GB | 250GB  | ~$2          |
| Medium | r6i.16xlarge  | 64    | 512GB | 250GB  | ~$4          |
| Large  | r6i.32xlarge  | 128   | 1TB   | 250GB  | ~$8          |

*Approximate.

All tiers use the stock Canonical Ubuntu 24.04 AMI in eu-west-1 (`ami-0445442ed20a2c6f8`, snapshot
`snap-0798e81b434f07f27`, root volume resized to 250 GB). The `setup-base` role installs build deps and
`time`, rustup, awscli, syncs the models from S3 and builds with the toolchain pinned in `rust-toolchain.toml`.
The old custom AMIs are gone from Ireland (`ami-0714b2f0040bb3a42` only exists in eu-north-1 Stockholm).
To move region again: new AMI id + snapshot id in the three `group_vars/<size>/main.yml`, `aws_region` in
`group_vars/all/main.yml`, and a key pair in that region.

## Build Reproducibility on the Instances

- `Cargo.lock` is committed. A fresh resolution pulls crates needing `edition2024`, which the pinned
  `nightly-2024-05-09` does not have (`feature edition2024 is required`, e.g. rmp-serde 1.3.1).
- `src/bin/test.zsh` builds with `(cd "$dir" && cargo build --release --bin time_circuit)`: rustup resolves
  `rust-toolchain.toml` from the current directory, not from `--manifest-path`, and doe-suite runs the command
  from the results directory (stable toolchain → `E0554` on nightly features).
- Never `cargo +nightly`: it picks whatever `nightly` is on the machine.

## Validation

```bash
cd doe-suite
source ../doe-suite-config/.envrc
export PYTHONPATH=$DOES_PROJECT_DIR/doe-suite-config
make design-validate suite=poly-small
```

`make design-validate` imports the custom ETL modules, so it also catches ETL config errors before any
instance is started.

[← Back to main guide](./SKILL.md)
