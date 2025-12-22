# Prerequisites and Setup

## Required Configuration

### Environment Variables

Create `.envrc` in the `doe-suite/` directory:

```bash
# Project configuration
export DOES_PROJECT_DIR="/absolute/path/to/artemis"
export DOES_PROJECT_ID_SUFFIX="artemis"

# AWS configuration
export DOES_CLOUD="aws"
export AWS_PROFILE="your-aws-profile"
export AWS_REGION="eu-north-1"  # or your preferred region

# SSH configuration
export DOES_SSH_KEY_NAME="your-key-name"  # Must exist in target region
```

**Load environment:**
```bash
cd doe-suite
source .envrc
```

DoE-Suite validates configuration on launch.

## AWS Setup

### SSH Key

Your SSH key must exist in the target AWS region.

**Check existing keys:**
```bash
aws ec2 describe-key-pairs --region eu-north-1
```

**Create key if needed:**
```bash
aws ec2 create-key-pair --key-name my-key --region eu-north-1 \
  --query 'KeyMaterial' --output text > ~/.ssh/my-key.pem
chmod 400 ~/.ssh/my-key.pem
```

**Update `.envrc`:**
```bash
export DOES_SSH_KEY_NAME="my-key"
```

### AWS Credentials

**Configure AWS CLI:**
```bash
aws configure --profile your-profile
```

**Verify access:**
```bash
aws sts get-caller-identity --profile your-profile
```

### Billing Alerts

Set up billing alerts to monitor costs:

1. AWS Console → Billing → Billing Preferences
2. Enable "Receive Billing Alerts"
3. CloudWatch → Alarms → Create Alarm
4. Set threshold (e.g., $50/day)

## Machine Configurations

Instance types defined in `doe-suite-config/group_vars/<size>/main.yml`:

| Size   | Instance Type | vCPUs | RAM   | Volume | Hourly Cost* |
|--------|---------------|-------|-------|--------|--------------|
| Small  | r6i.8xlarge   | 32    | 256GB | 250GB  | ~$2          |
| Medium | r6i.16xlarge  | 64    | 512GB | 250GB  | ~$4          |
| Large  | r6i.32xlarge  | 128   | 1TB   | 250GB  | ~$8          |

*Costs vary by region and are approximate.

All use AMI: `ami-0714b2f0040bb3a42` (eu-north-1 Stockholm).

## Validation

Test configuration:

```bash
cd doe-suite
source .envrc

# Check available suites
make info

# Validate a suite design
make design-validate suite=poly-small
```

If configuration is incorrect, DoE-Suite will report specific errors.

[← Back to main guide](./SKILL.md)
