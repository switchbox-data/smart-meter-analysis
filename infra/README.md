# EC2 Infrastructure for Smart Meter Analysis

This directory contains Terraform configuration for provisioning a shared EC2 instance for the smart meter analysis project.

## Overview

The infrastructure includes:

- EC2 instance (Ubuntu 22.04) with configurable instance type
- Persistent EBS volume (mounted at `/ebs`) for user home directories and shared data
- S3 bucket mount (`s3://data.sb/` mounted at `/data.sb/`) for large data files
- IAM roles and security groups for secure access
- Automatic user account creation based on AWS IAM users

## Prerequisites

Before you begin, ensure you have the following tools installed on your local machine:

### Required Tools

1. **AWS CLI** - For authenticating with AWS
   - Installation: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html
   - Verify: `aws --version`

2. **just** - Command runner (like `make` but simpler)
   - Installation: https://just.systems/
   - macOS: `brew install just`
   - Verify: `just --version`

3. **Cursor CLI** (optional but recommended) - For automatic remote connection
   - Installation: https://cursor.com/install
   - Or run: `curl https://cursor.com/install -fsS | bash`
   - Verify: `cursor --version`
   - **Note**: If you don't have this installed, you'll need to manually connect to the remote server in Cursor after running `just dev-login`

4. **Terraform** (optional) - Automatically installed by `just dev-setup` if not present
   - Manual installation: https://developer.hashicorp.com/terraform/downloads

### Required Files

- `.secrets/aws-sso-config.sh` - AWS SSO configuration file (ask a team member for this file)

## Quick Start Guide

This guide walks you through spinning up the VM, logging in, and tearing it down.

### Step 1: Spin Up the VM (One-Time Setup)

**Note**: This step only needs to be done once by an admin. If the VM already exists, skip to Step 2.

Run this command from the repository root:

```bash
just dev-setup
```

**What to expect:**

1. **AWS Authentication**: You'll see a browser window open (or a URL to visit) for AWS SSO login:
   ```
   🔓 Starting AWS SSO login...
   Attempting to automatically open the SSO authorization page...
   ```
   - If your browser opens automatically, complete the login there
   - If not, copy the URL shown and paste it into your browser
   - Enter the verification code shown in your terminal

2. **Terraform Installation** (if needed): If Terraform isn't installed, you'll see:
   ```
   📦 Terraform not found. Installing...
   ```
   This may take a minute.

3. **Infrastructure Creation**: Terraform will create AWS resources. You'll see output like:
   ```
   🏗️  Applying Terraform configuration...
   Plan: 9 to add, 0 to change, 0 to destroy.
   ```
   This process takes about 2-3 minutes.

4. **Instance Setup**: The script waits for the instance to be ready and installs tools:
   ```
   ⏳ Waiting for instance to be ready...
   ✅ Instance is ready
   🔧 Installing just, uv, gh, AWS CLI...
   ✅ Tools installed successfully
   ```

5. **Success**: You'll see the instance details:
   ```
   ✅ Setup complete!
   Instance ID: i-0afd6eb198e90d022
   Public IP: 35.94.213.195
   ```

**Important**: The VM will continue running and incur AWS costs until you tear it down (see Step 3). There is **no automatic shutdown** configured.

### Step 2: Log In to the VM

Once the VM is set up (or if it already exists), any authorized user can log in:

```bash
just dev-login
```

**What to expect:**

1. **AWS Authentication**: Similar to Step 1, you may need to authenticate with AWS SSO if your session expired.

2. **Session Manager Plugin Installation** (first time only): If you don't have the AWS Session Manager plugin:
   ```
   📦 Session Manager plugin not found. Installing from AWS...
   ```
   This requires `sudo` access - you'll be prompted for your password.

3. **User Account Creation** (first time only): If this is your first login:
   ```
   👤 Setting up user account...
   Creating user account: your_username
   ```
   Your Linux username is derived from your AWS IAM username.

4. **Repository Setup**: The script clones the repository:
   ```
   📦 Setting up development environment...
   Cloning repository...
   ✅ Repository cloned
   ```

5. **GitHub Authentication**: On first login, you'll be prompted to authenticate with GitHub:
   ```
   🚀 Running first-login setup (gh auth + uv sync)...
   📦 GitHub authentication required for private dependencies...
   Opening browser for GitHub authentication...
   ```
   - A browser window will open (or you'll get a URL to visit)
   - Complete the GitHub authentication
   - **Note**: You may need to manually open the URL if the browser doesn't open automatically

6. **Python Environment Setup**: After GitHub auth, dependencies are installed:
   ```
   📦 Installing dependencies...
   uv sync --python 3.13
   ```
   This may take a few minutes the first time.

7. **Cursor Connection** (if Cursor CLI is installed):
   ```
   Opening Cursor with remote workspace...
   ✅ Cursor opened successfully
   ```
   - If Cursor opens automatically, you're all set!
   - If you see "Could not open Cursor remotely", you need to install the Cursor CLI (see Prerequisites)

8. **Interactive Shell**: At the end, you'll see:
   ```
   Opening interactive session...
   (Press Ctrl+D to exit)
   ```
   You'll be dropped into a `zsh` shell on the remote server. This is your development environment.

**Troubleshooting**:
- If Cursor doesn't open automatically, manually connect in Cursor to: `ssh-remote+smart-meter-analysis`
- The SSH port forwarding runs in the background - keep the terminal open while using Cursor
- To exit the interactive shell, press `Ctrl+D`

### Step 3: Tear Down the VM

**Important**: Always tear down the VM when you're done to avoid unnecessary AWS costs.

#### Option A: Preserve Data (Recommended)

This stops the VM but keeps your data on the EBS volume. You can recreate the VM later and your data will still be there:

```bash
just dev-teardown
```

**What to expect:**
```
🗑️  Destroying EC2 instance (preserving data volume)...
🏗️  Destroying instance resources (keeping data volume)...
✅ Teardown complete
   📦 Data volume preserved: vol-0e0f7ea32905a7dce
To recreate the instance, run: just dev-setup
```

#### Option B: Complete Teardown (Destroys All Data)

**Warning**: This permanently deletes everything including the data volume. Only use this if you're sure you don't need any data:

```bash
just dev-teardown-all
```

**What to expect:**
```
⚠️  WARNING: This will destroy EVERYTHING including the data volume!
   All data on the EBS volume will be permanently deleted.
Are you sure? Type 'yes' to confirm:
```
Type `yes` to confirm, then the cleanup proceeds.

## Understanding Storage: Root Volume, EBS Volume, and S3 Mount

The VM has three types of storage, each serving different purposes:

### 1. Root Volume (`/`)

- **What it is**: The main system disk where Ubuntu and system files live
- **Size**: 30 GB
- **Location**: `/` (root of the filesystem)
- **Persistence**: **Not persistent** - destroyed when the VM is terminated
- **Use case**: Operating system, system packages, temporary files
- **Don't store your work here** - it will be lost if the VM is recreated

### 2. EBS Volume (`/ebs`)

- **What it is**: A persistent data disk that survives VM recreation
- **Size**: 500 GB (configurable)
- **Location**: `/ebs/`
- **Persistence**: **Persistent** - survives VM termination (unless you run `dev-teardown-all`)
- **Use case**: Your home directory, project files, installed packages, local data
- **Structure**:
  - `/ebs/home/your_username/` - Your home directory (where `~/smart-meter-analysis/` lives)
  - `/ebs/shared/` - Shared directory accessible to all users
  - `/ebs/tmp/` - Temporary files directory

**Important**: Your home directory (`~`) is actually `/ebs/home/your_username/`, so files you save in your home directory are automatically on the persistent EBS volume.

### 3. S3 Mount (`/data.sb`)

- **What it is**: A direct mount of the S3 bucket `s3://data.sb/`
- **Size**: Unlimited (S3 storage)
- **Location**: `/data.sb/`
- **Persistence**: **Persistent** - data lives in S3, independent of the VM
- **Use case**: Large datasets, shared data files, data that needs to persist across VM recreations
- **Performance**: Slower than local disk (network access), but great for large files you don't need frequently

**Note**: Files written to `/data.sb/` are automatically synced to S3. Changes may take a moment to appear.

### Quick Reference

| Storage Type | Path | Size | Persistent? | Best For |
|-------------|------|------|-------------|----------|
| Root Volume | `/` | 30 GB | ❌ No | System files only |
| EBS Volume | `/ebs/` | 500 GB | ✅ Yes | Your work, home directory |
| S3 Mount | `/data.sb/` | Unlimited | ✅ Yes | Large datasets, shared data |

## Cost Management

**⚠️ Important**: The VM does **not** automatically shut down. It will continue running and incur AWS costs until you explicitly tear it down.

- **Instance Type**: `m7i.2xlarge` (8 vCPUs, 32 GB RAM)
- **Estimated Cost**: ~$0.50-1.00/hour (varies by region and pricing)
- **Recommendation**: Always run `just dev-teardown` when you're done working

The EBS volume incurs minimal storage costs (~$0.10/GB/month) even when the VM is stopped, but this is much cheaper than leaving the VM running.

## Configuration

Edit `variables.tf` to customize:

- `instance_type` - EC2 instance type (default: `m7i.2xlarge`)
- `ebs_volume_size` - EBS volume size in GB (default: 500)
- `aws_region` - AWS region (default: `us-west-2`)
- `s3_bucket_name` - S3 bucket to mount (default: `data.sb`)

## Changing Instance Type

To change the instance type:

1. Update `instance_type` in `variables.tf` or pass via command line:
   ```bash
   terraform apply -var="instance_type=m7i.4xlarge"
   ```

2. Terraform will automatically:
   - Stop the instance
   - Change the instance type
   - Start the new instance
   - Reattach the EBS volume
   - All user data in `/ebs/home/` persists automatically

## EBS Volume Resizing

To increase the EBS volume size:

1. Update `ebs_volume_size` in `variables.tf`
2. Run `terraform apply`
3. The user-data script automatically detects the larger volume and runs `resize2fs`

**Note:** EBS volumes cannot be decreased in size.

## Authorization

Users need the following AWS IAM permissions:

- `ec2-instance-connect:SendSSHPublicKey`
- `ec2:DescribeInstances`

To grant access, add these permissions to the user's IAM role/user. Their Linux account will be created automatically on first login.

## Files

- `main.tf` - Main infrastructure (EC2, EBS, IAM, security groups)
- `variables.tf` - Configuration variables
- `outputs.tf` - Terraform outputs (instance ID, IPs, etc.)
- `user-data.sh` - Bootstrap script that runs on instance startup
- `.gitignore` - Ignores Terraform state files

## Troubleshooting

### Instance not accessible

- Check security groups allow SSH from your IP/VPC
- Verify AWS SSO login: `just aws`
- Check instance status: `aws ec2 describe-instances --instance-ids <instance-id>`

### S3 mount not working

- Verify IAM instance profile has S3 permissions
- Check S3 bucket name is correct in `variables.tf`
- Check logs: `sudo journalctl -u user-data` or `/var/log/user-data.log`

### User account issues

- Verify AWS IAM username is valid
- Check `/ebs/home/` directory permissions
- User accounts are created automatically on first login

### Cursor doesn't open automatically

- Install the Cursor CLI: `curl https://cursor.com/install -fsS | bash`
- Or manually connect in Cursor to: `ssh-remote+smart-meter-analysis`
- Make sure the SSH port forwarding is still running (check the terminal where you ran `dev-login`)
