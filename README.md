# smart-meter-analysis

[![Release](https://img.shields.io/github/v/release/switchbox-data/smart-meter-analysis)](https://img.shields.io/github/v/release/switchbox-data/smart-meter-analysis)
[![Build status](https://img.shields.io/github/actions/workflow/status/switchbox-data/smart-meter-analysis/main.yml?branch=main)](https://github.com/switchbox-data/smart-meter-analysis/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/switchbox-data/smart-meter-analysis/branch/main/graph/badge.svg)](https://codecov.io/gh/switchbox-data/smart-meter-analysis)
[![Commit activity](https://img.shields.io/github/commit-activity/m/switchbox-data/smart-meter-analysis)](https://img.shields.io/github/commit-activity/m/switchbox-data/smart-meter-analysis)
[![License](https://img.shields.io/github/license/switchbox-data/smart-meter-analysis)](https://img.shields.io/github/license/switchbox-data/smart-meter-analysis)

This is the repo for the smart-meter-analysis project

- **Github repository**: <https://github.com/switchbox-data/smart-meter-analysis/>
- **Documentation** <https://switchbox-data.github.io/smart-meter-analysis/>

## 🚀 Quick Start: Using the Development VM

For data scientists working on this project, we provide a shared EC2 VM with all dependencies pre-configured. This is the easiest way to get started.

### Prerequisites

Before you begin, ensure you have the following tools installed on your local machine:

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

4. **Required File**: `.secrets/aws-sso-config.sh` - AWS SSO configuration file (ask a team member for this file)

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
   - If Cursor doesn't open automatically, manually connect in Cursor to: `ssh-remote+smart-meter-analysis`

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

### Understanding Storage on the VM

The VM has three types of storage:

1. **Root Volume (`/`)** - 30 GB system disk (not persistent, destroyed when VM terminates)
   - Use for: Operating system and system files only
   - **Don't store your work here** - it will be lost if the VM is recreated

2. **EBS Volume (`/ebs`)** - 500 GB persistent data disk
   - Use for: Your home directory, project files, installed packages
   - Your home directory (`~`) is actually `/ebs/home/your_username/`
   - Structure:
     - `/ebs/home/your_username/` - Your home directory (where `~/smart-meter-analysis/` lives)
     - `/ebs/shared/` - Shared directory accessible to all users

3. **S3 Mount (`/data.sb`)** - Unlimited S3 storage
   - Use for: Large datasets, shared data files
   - Files written here are automatically synced to S3
   - Great for data that needs to persist across VM recreations

**Quick Reference:**

| Storage Type | Path | Size | Persistent? | Best For |
|-------------|------|------|-------------|----------|
| Root Volume | `/` | 30 GB | ❌ No | System files only |
| EBS Volume | `/ebs/` | 500 GB | ✅ Yes | Your work, home directory |
| S3 Mount | `/data.sb/` | Unlimited | ✅ Yes | Large datasets, shared data |

### Cost Management

**⚠️ Important**: The VM does **not** automatically shut down. It will continue running and incur AWS costs until you explicitly tear it down.

- **Instance Type**: `m7i.2xlarge` (8 vCPUs, 32 GB RAM)
- **Estimated Cost**: ~$0.50-1.00/hour (varies by region and pricing)
- **Recommendation**: Always run `just dev-teardown` when you're done working

For more details, see [`infra/README.md`](infra/README.md).

## Getting started with your project

### 1. Create a New Repository

First, create a repository on GitHub with the same name as this project, and then run the following commands:

```bash
git init -b main
git add .
git commit -m "init commit"
git remote add origin git@github.com:switchbox-data/smart-meter-analysis.git
git push -u origin main
```

### 2. Set Up Your Development Environment

Then, install [just](https://github.com/casey/just) and use it to install our python packages and the pre-commit hooks with

```bash
just install
```

This will also generate your `uv.lock` file

### 3. Run the pre-commit hooks

Initially, the CI/CD pipeline might be failing due to formatting issues. To resolve those run:

```bash
uv run pre-commit run -a
```

### 4. Commit the changes

Lastly, commit the changes made by the two steps above to your repository.

```bash
git add .
git commit -m 'Fix formatting issues'
git push origin main
```

You are now ready to start development on your project!
The CI/CD pipeline will be triggered when you open a pull request, merge to main, or when you create a new release.

To finalize the set-up for publishing to PyPI, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/publishing/#set-up-for-pypi).
For activating the automatic documentation with MkDocs, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/mkdocs/#enabling-the-documentation-on-github).
To enable the code coverage reports, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/codecov/).

## Releasing a new version

- Create an API Token on [PyPI](https://pypi.org/).
- Add the API Token to your projects secrets with the name `PYPI_TOKEN` by visiting [this page](https://github.com/switchbox-data/smart-meter-analysis/settings/secrets/actions/new).
- Create a [new release](https://github.com/switchbox-data/smart-meter-analysis/releases/new) on Github.
- Create a new tag in the form `*.*.*`.

For more details, see [here](https://fpgmaas.github.io/cookiecutter-uv/features/cicd/#how-to-trigger-a-release).

---

Repository initiated with [fpgmaas/cookiecutter-uv](https://github.com/fpgmaas/cookiecutter-uv).
