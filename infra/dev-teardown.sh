#!/usr/bin/env bash
set -euo pipefail

# Destroy EC2 instance and supporting resources, but PRESERVE the EBS data volume.
#
# Strategy:
#   1. Load AWS credentials (same pattern as dev-setup.sh)
#   2. Initialize Terraform if needed
#   3. Import any resources that exist in AWS but are missing from Terraform state
#      (self-healing for manual terminations or partial previous teardowns)
#   4. Remove the EBS volume from Terraform state so destroy won't delete it
#   5. Run terraform destroy to clean up everything else
#   6. Re-import the EBS volume so state is ready for the next dev-setup
#
# Run from repo root: infra/dev-teardown.sh (or from infra: ./dev-teardown.sh)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── AWS credentials ──────────────────────────────────────────────────────────

CONFIG_FILE="$REPO_ROOT/.secrets/aws-sso-config.sh"
if [ -f "$CONFIG_FILE" ]; then
  # shellcheck source=.secrets/aws-sso-config.sh
  . "$CONFIG_FILE"
fi

export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-west-2}"

export_aws_creds() {
  eval "$(aws configure export-credentials --format env 2>/dev/null)"
}
if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
  if ! export_aws_creds || [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo "⚠️  Credentials not exported (SSO may be expired). Running 'aws sso login'..."
    aws sso login || true
    if ! export_aws_creds || [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
      echo "❌ Could not export AWS credentials for Terraform. Run 'just aws' to log in, then run this script again." >&2
      exit 1
    fi
  fi
fi

PROJECT_NAME="${PROJECT_NAME:-smart-meter-analysis}"

echo "🗑️  Destroying EC2 instance (preserving data volume)..."
echo

cd "$SCRIPT_DIR"

# ── Terraform init ───────────────────────────────────────────────────────────

if [ ! -d ".terraform" ]; then
  echo "📦 Initializing Terraform..."
  terraform init
  echo
fi

# ── State drift recovery ────────────────────────────────────────────────────
#
# If someone terminated the instance via the AWS console, Terraform state may
# be out of sync. Import any resources that exist in AWS but not in state so
# that terraform destroy can clean them up properly.

import_if_missing() {
  local addr="$1"
  local import_id="$2"

  if terraform state show "$addr" >/dev/null 2>&1; then
    echo "   ℹ️  $addr — already in state, skipping"
    return 0
  fi

  echo "   🔍 $addr — not in state, attempting import..."
  if terraform import "$addr" "$import_id" >/dev/null 2>&1; then
    echo "   ✅ $addr — imported successfully"
  else
    echo "   ⚠️  $addr — import failed (resource may not exist in AWS)"
  fi
}

echo "🔍 Checking for state drift..."

# IAM resources (static names)
import_if_missing "aws_iam_role.ec2_role" \
  "${PROJECT_NAME}-ec2-role"

import_if_missing "aws_iam_instance_profile.ec2_profile" \
  "${PROJECT_NAME}-ec2-profile"

import_if_missing "aws_iam_role_policy.ssm_managed_instance" \
  "${PROJECT_NAME}-ec2-role:${PROJECT_NAME}-ssm-managed-instance"

import_if_missing 'aws_iam_role_policy.s3_access[0]' \
  "${PROJECT_NAME}-ec2-role:${PROJECT_NAME}-s3-access"

import_if_missing "aws_iam_role_policy_attachment.ssm_managed_instance_core" \
  "${PROJECT_NAME}-ec2-role/arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"

# Security group (need to look up sg-xxxx ID)
SG_ID=$(aws ec2 describe-security-groups \
  --filters "Name=group-name,Values=${PROJECT_NAME}-sg" \
  --query 'SecurityGroups[0].GroupId' \
  --output text 2>/dev/null || echo "None")

if [ -n "$SG_ID" ] && [ "$SG_ID" != "None" ]; then
  import_if_missing "aws_security_group.ec2_sg" "$SG_ID"
fi

# EBS volume (need to look up vol-xxxx ID)
VOLUME_ID=$(aws ec2 describe-volumes \
  --filters "Name=tag:Name,Values=${PROJECT_NAME}-data" \
  --query 'Volumes[0].VolumeId' \
  --output text 2>/dev/null || echo "None")

if [ -n "$VOLUME_ID" ] && [ "$VOLUME_ID" != "None" ]; then
  import_if_missing "aws_ebs_volume.data" "$VOLUME_ID"
fi

# EC2 instance (may already be terminated — that's fine)
INSTANCE_ID=$(aws ec2 describe-instances \
  --filters "Name=tag:Project,Values=${PROJECT_NAME}" "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text 2>/dev/null || echo "None")

if [ -n "$INSTANCE_ID" ] && [ "$INSTANCE_ID" != "None" ]; then
  import_if_missing "aws_instance.main" "$INSTANCE_ID"

  # Volume attachment only exists if both instance and volume are present
  if [ -n "$VOLUME_ID" ] && [ "$VOLUME_ID" != "None" ]; then
    import_if_missing "aws_volume_attachment.data" \
      "/dev/sdf:${VOLUME_ID}:${INSTANCE_ID}"
  fi
fi

echo

# ── Save EBS volume ID ──────────────────────────────────────────────────────
# Try Terraform state first, fall back to AWS CLI lookup above.

EBS_VOL_ID=""
EBS_VOL_ID=$(terraform state show aws_ebs_volume.data 2>/dev/null \
  | grep '^\s*id\s*=' | head -1 | sed 's/.*= *"//;s/".*//' || true)

if [ -z "$EBS_VOL_ID" ] && [ -n "$VOLUME_ID" ] && [ "$VOLUME_ID" != "None" ]; then
  EBS_VOL_ID="$VOLUME_ID"
fi

if [ -z "$EBS_VOL_ID" ]; then
  echo "⚠️  Could not find EBS data volume — nothing to preserve."
  echo "   Proceeding with destroy anyway."
  echo
fi

# ── Remove EBS volume from state so destroy won't delete it ─────────────────

if [ -n "$EBS_VOL_ID" ]; then
  echo "📦 Removing EBS volume from Terraform state (to preserve it)..."
  terraform state rm aws_volume_attachment.data 2>/dev/null || true
  terraform state rm aws_ebs_volume.data 2>/dev/null || true
  echo
fi

# ── Destroy everything else ─────────────────────────────────────────────────

echo "🏗️  Running terraform destroy..."
terraform destroy -auto-approve
echo

# ── Re-import EBS volume into state ─────────────────────────────────────────

if [ -n "$EBS_VOL_ID" ]; then
  echo "📦 Re-importing EBS volume into Terraform state..."
  if terraform import aws_ebs_volume.data "$EBS_VOL_ID" >/dev/null 2>&1; then
    echo "   ✅ Volume re-imported: $EBS_VOL_ID"
  else
    echo "   ⚠️  Could not re-import volume. Run manually before next dev-setup:"
    echo "      cd infra && terraform import aws_ebs_volume.data $EBS_VOL_ID"
  fi
  echo
fi

# ── Done ─────────────────────────────────────────────────────────────────────

echo "✅ Teardown complete"
if [ -n "$EBS_VOL_ID" ]; then
  echo "   📦 Data volume preserved: $EBS_VOL_ID"
fi
echo "   To recreate the instance, run: just dev-setup"
