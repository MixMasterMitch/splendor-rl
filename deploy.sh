#!/usr/bin/env bash
#
# Deploy the Splendor webapp to AWS.
#
# This script:
#   1. Builds the frontend (if needed)
#   2. Runs CDK deploy (handles Docker group permissions)
#   3. Syncs all local game records to DynamoDB
#   4. Refits ratings for all users
#
# Usage:
#   ./deploy.sh [--skip-frontend] [--skip-sync] [--dry-run]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

REGION="us-west-2"
STACK_NAME="SplendorStack"
STORE_ROOT="play/play_data"

# --- Parse arguments ---
SKIP_FRONTEND=false
SKIP_SYNC=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --skip-frontend) SKIP_FRONTEND=true ;;
        --skip-sync)     SKIP_SYNC=true ;;
        --dry-run)       DRY_RUN=true ;;
        -h|--help)
            echo "Usage: ./deploy.sh [--skip-frontend] [--skip-sync] [--dry-run]"
            exit 0
            ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# --- Helpers ---
log() { echo -e "\033[1;34m==>\033[0m $*"; }
err() { echo -e "\033[1;31mERROR:\033[0m $*" >&2; }

# --- Check prerequisites ---
log "Checking prerequisites..."

if ! command -v cdk &>/dev/null; then
    err "CDK CLI not found. Install with: npm install -g aws-cdk"
    exit 1
fi

if ! command -v aws &>/dev/null; then
    err "AWS CLI not found."
    exit 1
fi

# Verify AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    err "AWS credentials not configured or expired."
    exit 1
fi

# --- Check Docker access ---
DOCKER_CMD="docker"
if ! docker info &>/dev/null 2>&1; then
    # Try with sg docker (user may have been added to group but not re-logged)
    if sg docker -c "docker info" &>/dev/null 2>&1; then
        DOCKER_CMD="sg docker -c"
        log "Using 'sg docker' for Docker access"
    else
        err "Docker is not accessible. Ensure Docker is running and your user is in the 'docker' group."
        err "Fix with: sudo usermod -aG docker \$USER && newgrp docker"
        exit 1
    fi
fi

# --- Step 1: Build frontend ---
if [ "$SKIP_FRONTEND" = false ]; then
    log "Building frontend..."
    if [ -d "replay_webapp/node_modules" ]; then
        (cd replay_webapp && npx tsc && npx vite build)
    else
        log "Installing frontend dependencies..."
        (cd replay_webapp && npm ci && npx tsc && npx vite build)
    fi
    log "Frontend built successfully."
else
    log "Skipping frontend build."
    if [ ! -d "replay_webapp/dist" ]; then
        err "replay_webapp/dist does not exist. Run without --skip-frontend."
        exit 1
    fi
fi

# --- Step 2: CDK Deploy ---
log "Deploying CDK stack..."

if [ "$DOCKER_CMD" = "docker" ]; then
    cdk deploy --require-approval never --force
else
    sg docker -c "cdk deploy --require-approval never --force"
fi

log "CDK deploy complete."

# --- Step 3: Sync local games to DynamoDB ---
if [ "$SKIP_SYNC" = false ]; then
    log "Syncing local game records to DynamoDB..."

    # Get table names from CloudFormation
    GAMES_TABLE=$(aws cloudformation describe-stack-resources \
        --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackResources[?starts_with(LogicalResourceId, 'GamesTable')].PhysicalResourceId" \
        --output text)

    USERS_TABLE=$(aws cloudformation describe-stack-resources \
        --stack-name "$STACK_NAME" --region "$REGION" \
        --query "StackResources[?starts_with(LogicalResourceId, 'UsersTable')].PhysicalResourceId" \
        --output text)

    if [ -z "$GAMES_TABLE" ] || [ -z "$USERS_TABLE" ]; then
        err "Could not find DynamoDB table names from stack $STACK_NAME"
        exit 1
    fi

    log "  Games table: $GAMES_TABLE"
    log "  Users table: $USERS_TABLE"

    # Run the sync-all script (uploads games + refits all user ratings)
    SYNC_ARGS="--store-root $STORE_ROOT --games-table $GAMES_TABLE --users-table $USERS_TABLE --region $REGION"
    if [ "$DRY_RUN" = true ]; then
        SYNC_ARGS="$SYNC_ARGS --dry-run"
    fi

    python3 -m play.sync_all $SYNC_ARGS

    log "Sync complete."
else
    log "Skipping game sync."
fi

# --- Done ---
log "Deployment finished!"

# Print outputs
echo ""
aws cloudformation describe-stacks --stack-name "$STACK_NAME" --region "$REGION" \
    --query 'Stacks[0].Outputs[].[Description, OutputValue]' --output table
