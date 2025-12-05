#!/bin/bash
source .env

# Configuration
INSTANCE_TYPE=gpu_1x_h100_pcie #gpu_8x_a100_80gb_sxm4
SSH_NAME=scoobdoob
SSH_PATH=~/.ssh/scoobdoob.pem
GCS_BUCKET=energy-mpc
LOCAL_CACHE=/tmp/datasets
DATASET_NAME=action_dataset
PROJECT_NAME=energy_mpc_toy


echo "🚀 Setting up action VAE training on Lambda Labs..."

# Get available region
echo "📍 Finding available region for $INSTANCE_TYPE..."
REGION=$(curl -s -u $LAMBDA_LABS_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-types \
    | jq -r --arg instance_type "$INSTANCE_TYPE" \
    '.data | .[$instance_type] | .regions_with_capacity_available | .[0] | .name')

if [ "$REGION" = "null" ]; then
    echo "❌ No capacity available for $INSTANCE_TYPE"
    exit 1
fi

echo "✅ Found capacity in region: $REGION"

# Create launch config
jq -n --arg region "$REGION" \
      --arg ssh_name "$SSH_NAME" \
      --arg instance_type "$INSTANCE_TYPE" \
    '{
        "region_name": $region,
        "instance_type_name": $instance_type,
        "ssh_key_names": [$ssh_name],
        "file_system_names": [],
        "quantity": 1
    }' > lambda-config.json

# Launch instance with retry logic
echo "🔄 Launching instance..."
INSTANCE_ID=""
MAX_LAUNCH_RETRIES=5

for attempt in $(seq 1 $MAX_LAUNCH_RETRIES); do
    echo "   Launch attempt $attempt/$MAX_LAUNCH_RETRIES..."
    
    LAUNCH_RESPONSE=$(curl -s -u "$LAMBDA_LABS_API_KEY": https://cloud.lambdalabs.com/api/v1/instance-operations/launch \
        -d @lambda-config.json -H "Content-Type: application/json")
    
    INSTANCE_ID=$(echo $LAUNCH_RESPONSE | jq -r '.data.instance_ids[0]')
    
    if [ "$INSTANCE_ID" != "null" ] && [ "$INSTANCE_ID" != "" ]; then
        echo "✅ Instance launched: $INSTANCE_ID"
        break
    fi
    
    echo "   Launch failed, response:"
    echo $LAUNCH_RESPONSE | jq '.error'
    
    if [ $attempt -lt $MAX_LAUNCH_RETRIES ]; then
        echo "   Waiting 30s before retry..."
        sleep 30s
        
        # Refresh region availability for retry
        echo "   Checking for new capacity..."
        NEW_REGION=$(curl -s -u $LAMBDA_LABS_API_KEY: https://cloud.lambdalabs.com/api/v1/instance-types \
            | jq -r --arg instance_type "$INSTANCE_TYPE" \
            '.data | .[$instance_type] | .regions_with_capacity_available | .[0] | .name')
        
        if [ "$NEW_REGION" != "null" ] && [ "$NEW_REGION" != "$REGION" ]; then
            echo "   Found capacity in new region: $NEW_REGION"
            REGION=$NEW_REGION
            # Regenerate config with new region
            jq -n --arg region "$REGION" \
                  --arg ssh_name "$SSH_NAME" \
                  --arg instance_type "$INSTANCE_TYPE" \
                '{
                    "region_name": $region,
                    "instance_type_name": $instance_type,
                    "ssh_key_names": [$ssh_name],
                    "file_system_names": [],
                    "quantity": 1
                }' > lambda-config.json
        fi
    fi
done

if [ "$INSTANCE_ID" = "null" ] || [ "$INSTANCE_ID" = "" ]; then
    echo "❌ Failed to launch instance after $MAX_LAUNCH_RETRIES attempts"
    rm lambda-config.json
    exit 1
fi

# Wait for instance to be ready
echo "⏳ Waiting for instance to be ready..."
sleep 3m

# Get IP address with retry logic
echo "🔍 Getting instance IP..."
INSTANCE_IP=""
MAX_IP_RETRIES=15

for attempt in $(seq 1 $MAX_IP_RETRIES); do
    echo "   IP attempt $attempt/$MAX_IP_RETRIES..."
    
    IP_RESPONSE=$(curl -s -u $LAMBDA_LABS_API_KEY: https://cloud.lambdalabs.com/api/v1/instances/${INSTANCE_ID})
    INSTANCE_IP=$(echo $IP_RESPONSE | jq -r '.data.ip')
    
    if [ "$INSTANCE_IP" != "null" ] && [ "$INSTANCE_IP" != "" ]; then
        echo "✅ Instance IP: $INSTANCE_IP"
        break
    fi
    
    if [ $attempt -lt $MAX_IP_RETRIES ]; then
        echo "   No IP yet, waiting 20s..."
        sleep 20s
    fi
done

if [ "$INSTANCE_IP" = "null" ] || [ "$INSTANCE_IP" = "" ]; then
    echo "❌ Failed to get instance IP after $MAX_IP_RETRIES attempts"
    exit 1
fi

# Wait for SSH to be ready
echo "⏳ Waiting for SSH to be ready..."
sleep 6m

# Test SSH connection with retry logic
echo "🔌 Testing SSH connection..."
SSH_READY=false
MAX_SSH_RETRIES=20

for attempt in $(seq 1 $MAX_SSH_RETRIES); do
    echo "   SSH attempt $attempt/$MAX_SSH_RETRIES..."
    
    if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 \
           -i $SSH_PATH ubuntu@${INSTANCE_IP} 'echo "SSH ready"' >/dev/null 2>&1; then
        echo "✅ SSH connection successful"
        SSH_READY=true
        break
    fi
    
    if [ $attempt -lt $MAX_SSH_RETRIES ]; then
        echo "   SSH not ready, waiting 30s..."
        sleep 30s
    fi
done

if [ "$SSH_READY" = false ]; then
    echo "❌ SSH connection failed after $MAX_SSH_RETRIES attempts"
    echo "   Instance might not be fully booted. Try manually connecting:"
    echo "   ssh -i $SSH_PATH ubuntu@${INSTANCE_IP}"
    exit 1
fi

# Setup cleanup trap
cleanup() {
    echo "🧹 Cleaning up..."
    rm -f lambda-config.json
    
    if [ ! -z "$INSTANCE_ID" ]; then
        echo "🛑 Terminating instance $INSTANCE_ID..."
        curl -s -u "$LAMBDA_LABS_API_KEY": \
            https://cloud.lambdalabs.com/api/v1/instance-operations/terminate \
            -d "{\"instance_ids\": [\"$INSTANCE_ID\"]}" \
            -H "Content-Type: application/json" > /dev/null
        echo "✅ Instance terminated"
    fi
}

# Set trap for cleanup on exit
trap cleanup EXIT

# Sync project code
cp ~/.config/gcloud/servicekey.json ./servicekey.json

# Add to gitignore
echo "servicekey.json" >> .gitignore
echo "📁 Syncing project code..."
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='wandb' \
    --exclude='outputs' \
    --exclude='.venv' \
    --exclude='tests' \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${SSH_PATH}" \
    . ubuntu@${INSTANCE_IP}:~/${PROJECT_NAME}/

rm ./servicekey.json

echo "✅ Code synced"

# Create and run setup script
echo "⚙️  Setting up environment and starting training..."

cat > setup_and_train.sh << 'EOF'
#!/bin/bash
set -e

PROJECT_NAME=$1
WANDB_API_KEY=$2
GCS_BUCKET=$3 
LOCAL_CACHE=$4      
DATASET_NAME=$5

echo "⚡ Installing uv..."

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Add uv to PATH for this session
source $HOME/.local/bin/env

echo "🐍 Setting up Python environment with uv..."

# Navigate to project directory
cd ~/${PROJECT_NAME}

# Sync all dependencies (creates .venv and installs everything from pyproject.toml)
uv sync

# Set environment variables
export GOOGLE_APPLICATION_CREDENTIALS="./servicekey.json"
export GCS_BUCKET=${GCS_BUCKET}

export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$WANDB_API_KEY

echo "🔥 Starting training..."
echo "GPU Info:"
nvidia-smi

echo "Python Info:"
uv run python --version
uv pip list | grep torch

# Set up cloud authorizations.

if [ -f "./servicekey.json" ]; then
    echo "✅ Key file exists, size: $(wc -c < ./servicekey.json) bytes"
else
    echo "❌ Key file missing!"
    ls -la
    exit 1
fi

gcloud auth activate-service-account --key-file=./servicekey.json

# Get dataset. 

uv run python --version
uv pip list | grep torch

# Set up cloud authorizations.

if [ -f "./servicekey.json" ]; then
    echo "✅ Key file exists, size: $(wc -c < ./servicekey.json) bytes"
else
    echo "❌ Key file missing!"
    ls -la
    exit 1
fi

gcloud auth activate-service-account --key-file=./servicekey.json

# Get dataset. 
if gsutil ls "gs://$GCS_BUCKET/datasets/${DATASET_NAME}.tar.gz" &> /dev/null; then
    echo "Downloading compressed dataset from GCS..."
    mkdir -p "$LOCAL_CACHE"
    gsutil -m cp "gs://$GCS_BUCKET/datasets/${DATASET_NAME}.tar.gz" "$LOCAL_CACHE/"
    
    # Decompress
    echo "Decompressing dataset..."
    tar -I pigz -xf "$LOCAL_CACHE/${DATASET_NAME}.tar.gz" -C "$LOCAL_CACHE"
    
    echo "Cleaning up compressed file..."
    rm "$LOCAL_CACHE/${DATASET_NAME}.tar.gz"
    
    echo "✅ Dataset downloaded and decompressed"
else
    # Build locally if not in GCS
    echo "Dataset not in GCS, building locally..."
    cd ~/${PROJECT_NAME}
    uv run python dataset/action_dataset.py
    
    # Check if build succeeded
    if [ ! -d "$LOCAL_CACHE/$DATASET_NAME" ]; then
        echo "❌ Dataset build failed - directory not found at $LOCAL_CACHE/$DATASET_NAME"
        exit 1
    fi
    
    echo "✅ Dataset built successfully"
    
    # Compress and upload
    echo "Compressing dataset..."
    tar -I pigz -cf "$LOCAL_CACHE/${DATASET_NAME}.tar.gz" -C "$LOCAL_CACHE" "${DATASET_NAME}/"
    
    echo "Uploading compressed dataset..."
    gsutil -m cp "$LOCAL_CACHE/${DATASET_NAME}.tar.gz" "gs://$GCS_BUCKET/datasets/"
    
    echo "Cleaning up compressed file..."
    rm "$LOCAL_CACHE/${DATASET_NAME}.tar.gz"
    
    echo "✅ Dataset uploaded to GCS"
fi

# Continue with training
cd ~/${PROJECT_NAME}

# Train.

cd ~/${PROJECT_NAME}

uv run accelerate launch training/train_action_vae.py 

TRAIN_EXIT_CODE=$?

echo "✅ Training completed!"

# Upload to GCS (runs on the remote instance where files exist)

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "📤 Uploading to GCS..."
    
    if [ -d "data/out" ]; then
        gsutil -m cp -r data/out/ gs://$GCS_BUCKET/training/action-model/$(date +%Y%m%d_%H%M%S)/
        echo "✅ Upload complete"
    else
        echo "❌ No outputs found"
    fi
fi

EOF

# Copy and run the setup script
scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_PATH \
    setup_and_train.sh ubuntu@${INSTANCE_IP}:~/

# Clean up local setup script
rm -f setup_and_train.sh

# Pass GCS_BUCKET as third argument
ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_PATH \
    ubuntu@${INSTANCE_IP} \
    "chmod +x setup_and_train.sh && ./setup_and_train.sh \
    ${PROJECT_NAME} \
    ${WANDB_API_KEY} \
    ${GCS_BUCKET} \
    ${LOCAL_CACHE} \
    ${DATASET_NAME}"

echo "🎉 Training job completed!"

# Note: cleanup will run automatically due to the trap