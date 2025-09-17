#!/bin/bash

source .env

# Configuration
INSTANCE_TYPE=gpu_1x_a100_sxm4  
SSH_NAME=scoobdoob
SSH_PATH=~/.ssh/scoobdoob.pem
PROJECT_NAME=energy_mpc_toy

echo "🚀 Setting up Energy MPC training on Lambda Labs..."

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
echo "📁 Syncing project code..."
rsync -av --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.env' \
    --exclude='wandb' \
    --exclude='outputs' \
    --exclude='data' \
    -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i ${SSH_PATH}" \
    . ubuntu@${INSTANCE_IP}:~/${PROJECT_NAME}/

echo "✅ Code synced"

# Create and run setup script
echo "⚙️  Setting up environment and starting training..."

cat > setup_and_train.sh << 'EOF'
#!/bin/bash
set -e

PROJECT_NAME=$1
WANDB_API_KEY=$2

echo "🐍 Setting up Python environment..."

# Update system and install python3-venv if needed
sudo apt-get update -qq
sudo apt-get install -y python3-venv python3-pip

# Create virtual environment
cd ~/${PROJECT_NAME}
python3 -m venv mpc_env

# Activate virtual environment
source mpc_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project in development mode (gets all dependencies from pyproject.toml)
pip install -e .

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export WANDB_API_KEY=$WANDB_API_KEY
export CUDA_VISIBLE_DEVICES=0

echo "🔥 Starting training..."
echo "GPU Info:"
nvidia-smi

echo "Python Info:"
python --version
pip list | grep torch

# Run training (model will auto-save to WandB)
python training/train.py

echo "✅ Training completed!"
EOF

# Copy and run the setup script
scp -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_PATH \
    setup_and_train.sh ubuntu@${INSTANCE_IP}:~/

# Clean up local setup script
rm -f setup_and_train.sh

ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $SSH_PATH \
    ubuntu@${INSTANCE_IP} \
    "chmod +x setup_and_train.sh && ./setup_and_train.sh ${PROJECT_NAME} ${WANDB_API_KEY}"

echo "🎉 Training job completed!"

# Note: cleanup will run automatically due to the trap