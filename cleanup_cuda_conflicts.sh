#!/bin/bash

# CUDA Cleanup Script - Remove conflicting CUDA installations
# This script removes Ollama's CUDA libraries that conflict with system CUDA

echo "🧹 CUDA Cleanup Script - Removing conflicting installations"
echo "=========================================================="

# Check current CUDA installations
echo "📋 Current CUDA libraries found:"
find /usr -name "libcudart*" 2>/dev/null | sort

echo ""
echo "📋 Ollama CUDA directories:"
find /usr/local/lib -name "*ollama*" -type d 2>/dev/null

echo ""
echo "⚠️  This will remove Ollama's CUDA libraries but keep system CUDA"
read -p "Continue? (y/N): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

echo "🗑️  Removing Ollama CUDA libraries..."

# Remove Ollama CUDA directories
if [ -d "/usr/local/lib/ollama" ]; then
    echo "  - Removing /usr/local/lib/ollama"
    sudo rm -rf /usr/local/lib/ollama
else
    echo "  - /usr/local/lib/ollama not found"
fi

# Remove any other Ollama CUDA paths
for path in $(find /usr/local -name "*ollama*" -type d 2>/dev/null); do
    if [[ "$path" == *"cuda"* ]]; then
        echo "  - Removing $path"
        sudo rm -rf "$path"
    fi
done

# Clear LD_LIBRARY_PATH of Ollama paths
echo "🔄 Cleaning environment variables..."
if [ -f ~/.bashrc ]; then
    sed -i '/ollama.*cuda/d' ~/.bashrc
    echo "  - Cleaned ~/.bashrc"
fi

if [ -f ~/.profile ]; then
    sed -i '/ollama.*cuda/d' ~/.profile
    echo "  - Cleaned ~/.profile"
fi

# Update library cache
echo "📚 Updating library cache..."
sudo ldconfig

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📋 Remaining CUDA libraries:"
find /usr -name "libcudart*" 2>/dev/null | sort

echo ""
echo "🔄 Next steps:"
echo "1. Reboot your system: sudo reboot"
echo "2. Test CUDA: cd /home/soso/godot-llama-cpp && ./cuda_runtime_test"
echo "3. Run Godot and test GPU acceleration"

echo ""
echo "📝 If Ollama stops working, reinstall it with:"
echo "   curl -fsSL https://ollama.ai/install.sh | sh"