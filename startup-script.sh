#!/usr/bin/env bash

# Define source and destination directories
XDG_SRC_BASE="/workspace/.xdg"
XDG_DEST_CONFIG="${XDG_CONFIG_HOME:-$HOME/.config}"
XDG_DEST_DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
XDG_DEST_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}"
XDG_DEST_STATE="${XDG_STATE_HOME:-$HOME/.local/state}"

# Function to copy contents if source exists
copy_dir() {
  local src="${1}"
  local dest="${2}"
  local exclude="${3}"
  if [ -d "${src}" ]; then
    echo "Copying from ${src} to ${dest}"
    mkdir -p "${dest}"
    if [ -n "${exclude}" ]; then
      rsync -av --exclude="${exclude}" "${src}/" "${dest}/"
    else
      rsync -av "${src}/" "${dest}/"
    fi
  else
    echo "Source directory ${src} does not exist. Skipping."
  fi
}

# Set up SSH authorized keys
mkdir -pv ~/.ssh
chmod -v 700 ~/.ssh
echo "ssh-rsa AAAAB3NzaC1yc2EAAAABIwAAAQEAxP5jcpfSLiS6+AbdvRCFxtBFz3tiy4fjEVxdIcsNAmu3wzZXECO1OZIV3dpgUBmyorl6iCn4unm8bpd5qS+5MLowOCZm7idETfod5A5U5i0nqGu2IMWmhgqYCMg3JZqrwub878ZJlJC0UPUJhA1akWsoQslatRXkx6p94OVCWWojUe2BoE4wZt8JZuzQ7D7EvzN+K70PRiZ9sHpwLhizCw9OVxVe3G2MX5xyRsM+Igv1s9B7FhmJr7RM4aRotyezfbGNjT0Wroz/c0kUysHMI96m1+GYShIJsXybINPZGjtKkXfNJ6OuhC9EVWgbUXzHsw0KIu65JNlOJJt/tzWXMw== hunmonk@colossus" > ~/.ssh/authorized_keys
echo "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQDdtaFtjR25oiTPK0tZ7YaeaqghNPk/yAhExR6csldRapuqvmVqkuZlF/V/D+FTeyBH5ixvM0JWSMaBtsNPis32v6tQtQIEqmgvoz/7eESNOigbBE2184Vz2Gxn+xk7QK50wTtm8TPz62mUqelIhM/LfJfXWuhc5Qvcw86rZkVtwGycJLzpGwkTCS58yKlxiYSQW91+50WrY6c4JVz/KhRIuuPghydr2ZohQ4rpWFAFJMpvqhgHR7bn6n1XCutcwRvWdX+Gg2/XZ2kuNG5sHd2zskYopLs1xrL6Dg0ZC7nd05IKuAS2PolA2xCbJxFkMpXTYsZ9V9x+I4oe3NDlQdD59xiwGifbc/Dbyo4OehO1FlGnUO4N+0Ry/2TczqkLTDJW/bSkxaNrId+ZkSl+S4LMXSASssqBCjrXutVpx6QStPdTJ3vnnIciveXbdYeY8gG2iYnfSjlseHHMNdQ4VB/hXvddYGV5reI0NXqW/Mzl/8ETJbM9iqOglSFVwyMHQ+FNT0Bwtc9ZtlUzAnS6ow0WXUR79Ebws89UXr+HTdYxinfrcmdS9M8v015OC8yFzVCOOCghwpbrLsYzdyBf2aGHuXZn+4H8sQ9duOj0V7nxOC8aEhFH8lmupPzi+HTOumC5xy0QxD3Wzt3PFtp27ERhFEOwNRGLvp11lNZO3DZyhQ== adamgabriel@new-host-4.home" >> ~/.ssh/authorized_keys
chmod -v 600 ~/.ssh/authorized_keys

# Set SSH private key location.
mkdir -pv /tmp/.ssh
cp -v /workspace/.ssh/id_rsa /tmp/.ssh/id_rsa
chmod -v 600 /tmp/.ssh/id_rsa

# XDG storage
mkdir -pv /workspace/.xdg/{config,data,cache,state}

# Login customizations
if ! grep -q "CUSTOMIZATIONS" ~/.bashrc; then
  echo "Adding customizations to ~/.bashrc"
  cat >> ~/.bashrc << 'EOF'

# CUSTOMIZATIONS
if [ -r /workspace/.shell-customizations ]; then
  source /workspace/.shell-customizations
fi
EOF
fi

# Install Apartment Lines repo
mkdir -pv /etc/apt/keyrings
curl -fsSL https://s3.amazonaws.com/apartmentlines-repo-debian/repo.pub > /etc/apt/keyrings/apartmentlines-repo-debian.pub
chmod -v 644 /etc/apt/keyrings/apartmentlines-repo-debian.pub
cat > /etc/apt/sources.list.d/apartmentlines.list << 'EOF'
deb [signed-by=/etc/apt/keyrings/apartmentlines-repo-debian.pub] https://s3.amazonaws.com/apartmentlines-repo-debian bookworm main
EOF
cat > /etc/apt/preferences.d/apartmentlines << 'EOF'
Package: *
Pin: origin s3.amazonaws.com
Pin-Priority: 900

Package: *
Pin: release n=bookworm, o=Debian
Pin-Priority: 500
EOF

# Update
apt update

# Install packages
apt install -y \
  aptitude \
  curl \
  git \
  golang-1.23-go \
  less \
  lua5.4 \
  neovim \
  nodejs \
  npm \
  rsync \
  ruby \
  ruby-dev \
  tmux \
  tree \
  vim \
  unzip

# Git credential helper for HuggingFace huggingface_hub
git config --global credential.helper store
git config --global user.email "chad@apartmentlines.com"
git config --global user.name "Chad Phillips"

# Install Python packages
pip install -U \
  accelerate \
  bitsandbytes \
  datasets \
  fsspec \
  ninja \
  gcsfs \
  huggingface_hub \
  peft \
  prompt-toolkit>=3 \
  protobuf \
  pyyaml \
  sentencepiece \
  tenacity \
  torch \
  transformers \
  triton==3.2.0 \
  trl

# Install Neovim config
if [ ! -d /workspace/.xdg/config/nvim ]; then
  cd /workspace/.xdg/config && git clone https://github.com/thehunmonkgroup/lazyvim-config.git nvim
else
  cd /workspace/.xdg/config/nvim && git pull
fi

# Copy over XDG directories to home
# copy_dir "$XDG_SRC_BASE/config" "$XDG_DEST_CONFIG"
# copy_dir "$XDG_SRC_BASE/data" "$XDG_DEST_DATA"
# copy_dir "$XDG_SRC_BASE/cache" "$XDG_DEST_CACHE" "huggingface/"
# copy_dir "$XDG_SRC_BASE/state" "$XDG_DEST_STATE"

echo "Migration completed."

# Start SSH service
ssh-keygen -A
service ssh start

if [ "${1}" = "docker" ]; then
  sleep infinity
fi
