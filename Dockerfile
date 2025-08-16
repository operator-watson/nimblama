FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# Install build tools, Python, SSH client, and ccache
RUN apt-get update && apt-get install -y --no-install-recommends \
  git build-essential cmake ninja-build pkg-config \
  python3 python3-pip curl ca-certificates \
  openssh-client ccache gdb \
  && rm -rf /var/lib/apt/lists/*

# VS Code remote user defaults
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && mkdir -p /home/$USERNAME/.vscode-server \
  && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Enable ccache globally
ENV CCACHE_DIR=/home/$USERNAME/.ccache
ENV PATH="/usr/lib/ccache:$PATH"

# Configure SSH for GitHub when using agent forwarding
RUN mkdir -p /home/$USERNAME/.ssh && chmod 700 /home/$USERNAME/.ssh \
  && echo "Host github.com\n  User git\n  StrictHostKeyChecking accept-new\n" \
  > /home/$USERNAME/.ssh/config \
  && chown -R $USERNAME:$USERNAME /home/$USERNAME/.ssh

WORKDIR /workspaces/nimblama
USER $USERNAME

CMD ["bash"]
