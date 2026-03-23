FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

# Raise file descriptor limits
RUN echo "* soft nofile 65536\n* hard nofile 65536" >> /etc/security/limits.conf

# Install just
RUN curl --proto '=https' --tlsv1.2 -sSf https://just.systems/install.sh | bash -s -- --to /usr/local/bin

# Install Quarto CLI
RUN QUARTO_VERSION=1.6.42 && \
    curl -LO "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-arm64.deb" && \
    dpkg -i "quarto-${QUARTO_VERSION}-linux-arm64.deb" && \
    rm "quarto-${QUARTO_VERSION}-linux-arm64.deb"

# Install AWS CLI v2
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# Install Chromium (remove stale yarn repo first to unblock apt-get update)
RUN rm -f /etc/apt/sources.list.d/yarn.list && \
    apt-get update && \
    apt-get install -y chromium && \
    apt-get clean
