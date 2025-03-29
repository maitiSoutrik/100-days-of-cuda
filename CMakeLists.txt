cmake_minimum_required(VERSION 3.10)
project(100-days-of-cuda LANGUAGES CXX CUDA)

# CI/CD Documentation
# ==================
#
# This project uses GitHub Actions for CI/CD to automate deployment to a Jetson Nano.
#
# Workflow: Deploy to Jetson Nano
# -----------------------------
# The workflow automatically deploys CUDA code to the Jetson Nano when pushing to
# the `development` or `main` branches.
#
# Prerequisites:
# 1. SSH Key for Jetson Nano Access:
#    - Generate an SSH key pair: `ssh-keygen -t ed25519 -C "github-actions-jetson"`
#    - Add public key to Jetson Nano: `ssh-copy-id -i ~/.ssh/id_ed25519.pub user@jetson-ip`
#    - Add private key as GitHub Secret named `JETSON_SSH_KEY`
#
# 2. Network Considerations:
#    - Ensure Jetson Nano has a static IP address
#    - GitHub Actions runners need network access to the Jetson Nano
#    - Consider port forwarding or using a self-hosted runner if behind NAT/firewall
#
# 3. Self-Hosted Runner Option:
#    - Set up a self-hosted runner on your local network if Jetson is not internet-accessible
#    - Update workflow to use: `runs-on: self-hosted`
#
# Troubleshooting:
# - SSH Connection Issues: Verify SSH key is correctly added to Jetson and GitHub secrets
# - Build Failures: Check that all dependencies are installed on the Jetson Nano
# - Network Access: Confirm GitHub Actions runner can reach your Jetson Nano's IP address

# Add subdirectories for each day's project
add_subdirectory(day001)
add_subdirectory(day002)
add_subdirectory(day003)
add_subdirectory(day004)
add_subdirectory(day005)
add_subdirectory(day006)
add_subdirectory(day007)
add_subdirectory(day008)
add_subdirectory(day009)
add_subdirectory(day010)
add_subdirectory(day011)
add_subdirectory(day012)
add_subdirectory(day013)
add_subdirectory(day014)
add_subdirectory(day015)
add_subdirectory(day016)
add_subdirectory(day017)
add_subdirectory(day018)
add_subdirectory(day019)
# Add a badge for each completed day to README.md
# etc.
