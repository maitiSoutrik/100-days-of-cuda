cmake_minimum_required(VERSION 3.10)
project(100-days-of-cuda LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)

# Enable CTest at the root level so all subdirectory tests are discoverable
enable_testing()

# Google Test support using FetchContent
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG    release-1.12.1 # Or a more recent stable tag, e.g. main or a specific release
)
# For newer versions of CMake (3.11+), FetchContent_MakeAvailable is preferred.
# For older versions, you might need to use FetchContent_GetProperties and manually add_subdirectory.
FetchContent_MakeAvailable(googletest)


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
add_subdirectory(day020)
add_subdirectory(day021)
add_subdirectory(day022)
add_subdirectory(day023)
add_subdirectory(day024)
add_subdirectory(day025)
add_subdirectory(day026)
add_subdirectory(day027)
add_subdirectory(day028)
add_subdirectory(day029)
add_subdirectory(day030)
add_subdirectory(day031)
add_subdirectory(day032)
add_subdirectory(day033)
add_subdirectory(day034)
add_subdirectory(day035)
add_subdirectory(day036)
add_subdirectory(day037)
add_subdirectory(day038)
add_subdirectory(day039)
add_subdirectory(day040)
add_subdirectory(day041)
add_subdirectory(day042)
add_subdirectory(day043)
add_subdirectory(day044)
add_subdirectory(day045)
add_subdirectory(day046)
add_subdirectory(day047)
add_subdirectory(day048)
add_subdirectory(day049)
add_subdirectory(day050)
add_subdirectory(day051)
add_subdirectory(day052)
add_subdirectory(day053)
add_subdirectory(day054)
add_subdirectory(day055)
add_subdirectory(day056)
add_subdirectory(day057)
add_subdirectory(day058)
add_subdirectory(day059)
add_subdirectory(day060)
add_subdirectory(day061)
add_subdirectory(day062)
add_subdirectory(day063)
add_subdirectory(day064)
add_subdirectory(day065)
add_subdirectory(day066)
add_subdirectory(day067)
add_subdirectory(day068)
add_subdirectory(day069)
add_subdirectory(day070)
add_subdirectory(day071)
add_subdirectory(day072)
add_subdirectory(day073)
add_subdirectory(day074)
add_subdirectory(day075)
add_subdirectory(day076)
add_subdirectory(day077)
add_subdirectory(day078)
add_subdirectory(day079)
add_subdirectory(day080)
# Add a badge for each completed day to README.md
# etc.
