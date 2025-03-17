# GitHub CI/CD for 100 Days of CUDA

This directory contains the GitHub Actions workflows for automating the deployment of CUDA code to the Jetson Nano.

## Workflow: Deploy to Jetson Nano

The `deploy-to-jetson.yml` workflow automatically deploys your CUDA code to your Jetson Nano whenever you push to the `development` or `main` branches.

### Prerequisites

Before this workflow can run successfully, you need to set up the following:

1. **SSH Key for Jetson Nano Access**:
   - Generate an SSH key pair on your local machine (if you don't already have one):

     ```bash
     ssh-keygen -t ed25519 -C "github-actions-jetson"
     ```

   - Add the public key to your Jetson Nano's `~/.ssh/authorized_keys` file:

     ```bash
     # On your local machine
     ssh-copy-id -i ~/.ssh/id_ed25519.pub drboom@192.168.1.184
     ```

   - Add the private key as a GitHub Secret (see below)

2. **GitHub Repository Secrets**:
   - Go to your GitHub repository → Settings → Secrets and variables → Actions
   - Add a new repository secret:
     - Name: `JETSON_SSH_KEY`
     - Value: *The entire content of your private key file (~/.ssh/id_ed25519)*

### Network Considerations

- **Static IP Address**: Ensure your Jetson Nano has a static IP address (currently set to `192.168.1.184`)
- **Network Access**: GitHub Actions runners need to be able to access your Jetson Nano
  - If your Jetson is behind a firewall or NAT, you may need to set up port forwarding
  - Alternatively, consider using a self-hosted GitHub Actions runner on your local network

### Self-Hosted Runner Option

If your Jetson Nano is not accessible from the internet, you can set up a self-hosted GitHub Actions runner on a machine in the same network:

1. On your GitHub repository, go to Settings → Actions → Runners
2. Click "New self-hosted runner"
3. Follow the instructions to set up a runner on a machine in your local network
4. Update the workflow file to use your self-hosted runner:

   ```yaml
   jobs:
     deploy:
       runs-on: self-hosted  # Instead of ubuntu-latest
   ```

## Troubleshooting

- **SSH Connection Issues**: Ensure the SSH key is correctly added to the Jetson Nano and the GitHub secret
- **Build Failures**: Check that all dependencies are installed on the Jetson Nano
- **Network Access**: Verify that the GitHub Actions runner can reach your Jetson Nano's IP address
