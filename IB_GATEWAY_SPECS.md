# Interactive Brokers Gateway Automated Launch Specifications

This document outlines methods for programmatically launching Interactive Brokers Gateway with automated credential entry for full unattended operation.

## Overview

The Interactive Brokers Gateway can be fully automated using specialized tools that handle login credentials, dialog management, and headless operation. The primary solution is **IBC (Interactive Brokers Controller)**, which enables hands-free operation for automated trading systems.

## Solution 1: IBC (Interactive Brokers Controller)

### Key Features
- **Automated Login**: Automatically fills username/password and clicks login button
- **Dialog Management**: Handles various TWS dialogs that would require user intervention
- **Cross-Platform**: Works on Windows, macOS, and Linux
- **Indefinite Runtime**: Keeps Gateway running continuously
- **Scheduled Shutdown**: Can terminate at specified times/days

### Installation Requirements
- **CRITICAL**: Must use **offline/standalone** TWS version - IBC does NOT work with self-updating TWS
- Gateway installer works normally (no self-updating version exists)
- English language setting required
- Download: https://github.com/IbcAlpha/IBC/releases/latest

### Configuration

#### Basic Configuration File (config.ini)
```ini
# Login credentials
IbLoginId=your_username
IbPassword=your_password

# Application to start (TWS or GATEWAY)
IbAutoCloseDown=no
IbExistingSessionDetectedAction=secondary
ClosedownAt=
SuppressInfoMessages=yes

# API Settings
AcceptIncomingConnectionAction=accept
AllowBlindTrading=yes
```

#### Environment Variables
```bash
export TWS_USERID=your_username
export TWS_PASSWORD=your_password
export TRADING_MODE=live  # or 'paper'
export TWS_MAJOR_VRSN=1031
export IBC_INI=~/ibc/config.ini
export IBC_PATH=~/ibc
export TWS_PATH=/opt/ibgateway
```

#### Launch Command
```bash
/opt/ibc/scripts/displaybannerandlaunch.sh 1031 \
  --tws-path=/opt/ibgateway \
  --ibc-path=/opt/ibc \
  --ibc-ini=/home/ibuser/ibc/config.ini \
  --mode=gateway
```

## Solution 2: Docker Automation

### Popular Docker Images

#### 1. IBGA (IB Gateway Automation)
**Features:**
- Full headless operation with VNC access
- TOTP/2FA automation support
- Screen scraping with JAuto JVMTI agent
- Automated mouse/keyboard input via xdotool

```bash
docker run -d \
  --name ibga \
  -p 4001:4001 \
  -p 5900:5900 \
  -e TWS_USERID=your_username \
  -e TWS_PASSWORD=your_password \
  -e TOTP_KEY=your_totp_secret \
  -e TRADING_MODE=live \
  ibga:latest
```

#### 2. gnzsnz/ib-gateway-docker
**Features:**
- IBC integration for dialog handling
- Xvfb virtual framebuffer for headless graphics
- VNC server for remote access

```yaml
# docker-compose.yml
version: '3.8'
services:
  ib-gateway:
    image: gnzsnz/ib-gateway-docker:latest
    environment:
      - TWS_USERID=your_username
      - TWS_PASSWORD=your_password
      - TRADING_MODE=live
      - VNC_SERVER_PASSWORD=secret
    ports:
      - "4001:4001"  # IB API
      - "5900:5900"  # VNC
    restart: unless-stopped
```

#### 3. datawookie/ib-gateway
```bash
docker run -d \
  --name ib-gateway \
  --env-file .env \
  --restart always \
  --network host \
  datawookie/ib-gateway:1016.1h
```

### Required Components for Headless Operation
- **Xvfb**: Virtual X11 framebuffer (Gateway requires X11)
- **x11vnc**: VNC server for remote GUI access
- **JAuto**: JVMTI agent for screen element detection
- **xdotool**: Simulates keyboard/mouse input
- **oathtool**: Generates TOTP codes for 2FA

## Solution 3: Native Linux Shell Scripts

### Basic Launch Script (runibg.sh)
```bash
#!/bin/bash
# Make executable: chmod 750 runibg.sh

USERNAME="$1"
PASSWORD="$2"
GATEWAY_PATH="/opt/ibgateway"

# Background execution with nohup
nohup "$GATEWAY_PATH/ibgateway" \
  -username="$USERNAME" \
  -password="$PASSWORD" \
  > gateway.log 2>&1 &

echo "IB Gateway started in background. PID: $!"
```

### Advanced Launch with IBC
```bash
#!/bin/bash
export TWS_MAJOR_VRSN=1031
export IBC_INI="$HOME/ibc/config.ini"
export IBC_PATH="$HOME/ibc"
export TWS_PATH="/opt/ibgateway"
export TWSUSERID="your_username"
export TWSPASSWORD="your_password"

# Launch with IBC
"$IBC_PATH/scripts/displaybannerandlaunch.sh" $TWS_MAJOR_VRSN
```

## Two-Factor Authentication (2FA) Considerations

### Current Limitations
- **Manual Intervention Required**: IBC cannot handle hardware tokens or SMS codes
- **IBKR Mobile App**: Can complete login via push notification
- **TOTP Support**: Some Docker solutions (IBGA) support automated TOTP generation

### Recommended 2FA Setup
1. **Request 2FA Relaxation**: Contact IB support to disable 2FA for API access
2. **Use IBKR Mobile**: Configure for push notification authentication
3. **TOTP Integration**: Use Docker solutions with TOTP support

### IBGA TOTP Configuration
```yaml
environment:
  - TOTP_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

## API Configuration Requirements

### Gateway Settings
- **Enable API**: Check "Enable ActiveX and Socket Clients"
- **Port Configuration**: Default 4001 (live), 4002 (paper)
- **Localhost Only**: Restrict connections to 127.0.0.1 for security

### Network Security
```bash
# Port mapping for Docker
-p 127.0.0.1:4001:4001  # Localhost only
# vs
-p 4001:4001           # Network accessible (INSECURE)
```

## Best Practices

### Security
1. **Store Credentials Securely**: Use environment files, not hardcoded values
2. **Restrict Network Access**: Bind API port to localhost only
3. **Use Encrypted Storage**: Store configuration files in encrypted directories
4. **Regular Updates**: Keep IBC and Gateway versions current

### Reliability
1. **Health Checks**: Implement connection monitoring
2. **Restart Policies**: Configure automatic restart on failure
3. **Log Management**: Capture and rotate Gateway logs
4. **Resource Monitoring**: Monitor memory/CPU usage

### Development Workflow
```bash
# 1. Development with paper trading
export TRADING_MODE=paper
docker-compose -f docker-compose.paper.yml up -d

# 2. Production deployment
export TRADING_MODE=live
docker-compose up -d

# 3. Monitor via VNC (optional)
# Connect to localhost:5900 with VNC client
```

## Troubleshooting

### Common Issues
1. **Self-Updating TWS**: Ensure offline/standalone version
2. **Java Version**: Use recommended Java version for Gateway
3. **X11 Dependencies**: Install xvfb for headless operation
4. **Permission Errors**: Check file permissions for config files

### Monitoring Commands
```bash
# Check Gateway process
ps aux | grep gateway

# Monitor API connections
netstat -an | grep 4001

# View Docker logs
docker logs ib-gateway

# VNC connection test
vncviewer localhost:5900
```

## Integration with Trading Systems

### Connection Pattern
```python
from ib_insync import IB

# Connect to automated Gateway
ib = IB()
ib.connect('127.0.0.1', 4001, clientId=1)

# Verify connection
print(f"Connected: {ib.isConnected()}")
```

### Health Check Integration
```bash
#!/bin/bash
# health_check.sh
nc -z localhost 4001
if [ $? -eq 0 ]; then
    echo "Gateway API accessible"
else
    echo "Gateway API not responding"
    # Restart logic here
fi
```

## Conclusion

Full automation of IB Gateway launch is achievable using:
1. **IBC** for credential handling and dialog management
2. **Docker containers** for complete headless deployment
3. **Shell scripts** for simple local automation

The most robust solution combines Docker + IBC + TOTP support for completely unattended operation, suitable for production automated trading systems.