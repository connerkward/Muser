# Setup Guide

## Quick Start (No Server Required)

The app will attempt to use WebRTC P2P connections by default. However, WebRTC requires signaling servers which are often unreliable.

**Just run:**
```bash
npm run dev
```

The app will work, but connection may be unreliable due to public signaling server availability.

## Recommended: Run Local WebSocket Server (Reliable)

For reliable real-time sync, run the included WebSocket server:

### Terminal 1 - Start WebSocket Server
```bash
npm run signaling
```

This starts a WebSocket server on `ws://localhost:1234`

### Terminal 2+ - Start App Instances  
```bash
npm run dev
```

Run this in multiple terminals to test collaboration.

## How It Works

### With WebSocket Server (Recommended)
- All clients connect to `ws://localhost:1234`
- Server relays changes between clients
- Very reliable, low latency
- Requires running the signaling server

### Without Server (Fallback)
- Uses WebRTC peer-to-peer
- Attempts to use public signaling servers
- May fail if public servers are down
- No server required but less reliable

## Configuration

Set environment variable to use custom WebSocket server:
```bash
VITE_WS_URL=wss://your-server.com npm run dev
```

## Production Deployment

For production, you'll want to:
1. Deploy the signaling server (`signaling-server.js`) to a cloud service
2. Set `VITE_WS_URL` to your deployed server URL
3. Or use a paid WebRTC/WebSocket service

## Testing Collaboration

1. Start signaling server (optional but recommended)
2. Run `npm run dev` in 2+ terminals
3. In each window, click "Quick Join: test-room-123"
4. Draw in one window - it should appear in others
5. Status should show "Connected to Server"

