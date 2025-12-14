# WebRTC Connection Issue - Diagnosis and Fix

## Problem
The P2P Tldraw app showed "Disconnected • 0 peers" even when multiple instances were running.

## Root Causes Found

### 1. Public Signaling Servers Are Down
All the default WebRTC signaling servers were failing:
- `wss://signaling.yjs.dev/` - DNS resolution failed (ERR_NAME_NOT_RESOLVED)
- `wss://y-webrtc-signaling-eu.herokuapp.com/` - 404 error
- `wss://y-webrtc-signaling-us.herokuapp.com/` - 404 error  
- `wss://y-webrtc-eu.fly.dev/` - 503 service unavailable
- Other public servers also returned 308, 410 errors

### 2. Wrong Event Handler Structure
The code was checking for `status === 'connected'` but y-webrtc actually emits:
```javascript
{ connected: true }  // not { status: 'connected' }
```

## Solution Implemented

### 1. Local Signaling Server
Created `signaling-server.js` - a local WebSocket signaling server running on port 4444.

**To run:**
```bash
npm run signaling
```

### 2. Fixed Event Handler
Updated `src/yjs/YjsProvider.tsx`:
```typescript
// OLD (wrong):
const handleStatus = ({ status }: { status: string }) => {
  setIsConnected(status === 'connected')
}

// NEW (correct):
const handleStatus = (event: { connected: boolean }) => {
  setIsConnected(event.connected)
}
```

### 3. Configured App to Use Local Server
Updated `src/yjs/YjsProvider.tsx` to use local signaling:
```typescript
const config: TransportConfig = {
  roomId,
  password,
  signalingServers: ['ws://localhost:4444']
}
```

## Testing Results

✅ Signaling server successfully:
- Accepted WebSocket connections
- Subscribed clients to rooms
- Broadcast messages between peers

✅ Console logs showed:
- `[Signaling] Client subscribed to room: test-room-123`
- `[Signaling] Broadcasting to 1 peers in room: test-room-123`

✅ WebRTC provider emitted correct events:
- Status event: `{"connected":true}`
- Peers events fired correctly

## How to Test

1. Start the signaling server:
```bash
npm run signaling
```

2. In separate terminals, run 2+ instances:
```bash
npm run dev
```

3. In each window:
   - Click "Quick Join: test-room-123" button
   - Both should show "Connected • 1 peer" (or more)
   - Drawing in one window should sync to others

## Notes

- The local signaling server must be running for WebRTC to work
- All instances must use the same room ID to connect
- The signaling server logs show connection activity
- For production, deploy a persistent signaling server or use reliable paid services

