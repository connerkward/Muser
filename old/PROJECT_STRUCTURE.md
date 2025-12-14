# Project Structure

## Overview
P2P Tldraw Electron application with WebRTC-based collaboration.

## Directory Structure

```
/
├── electron/
│   ├── main.ts          # Electron main process
│   └── preload.ts       # Preload script for context isolation
├── src/
│   ├── yjs/
│   │   ├── YjsProvider.tsx    # React context provider for Yjs
│   │   └── transports.ts      # Transport manager (WebRTC, IndexedDB)
│   ├── components/
│   │   ├── TldrawCanvas.tsx   # Tldraw canvas with Yjs binding
│   │   ├── ConnectionStatus.tsx # Connection indicator
│   │   └── RoomManager.tsx    # Room creation/joining UI
│   ├── App.tsx          # Main application component
│   └── main.tsx         # React entry point
├── index.html           # HTML template
├── electron.vite.config.ts  # Electron Vite configuration
├── tsconfig.json        # TypeScript configuration
└── package.json         # Dependencies and scripts
```

## Key Components

### YjsProvider
- Manages Yjs document lifecycle
- Initializes WebRTC and IndexedDB providers
- Provides connection status and peer count
- Context API for accessing Yjs throughout the app

### TransportManager
- Abstracts transport layer (WebRTC, IndexedDB)
- Configurable signaling servers
- Clean connection/disconnection handling
- Designed for easy transport swapping

### TldrawCanvas
- Integrates Tldraw with Yjs
- Bidirectional sync: Tldraw ↔ Yjs
- Handles real-time collaboration updates
- Manages store lifecycle

### RoomManager
- UI for creating new rooms (generates random ID)
- UI for joining existing rooms
- Clean, modern interface

### ConnectionStatus
- Visual indicator of connection state
- Displays peer count
- Fixed position overlay

## Data Flow

1. User creates/joins room → RoomManager
2. App initializes YjsProvider with roomId
3. YjsProvider creates Y.Doc and connects transports
4. TldrawCanvas creates Tldraw store and syncs with Yjs
5. Changes in Tldraw → Yjs → WebRTC → Other peers
6. Changes from peers → WebRTC → Yjs → Tldraw

## Running the App

```bash
# Development mode (with hot reload)
npm run dev

# Build for production
npm run build

# Run production build
npm start
```

## Transport Swapping

The `YjsTransportManager` is designed to support multiple transports:
- Currently: WebRTC (y-webrtc) + IndexedDB (y-indexeddb)
- Future: WebSocket (y-websocket), custom transports
- Change transport by modifying `transports.ts` and `YjsProvider.tsx`


