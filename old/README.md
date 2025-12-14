# P2P Tldraw Electron

A peer-to-peer collaborative drawing application built with Electron, React, Tldraw, and Yjs.

## Features

- **Infinite Canvas**: Powered by Tldraw
- **Real-time Sync**: WebSocket-based collaboration via y-websocket
- **User Presence**: See active users with random names and colors
- **Offline Persistence**: IndexedDB storage for offline work
- **Room-based Collaboration**: Join rooms by ID for isolated sessions

## Tech Stack

- Electron + React + TypeScript
- Tldraw for canvas
- Yjs for CRDT state management
- y-websocket for real-time synchronization
- y-indexeddb for local persistence
- y-protocols/awareness for user presence

## Getting Started

### Install Dependencies

```bash
npm install
```

### Start WebSocket Server (Required)

```bash
npm run signaling
```

This starts a WebSocket server on `ws://localhost:1234` for real-time sync.

### Development

Open multiple terminals and run:

```bash
npm run dev  # Terminal 1
npm run dev  # Terminal 2
npm run dev  # Terminal 3...
```

Or open in browser: `http://localhost:5183`

### Build

```bash
npm run build
npm start
```

## Usage

1. Start the signaling server: `npm run signaling`
2. Launch the app: `npm run dev` (or open browser to `http://localhost:5183`)
3. Click "Quick Join: test-room-123" for instant testing
4. Or click "Create New Room" to generate a new collaboration room
5. Share the room ID with others to collaborate
6. See active users in the bottom-right panel
7. Each user gets a random name (e.g., "Happy Panda") and color

## Architecture

- **YjsProvider**: Manages Yjs document, transport connections, and awareness
- **TransportManager**: Handles WebSocket and IndexedDB providers
- **TldrawCanvas**: Integrates Tldraw with Yjs for real-time sync
- **RoomManager**: UI for creating/joining rooms
- **ConnectionStatus**: Shows connection state
- **UsersList**: Displays active users with names and colors
- **useYjsPresence**: Hook for managing user presence data
- **signaling-server.js**: WebSocket relay server for real-time sync


