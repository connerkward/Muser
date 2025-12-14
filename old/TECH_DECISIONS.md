# Technical and Design Decisions

This document captures all technical and design decisions made during the development of Muser (P2P Tldraw Electron application).

## Core Technology Stack

### Frontend Framework
- **React 19** - Modern React with latest features
- **TypeScript 5.9** - Type safety and developer experience
- **Decision Rationale**: Type safety prevents runtime errors, React provides component-based architecture

### Canvas & Drawing
- **Tldraw 4.2** - Infinite canvas drawing library
- **Decision Rationale**: 
  - Built-in infinite canvas capabilities
  - Rich drawing tools and shapes
  - Modern, performant rendering
  - Active development and community

### State Management & Synchronization
- **Yjs 13.6** - CRDT (Conflict-free Replicated Data Type) library
- **Decision Rationale**:
  - Automatic conflict resolution for concurrent edits
  - No need for operational transforms
  - Efficient binary protocol
  - Works offline and syncs when reconnected

### Desktop Application
- **Electron 39** - Cross-platform desktop app framework
- **Decision Rationale**:
  - Single codebase for multiple platforms
  - Access to native OS features
  - Web technologies (React/TypeScript) in desktop context
  - Large ecosystem and community

### Build Tools
- **Vite 7** - Fast build tool and dev server
- **electron-vite 4** - Electron-specific Vite integration
- **Decision Rationale**:
  - Fast HMR (Hot Module Replacement)
  - Optimized production builds
  - Modern ESM support
  - Better DX than Webpack

## Architecture Decisions

### Application Structure
- **Monorepo structure** with clear separation:
  - `electron/` - Main and preload processes
  - `src/` - React application code
  - `src/yjs/` - Yjs integration layer
  - `src/components/` - React components
- **Decision Rationale**: Clear separation of concerns, easy to navigate and maintain

### Context-Based State Management
- **React Context API** for Yjs document and transport management
- **YjsProvider** component wraps application and provides:
  - Yjs document instance
  - Transport manager
  - Connection status
  - Peer count
  - Awareness (user presence)
- **Decision Rationale**: 
  - Avoids prop drilling
  - Single source of truth for Yjs state
  - Easy to access from any component

### Component Architecture
- **Functional components** with hooks throughout
- **Separation of concerns**:
  - `TldrawCanvas` - Canvas rendering and sync
  - `RoomManager` - Room creation/joining UI
  - `ConnectionStatus` - Connection state display
  - `UsersList` - User presence display
- **Decision Rationale**: Modern React patterns, easier testing, better performance

## State Management & Synchronization

### CRDT-Based State Management
- **Yjs Y.Doc** as the single source of truth
- **Y.Map** for storing Tldraw records
- **Decision Rationale**:
  - Automatic conflict resolution
  - No need for complex merge strategies
  - Works seamlessly with offline scenarios
  - Proven in production (used by Notion, Linear, etc.)

### Bidirectional Sync Pattern
- **Tldraw → Yjs**: All user changes synced to Yjs document
- **Yjs → Tldraw**: Remote changes merged into Tldraw store
- **Sync guard** (`isSyncingRef`) prevents circular updates
- **Decision Rationale**:
  - Ensures consistency between Tldraw and Yjs
  - Prevents infinite sync loops
  - Handles both local and remote changes

### Data Structure
- **Y.Map<TLRecord>** stores all Tldraw records
- **Key**: Record ID, **Value**: Full record object
- **Decision Rationale**: 
  - Efficient lookups by ID
  - Yjs handles updates atomically
  - Easy to sync additions, updates, deletions

## Transport Layer Decisions

### Multi-Transport Architecture
- **YjsTransportManager** class abstracts transport layer
- Supports multiple transports simultaneously:
  - WebRTC (primary for P2P)
  - WebSocket (fallback/reliable)
  - IndexedDB (offline persistence)
- **Decision Rationale**:
  - Flexibility to swap transports
  - Can use multiple transports for redundancy
  - Easy to add new transports

### WebRTC for P2P Communication
- **y-webrtc** library for peer-to-peer connections
- **Local signaling server** (`signaling-server.js`) on port 1234
- **Decision Rationale**:
  - True P2P reduces server load
  - Lower latency for direct connections
  - Public signaling servers unreliable (discovered during development)
  - Local server provides control and reliability

### WebSocket as Alternative Transport
- **y-websocket** available but not primary
- **Decision Rationale**: 
  - More reliable than WebRTC in some network conditions
  - Easier to debug
  - Can be used as fallback

### Offline Persistence
- **y-indexeddb** for local storage
- **Per-room storage**: Database name includes room ID (`tldraw-${roomId}`)
- **Decision Rationale**:
  - Works offline
  - Data persists across sessions
  - Automatic sync when reconnected
  - Browser-native storage (no external dependencies)

### Signaling Server Design
- **Custom WebSocket server** (`signaling-server.js`)
- **Room-based pub/sub** pattern
- **Features**:
  - Subscribe/unsubscribe to rooms
  - Broadcast messages to room peers
  - Ping/pong for connection health
- **Decision Rationale**:
  - Simple, focused on signaling only
  - No state storage (stateless)
  - Easy to deploy and scale
  - Can be replaced with managed service

## User Presence System

### Awareness Protocol
- **y-protocols/awareness** for user presence
- **Decision Rationale**:
  - Lightweight protocol
  - Built into Yjs ecosystem
  - Automatic cleanup on disconnect
  - Works with any transport

### User Identity
- **Random user names**: Adjective + Animal (e.g., "Happy Panda")
- **Random colors**: 10 predefined colors
- **Random user IDs**: Generated on join
- **Decision Rationale**:
  - No authentication required (simplicity)
  - Fun, memorable names
  - Unique visual identification
  - Can be extended to custom names later

### Presence Data Structure
```typescript
interface UserPresence {
  id: string          // Unique user ID
  name: string        // Display name
  color: string       // Hex color code
  cursor?: { x, y }   // Optional cursor position (future)
}
```

## UI/UX Design Decisions

### Layout
- **Full viewport layout** (100vw × 100vh)
- **Collapsible sidebar** on the left
- **Canvas takes remaining space**
- **Decision Rationale**: Maximum drawing space, clean interface

### Sidebar Design
- **Fixed width** (280px when open)
- **Contains**:
  - Room information
  - Connection status
  - Active users list
  - Dark mode toggle
  - Dot grid toggle
  - Leave room button
- **Decision Rationale**: 
  - All controls in one place
  - Doesn't obstruct canvas
  - Easy to hide for full-screen drawing

### Dark Mode
- **User-toggleable** dark/light mode
- **Applies to**:
  - Sidebar background
  - Canvas background
  - Tldraw UI
- **Decision Rationale**: 
  - User preference
  - Reduces eye strain
  - Modern app expectation

### Visual Feedback
- **Connection status indicator**: Shows connection state and peer count
- **Users list**: Shows all active users with colors
- **"(you)" indicator**: Highlights local user
- **Decision Rationale**: 
  - Users need to know connection state
  - See who's collaborating
  - Clear visual feedback

### Room Management
- **Room ID-based** collaboration
- **Quick join** button for testing
- **Custom room creation** with random ID generation
- **Decision Rationale**:
  - Simple sharing model (just share room ID)
  - No authentication complexity
  - Easy to test with quick join

## Build & Development Tooling

### TypeScript Configuration
- **Strict mode enabled**
- **ES2020 target**
- **ESM modules**
- **React JSX transform**
- **Decision Rationale**:
  - Catch errors at compile time
  - Modern JavaScript features
  - Better tree-shaking

### Electron Configuration
- **Context isolation**: Enabled
- **Node integration**: Disabled in renderer
- **Preload script**: Bridges main and renderer
- **Decision Rationale**:
  - Security best practices
  - Prevents renderer from accessing Node.js directly
  - Safer architecture

### Development Server
- **Port 5183** for Vite dev server
- **Hot Module Replacement** enabled
- **DevTools** auto-opens in Electron
- **Decision Rationale**:
  - Fast iteration
  - Easy debugging
  - Modern development experience

### Build Output
- **Separate builds** for main, preload, and renderer
- **Output to `out/` directory**
- **Decision Rationale**: 
  - Clear separation
  - Easy to package
  - Follows electron-vite conventions

## Security & Privacy Decisions

### No Authentication
- **Anyone with room ID can join**
- **Decision Rationale**:
  - Simplicity for MVP
  - Easy to test
  - Can be added later if needed

### No Encryption
- **Messages sent in plain text**
- **Decision Rationale**:
  - Simplicity for MVP
  - Local network usage primarily
  - Can add E2E encryption later

### Context Isolation
- **Electron security best practices** followed
- **Preload script** for safe IPC
- **Decision Rationale**: 
  - Prevents XSS attacks
  - Limits attack surface
  - Industry standard

## Performance Decisions

### Sync Optimization
- **Sync guard** prevents circular updates
- **Batch updates** in Yjs transactions
- **Debounced cursor updates** (prepared but not active)
- **Decision Rationale**:
  - Prevents infinite loops
  - Reduces network traffic
  - Better performance

### IndexedDB Usage
- **Per-room databases** for isolation
- **Automatic sync** on reconnect
- **Decision Rationale**:
  - Fast local access
  - No network required for reads
  - Efficient storage

### Component Optimization
- **React hooks** for efficient re-renders
- **Context API** avoids unnecessary prop updates
- **Conditional rendering** for sidebar
- **Decision Rationale**:
  - Better performance
  - Smaller bundle size
  - Modern React patterns

## Python Component (Image Bucketing)

### Technology Choices
- **Python 3.10+** requirement
- **sentence-transformers** for vision-language embeddings
- **PyTorch** for ML models
- **Pillow** for image processing
- **NumPy** for numerical operations
- **Decision Rationale**:
  - Python ecosystem for ML
  - Standard libraries for image processing

### Model Selection: SigLIP over CLIP
- **SigLIP (Sigmoid Loss for Language-Image Pre-training)** chosen over CLIP
- **Decision Rationale**:
  - **Better loss function**: SigLIP uses Sigmoid loss instead of contrastive loss, which operates on image-text pairs without requiring global similarity normalization
  - **More efficient training**: Works better with varying batch sizes, improving scalability and performance
  - **Enhanced zero-shot capabilities**: Better generalization to new tasks without fine-tuning, crucial for diverse image categorization
  - **Superior pretraining**: Trained on extensive WebLI dataset, learning richer visual-language representations
  - **Multilingual support**: Certain SigLIP variants support multilingual text input, beneficial for global use cases
  - **Better accuracy**: Improved performance on image-text retrieval and zero-shot classification tasks compared to CLIP
  - **Available via sentence-transformers**: Easy integration with existing sentence-transformers library

### Build System
- **pyproject.toml** with hatchling
- **uv** for dependency management
- **Decision Rationale**:
  - Modern Python packaging
  - Fast dependency resolution
  - Reproducible builds

### Code Quality
- **Black** for code formatting (100 char line length)
- **Ruff** for linting (100 char line length)
- **Decision Rationale**:
  - Consistent code style
  - Fast linting
  - Modern Python tooling

## Known Limitations & Trade-offs

### Current Limitations
1. **Requires local signaling server** - Must run `npm run signaling`
2. **No authentication** - Anyone with room ID can join
3. **No encryption** - Messages in plain text
4. **No server-side persistence** - Server only relays
5. **No cursor positions** - User list only, no live cursors
6. **No version history** - No undo/redo across sessions

### Trade-offs Made
- **Simplicity over security** - No auth/encryption for MVP
- **Local server over cloud** - Control and reliability
- **WebRTC over WebSocket** - P2P benefits, but more complex
- **Yjs over custom sync** - Proven solution, but learning curve
- **Electron over native** - Cross-platform, but larger bundle

## Future Considerations

### Potential Enhancements
- [ ] Real-time cursor positions on canvas
- [ ] End-to-end encryption
- [ ] Authentication/authorization
- [ ] Server-side persistence
- [ ] Version history
- [ ] Export/import functionality
- [ ] Custom user names/avatars
- [ ] Video/voice chat
- [ ] Room permissions
- [ ] Managed signaling service integration

### Architecture Evolution
- **Current**: Local signaling server
- **Future**: Could use managed services (Liveblocks, PartyKit, Ably)
- **Current**: No persistence server
- **Future**: Could add database for room history
- **Current**: Simple room IDs
- **Future**: Could add room metadata, permissions, etc.

## Development Workflow Decisions

### Scripts
- `npm run dev` - Development mode with HMR
- `npm run dev:browser` - Browser-only development
- `npm run build` - Production build
- `npm run signaling` - Start signaling server
- **Decision Rationale**: Clear, simple commands for common tasks

### File Organization
- **Feature-based** component organization
- **Shared utilities** in `src/yjs/`
- **Separate configs** for Electron and Vite
- **Decision Rationale**: Easy to find code, clear structure

## Testing & Quality

### Current State
- **No automated tests** currently
- **Manual testing** with multiple instances
- **Decision Rationale**: MVP focus, can add tests later

### Future Testing Strategy
- Unit tests for Yjs sync logic
- Integration tests for transport layer
- E2E tests for collaboration scenarios
- Performance tests for large documents

## Documentation Decisions

### Documentation Files
- `README.md` - Quick start and usage
- `IMPLEMENTATION.md` - Feature list and architecture
- `PROJECT_STRUCTURE.md` - File structure overview
- `WEBRTC-FIX.md` - Troubleshooting guide
- `USER-PRESENCE-FEATURE.md` - Feature documentation
- `TECH_DECISIONS.md` - This document
- **Decision Rationale**: 
  - Separate concerns (usage vs. implementation vs. decisions)
  - Easy to find specific information
  - Historical context preserved

## Environment & Configuration

### Environment Variables
- `VITE_WS_URL` - WebSocket server URL (default: `ws://localhost:1234`)
- `VITE_SIGNALING_URL` - Signaling server URL (default: `ws://localhost:1234`)
- `ELECTRON_RENDERER_URL` - Electron dev server URL
- `PORT` - Signaling server port (default: 1234)
- **Decision Rationale**: 
  - Flexible configuration
  - Easy to switch between dev/prod
  - Environment-specific settings

### Port Choices
- **5183** - Vite dev server (renderer)
- **1234** - Signaling server
- **Decision Rationale**: 
  - Non-standard ports avoid conflicts
  - Easy to remember
  - Can be changed via env vars

## Integration Decisions

### Tldraw Integration
- **@tldraw/sync** package for enhanced sync
- **Direct Yjs binding** (not using Tldraw's built-in sync)
- **Decision Rationale**:
  - More control over sync behavior
  - Can customize sync logic
  - Direct Yjs integration

### Yjs Integration
- **Manual sync** between Tldraw and Yjs
- **Not using** Tldraw's built-in Yjs provider
- **Decision Rationale**:
  - Full control over sync
  - Can optimize for specific use cases
  - Understand sync flow completely

## Deployment Considerations

### Current State
- **Local development** focus
- **No production deployment** yet
- **Decision Rationale**: MVP stage, focus on functionality

### Future Deployment
- **Electron app** - Can be packaged with electron-builder
- **Signaling server** - Needs cloud deployment (Heroku, AWS, etc.)
- **HTTPS/WSS** - Required for production
- **CDN** - For static assets if web version
- **Decision Rationale**: 
  - Standard deployment patterns
  - Security requirements
  - Scalability needs

