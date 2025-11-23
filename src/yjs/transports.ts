import * as Y from 'yjs'
import { WebrtcProvider } from 'y-webrtc'
import { WebsocketProvider } from 'y-websocket'
import { IndexeddbPersistence } from 'y-indexeddb'

export interface TransportConfig {
  roomId: string
  password?: string
  signalingServers?: string[]
}

export class YjsTransportManager {
  private doc: Y.Doc
  private webrtcProvider: WebrtcProvider | null = null
  private websocketProvider: WebsocketProvider | null = null
  private indexeddbProvider: IndexeddbPersistence | null = null

  constructor(doc: Y.Doc) {
    this.doc = doc
  }
  
  connectWebSocket(url: string, roomId: string) {
    console.log('[TransportManager] connectWebSocket called with url:', url, 'room:', roomId)
    if (this.websocketProvider) {
      console.log('[TransportManager] Destroying existing WebSocket provider')
      this.websocketProvider.destroy()
    }

    this.websocketProvider = new WebsocketProvider(url, roomId, this.doc)
    
    console.log('[TransportManager] WebsocketProvider created')
    
    // Add debug listeners
    this.websocketProvider.on('status', (event: any) => {
      console.log('[TransportManager] WebSocket status event:', event)
    })
    
    this.websocketProvider.on('sync', (isSynced: boolean) => {
      console.log('[TransportManager] WebSocket synced:', isSynced)
    })
    
    return this.websocketProvider
  }

  connectWebRTC(config: TransportConfig) {
    console.log('[TransportManager] connectWebRTC called with room:', config.roomId)
    if (this.webrtcProvider) {
      console.log('[TransportManager] Destroying existing provider')
      this.webrtcProvider.destroy()
    }

    // Use default signaling servers from y-webrtc if not specified
    const options: any = {
      password: config.password
    }
    
    if (config.signalingServers) {
      options.signaling = config.signalingServers
      console.log('[TransportManager] Creating WebrtcProvider with custom signaling:', config.signalingServers)
    } else {
      console.log('[TransportManager] Creating WebrtcProvider with default signaling servers')
    }
    
    this.webrtcProvider = new WebrtcProvider(config.roomId, this.doc, options)

    console.log('[TransportManager] WebrtcProvider created')
    
    // Add debug listeners
    this.webrtcProvider.on('status', (event: any) => {
      console.log('[TransportManager] WebRTC status event:', event)
    })
    
    this.webrtcProvider.on('peers', (event: any) => {
      console.log('[TransportManager] WebRTC peers event:', event)
    })
    
    this.webrtcProvider.on('synced', (event: any) => {
      console.log('[TransportManager] WebRTC synced event:', event)
    })
    
    return this.webrtcProvider
  }

  async connectIndexedDB(name: string) {
    if (this.indexeddbProvider) {
      this.indexeddbProvider.destroy()
    }

    this.indexeddbProvider = new IndexeddbPersistence(name, this.doc)
    await this.indexeddbProvider.whenSynced

    return this.indexeddbProvider
  }

  getWebRTCProvider() {
    return this.webrtcProvider
  }

  getIndexedDBProvider() {
    return this.indexeddbProvider
  }

  getWebSocketProvider() {
    return this.websocketProvider
  }

  destroy() {
    if (this.webrtcProvider) {
      this.webrtcProvider.destroy()
      this.webrtcProvider = null
    }
    if (this.websocketProvider) {
      this.websocketProvider.destroy()
      this.websocketProvider = null
    }
    if (this.indexeddbProvider) {
      this.indexeddbProvider.destroy()
      this.indexeddbProvider = null
    }
  }
}

