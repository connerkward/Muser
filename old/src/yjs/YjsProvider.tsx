import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import * as Y from 'yjs'
import { YjsTransportManager, TransportConfig } from './transports'

interface YjsContextValue {
  doc: Y.Doc
  transportManager: YjsTransportManager
  isConnected: boolean
  peersCount: number
  awareness: any
}

const YjsContext = createContext<YjsContextValue | null>(null)

export const useYjs = () => {
  const context = useContext(YjsContext)
  if (!context) {
    throw new Error('useYjs must be used within YjsProvider')
  }
  return context
}

interface YjsProviderProps {
  children: ReactNode
  roomId: string
  password?: string
}

export const YjsProvider: React.FC<YjsProviderProps> = ({ children, roomId, password }) => {
  const [doc] = useState(() => new Y.Doc())
  const [transportManager] = useState(() => new YjsTransportManager(doc))
  const [isConnected, setIsConnected] = useState(false)
  const [peersCount, setPeersCount] = useState(0)
  const [awareness, setAwareness] = useState<any>(null)

  useEffect(() => {
    console.log('[YjsProvider] Initializing for room:', roomId)

    // Connect to IndexedDB for offline persistence
    const dbName = `tldraw-${roomId}`
    console.log('[YjsProvider] Connecting to IndexedDB:', dbName)
    transportManager.connectIndexedDB(dbName)

    // Use WebRTC with local signaling server
    console.log('[YjsProvider] Connecting via WebRTC...')
    const signalingUrl = import.meta.env.VITE_SIGNALING_URL || 'ws://localhost:1234'
    const config: TransportConfig = {
      roomId,
      password,
      signalingServers: [signalingUrl]
    }
    const provider = transportManager.connectWebRTC(config)

    const handleStatus = (event: any) => {
      // WebRTC provider can send {connected: true} or {status: 'connected'}
      const isConnected = event?.status === 'connected' || event?.connected === true || false
      console.log('[YjsProvider] WebRTC status changed:', event, 'isConnected:', isConnected)
      setIsConnected(isConnected)
    }

    const handlePeers = (event: { webrtcPeers: any[], webrtcConns: any[], bcPeers?: any[] }) => {
      // Count both WebRTC peers and broadcast channel peers
      const webrtcCount = event.webrtcPeers?.length || 0
      const bcCount = event.bcPeers?.length || 0
      const totalPeers = webrtcCount + bcCount
      console.log('[YjsProvider] WebRTC peers - webrtc:', webrtcCount, 'bc:', bcCount, 'total:', totalPeers)
      setPeersCount(totalPeers)
      // Consider connected if we have peers OR if the signaling connection is established
      const hasConnection = totalPeers > 0 || provider.wsconnected || provider.synced
      if (hasConnection) {
        setIsConnected(true)
      }
    }

    const handleSynced = (event: any) => {
      console.log('[YjsProvider] WebRTC synced:', event)
    }

    provider.on('status', handleStatus)
    provider.on('peers', handlePeers)
    provider.on('synced', handleSynced)
    
    // Set awareness - WebRTC provider should have awareness
    // Wait a bit for provider to initialize
    const setupAwareness = () => {
      if (provider.awareness) {
        console.log('[YjsProvider] Setting awareness')
        setAwareness(provider.awareness)
      } else {
        console.warn('[YjsProvider] No awareness found, retrying...')
        setTimeout(setupAwareness, 100)
      }
    }
    setupAwareness()
    
    console.log('[YjsProvider] Provider setup complete')

    return () => {
      console.log('[YjsProvider] Cleaning up...')
      provider.off('status', handleStatus)
      provider.off('peers', handlePeers)
      provider.off('synced', handleSynced)
      transportManager.destroy()
    }
  }, [roomId, password, doc, transportManager])

  const value: YjsContextValue = {
    doc,
    transportManager,
    isConnected,
    peersCount,
    awareness
  }

  return <YjsContext.Provider value={value}>{children}</YjsContext.Provider>
}

