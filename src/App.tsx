import React, { useState } from 'react'
import { YjsProvider } from './yjs/YjsProvider'
import { TldrawCanvas } from './components/TldrawCanvas'
import { ConnectionStatus } from './components/ConnectionStatus'
import { UsersList } from './components/UsersList'
import { RoomManager } from './components/RoomManager'

const App: React.FC = () => {
  const [currentRoomId, setCurrentRoomId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)

  if (!currentRoomId) {
    return <RoomManager onJoinRoom={setCurrentRoomId} />
  }

  return (
    <YjsProvider roomId={currentRoomId}>
      <div style={{ 
        width: '100vw', 
        height: '100vh', 
        display: 'flex',
        flexDirection: 'row',
        overflow: 'hidden'
      }}>
        {/* Sidebar */}
        <div
          style={{
            width: sidebarOpen ? 280 : 0,
            height: '100vh',
            background: '#f8f9fa',
            borderRight: '1px solid #e0e0e0',
            display: 'flex',
            flexDirection: 'column',
            transition: 'width 0.2s ease',
            overflow: 'hidden',
            flexShrink: 0
          }}
        >
          {sidebarOpen && (
            <div style={{ padding: 16, display: 'flex', flexDirection: 'column', gap: 16, height: '100%', overflowY: 'auto' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>Room Info</h2>
                <button
                  onClick={() => setSidebarOpen(false)}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontSize: 20,
                    cursor: 'pointer',
                    padding: 4,
                    color: '#666'
                  }}
                >
                  ×
                </button>
              </div>
              
              <div style={{
                padding: '12px 16px',
                background: 'white',
                border: '1px solid #e0e0e0',
                borderRadius: 8,
                fontSize: 14
              }}>
                <div style={{ color: '#666', marginBottom: 4 }}>Room ID</div>
                <div style={{ fontWeight: 600, wordBreak: 'break-all' }}>{currentRoomId}</div>
              </div>

              <ConnectionStatus />
              
              <UsersList />
              
              <div style={{ marginTop: 'auto', paddingTop: 16 }}>
                <button
                  onClick={() => setCurrentRoomId(null)}
                  style={{
                    width: '100%',
                    padding: 12,
                    fontSize: 14,
                    fontWeight: 600,
                    background: '#ef4444',
                    color: 'white',
                    border: 'none',
                    borderRadius: 8,
                    cursor: 'pointer'
                  }}
                >
                  Leave Room
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Main canvas area */}
        <div style={{ 
          flex: 1, 
          height: '100vh', 
          position: 'relative',
          overflow: 'hidden'
        }}>
          {!sidebarOpen && (
            <button
              onClick={() => setSidebarOpen(true)}
              style={{
                position: 'absolute',
                top: 10,
                left: 10,
                zIndex: 1000,
                padding: '8px 12px',
                background: 'white',
                border: '1px solid #ccc',
                borderRadius: 8,
                cursor: 'pointer',
                fontSize: 14,
                boxShadow: '0 2px 8px rgba(0,0,0,0.1)'
              }}
            >
              ☰ Menu
            </button>
          )}
          <TldrawCanvas />
        </div>
      </div>
    </YjsProvider>
  )
}

export default App


