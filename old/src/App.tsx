import React, { useState } from 'react'
import { YjsProvider } from './yjs/YjsProvider'
import { TldrawCanvas } from './components/TldrawCanvas'
import { ConnectionStatus } from './components/ConnectionStatus'
import { UsersList } from './components/UsersList'
import { RoomManager } from './components/RoomManager'

const App: React.FC = () => {
  const [currentRoomId, setCurrentRoomId] = useState<string | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [darkMode, setDarkMode] = useState(false)
  const [showDotGrid, setShowDotGrid] = useState(true)

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
            background: darkMode ? '#1e1e1e' : '#f8f9fa',
            borderRight: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
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
                <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600, color: darkMode ? '#fff' : '#000' }}>Room Info</h2>
                <button
                  onClick={() => setSidebarOpen(false)}
                  style={{
                    background: 'none',
                    border: 'none',
                    fontSize: 20,
                    cursor: 'pointer',
                    padding: 4,
                    color: darkMode ? '#ccc' : '#666'
                  }}
                >
                  ×
                </button>
              </div>
              
              <div style={{
                padding: '12px 16px',
                background: darkMode ? '#2a2a2a' : 'white',
                border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
                borderRadius: 8,
                fontSize: 14
              }}>
                <div style={{ color: darkMode ? '#aaa' : '#666', marginBottom: 4 }}>Room ID</div>
                <div style={{ fontWeight: 600, wordBreak: 'break-all', color: darkMode ? '#fff' : '#000' }}>{currentRoomId}</div>
              </div>

              <ConnectionStatus darkMode={darkMode} />
              
              <UsersList darkMode={darkMode} />
              
              <div style={{
                padding: '12px 16px',
                background: darkMode ? '#2a2a2a' : 'white',
                border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span style={{ fontSize: 14, color: darkMode ? '#fff' : '#333' }}>Dark Mode</span>
                <button
                  onClick={() => setDarkMode(!darkMode)}
                  style={{
                    width: 44,
                    height: 24,
                    borderRadius: 12,
                    background: darkMode ? '#4ade80' : '#ccc',
                    border: 'none',
                    cursor: 'pointer',
                    position: 'relative',
                    transition: 'background 0.2s'
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      background: 'white',
                      position: 'absolute',
                      top: 2,
                      left: darkMode ? 22 : 2,
                      transition: 'left 0.2s',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                  />
                </button>
              </div>

              <div style={{
                padding: '12px 16px',
                background: darkMode ? '#2a2a2a' : 'white',
                border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
                borderRadius: 8,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between'
              }}>
                <span style={{ fontSize: 14, color: darkMode ? '#fff' : '#333' }}>Dot Grid</span>
                <button
                  onClick={() => setShowDotGrid(!showDotGrid)}
                  style={{
                    width: 44,
                    height: 24,
                    borderRadius: 12,
                    background: showDotGrid ? '#4ade80' : '#ccc',
                    border: 'none',
                    cursor: 'pointer',
                    position: 'relative',
                    transition: 'background 0.2s'
                  }}
                >
                  <div
                    style={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      background: 'white',
                      position: 'absolute',
                      top: 2,
                      left: showDotGrid ? 22 : 2,
                      transition: 'left 0.2s',
                      boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
                    }}
                  />
                </button>
              </div>
              
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
          overflow: 'hidden',
          background: darkMode ? '#1a1a1a' : '#fafafa'
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
                background: darkMode ? '#2a2a2a' : 'white',
                color: darkMode ? '#fff' : '#000',
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
          <div style={{ position: 'absolute', inset: 0, zIndex: 1 }}>
            <TldrawCanvas 
              darkMode={darkMode} 
              showDotGrid={showDotGrid}
            />
          </div>
        </div>
      </div>
    </YjsProvider>
  )
}

export default App


