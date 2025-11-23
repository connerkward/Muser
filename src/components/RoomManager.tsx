import React, { useState } from 'react'

interface RoomManagerProps {
  onJoinRoom: (roomId: string) => void
}

export const RoomManager: React.FC<RoomManagerProps> = ({ onJoinRoom }) => {
  const [roomId, setRoomId] = useState('test-room-123')

  const handleGenerateRoom = () => {
    const newRoomId = Math.random().toString(36).substring(2, 15)
    onJoinRoom(newRoomId)
  }

  const handleJoinRoom = () => {
    if (roomId.trim()) {
      onJoinRoom(roomId.trim())
    }
  }
  
  const handleQuickJoin = () => {
    onJoinRoom('test-room-123')
  }

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'center',
        height: '100vh',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        fontFamily: 'system-ui, -apple-system, sans-serif'
      }}
    >
      <div
        style={{
          background: 'white',
          padding: 40,
          borderRadius: 16,
          boxShadow: '0 10px 40px rgba(0,0,0,0.2)',
          maxWidth: 400,
          width: '90%'
        }}
      >
        <h1 style={{ margin: '0 0 24px 0', fontSize: 28, textAlign: 'center' }}>
          P2P Tldraw
        </h1>
        
        <button
          onClick={handleQuickJoin}
          style={{
            width: '100%',
            padding: 12,
            fontSize: 16,
            fontWeight: 600,
            background: '#10b981',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            cursor: 'pointer',
            marginBottom: 12
          }}
        >
          Quick Join: test-room-123
        </button>
        
        <button
          onClick={handleGenerateRoom}
          style={{
            width: '100%',
            padding: 12,
            fontSize: 16,
            fontWeight: 600,
            background: '#667eea',
            color: 'white',
            border: 'none',
            borderRadius: 8,
            cursor: 'pointer',
            marginBottom: 24
          }}
        >
          Create New Room
        </button>

        <div style={{ textAlign: 'center', margin: '24px 0', color: '#666' }}>
          or
        </div>

        <div>
          <input
            type="text"
            placeholder="Enter room ID"
            value={roomId}
            onChange={(e) => setRoomId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleJoinRoom()}
            style={{
              width: '100%',
              padding: 12,
              fontSize: 16,
              border: '1px solid #ddd',
              borderRadius: 8,
              marginBottom: 12,
              boxSizing: 'border-box'
            }}
          />
          <button
            onClick={handleJoinRoom}
            disabled={!roomId.trim()}
            style={{
              width: '100%',
              padding: 12,
              fontSize: 16,
              fontWeight: 600,
              background: roomId.trim() ? '#764ba2' : '#ccc',
              color: 'white',
              border: 'none',
              borderRadius: 8,
              cursor: roomId.trim() ? 'pointer' : 'not-allowed'
            }}
          >
            Join Room
          </button>
        </div>
      </div>
    </div>
  )
}

