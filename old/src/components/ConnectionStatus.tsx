import React from 'react'
import { useYjs } from '../yjs/YjsProvider'

interface ConnectionStatusProps {
  darkMode?: boolean
}

export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({ darkMode = false }) => {
  const { isConnected, peersCount } = useYjs()

  return (
    <div
      style={{
        padding: '12px 16px',
        background: darkMode ? '#2a2a2a' : 'white',
        border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
        borderRadius: 8,
        fontSize: 14,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        color: darkMode ? '#fff' : '#000'
      }}
    >
      <div
        style={{
          width: 8,
          height: 8,
          borderRadius: '50%',
          background: isConnected ? '#4ade80' : '#f87171'
        }}
      />
      <span>
        {isConnected ? 'Connected' : 'Disconnected'}
      </span>
      {isConnected && peersCount > 0 && (
        <span style={{ color: darkMode ? '#aaa' : '#666', marginLeft: 4 }}>
          ({peersCount} peer{peersCount !== 1 ? 's' : ''})
        </span>
      )}
    </div>
  )
}


