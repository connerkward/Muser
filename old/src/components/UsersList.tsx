import React from 'react'
import { useYjs } from '../yjs/YjsProvider'
import { useYjsPresence } from '../yjs/useYjsPresence'

interface UsersListProps {
  darkMode?: boolean
}

export const UsersList: React.FC<UsersListProps> = ({ darkMode = false }) => {
  const { awareness } = useYjs()
  const { users, localUser } = useYjsPresence(awareness)

  const allUsers = localUser ? [localUser, ...users] : users

  return (
    <div
      style={{
        padding: '12px 16px',
        background: darkMode ? '#2a2a2a' : 'white',
        border: `1px solid ${darkMode ? '#333' : '#e0e0e0'}`,
        borderRadius: 8,
        fontSize: 14,
        fontFamily: 'system-ui, -apple-system, sans-serif',
        minWidth: 200
      }}
    >
      <div style={{ fontWeight: 600, marginBottom: 12, color: darkMode ? '#fff' : '#333', fontSize: 15 }}>
        Active Users ({allUsers.length})
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        {allUsers.length === 0 ? (
          <div style={{ color: darkMode ? '#666' : '#999', fontSize: 13 }}>No other users</div>
        ) : (
          allUsers.map((user, index) => (
            <div
              key={user.id}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 10
              }}
            >
              <div
                style={{
                  width: 14,
                  height: 14,
                  borderRadius: '50%',
                  background: user.color,
                  flexShrink: 0,
                  border: `2px solid ${darkMode ? '#2a2a2a' : 'white'}`,
                  boxShadow: darkMode ? '0 0 0 1px rgba(255,255,255,0.1)' : '0 0 0 1px rgba(0,0,0,0.1)'
                }}
              />
              <span style={{ fontSize: 13, color: darkMode ? '#fff' : '#333' }}>
                {user.name}
                {localUser && user.id === localUser.id && (
                  <span style={{ color: darkMode ? '#888' : '#999', marginLeft: 6 }}>(you)</span>
                )}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

