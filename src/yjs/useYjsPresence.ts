import { useEffect, useState } from 'react'
import { Awareness } from 'y-protocols/awareness'

export interface UserPresence {
  id: string
  name: string
  color: string
  cursor?: { x: number; y: number }
}

const COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
  '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B739', '#52B788'
]

function generateUserId() {
  return `user-${Math.random().toString(36).substr(2, 9)}`
}

function generateUserName() {
  const adjectives = ['Happy', 'Clever', 'Swift', 'Bright', 'Cool', 'Wise', 'Bold', 'Calm']
  const nouns = ['Panda', 'Fox', 'Eagle', 'Wolf', 'Bear', 'Owl', 'Tiger', 'Lion']
  const adj = adjectives[Math.floor(Math.random() * adjectives.length)]
  const noun = nouns[Math.floor(Math.random() * nouns.length)]
  return `${adj} ${noun}`
}

export function useYjsPresence(awareness: Awareness | null) {
  const [users, setUsers] = useState<Map<number, UserPresence>>(new Map())
  const [localUser, setLocalUser] = useState<UserPresence | null>(null)

  useEffect(() => {
    if (!awareness) return

    // Set local user
    const userId = generateUserId()
    const userName = generateUserName()
    const userColor = COLORS[Math.floor(Math.random() * COLORS.length)]
    
    const user: UserPresence = {
      id: userId,
      name: userName,
      color: userColor
    }
    
    setLocalUser(user)
    
    awareness.setLocalState({
      user
    })

    const handleChange = () => {
      const states = awareness.getStates()
      const newUsers = new Map<number, UserPresence>()
      
      states.forEach((state, clientId) => {
        if (state.user && clientId !== awareness.clientID) {
          newUsers.set(clientId, state.user)
        }
      })
      
      setUsers(newUsers)
    }

    awareness.on('change', handleChange)
    handleChange() // Initial call

    return () => {
      awareness.off('change', handleChange)
    }
  }, [awareness])

  const updateCursor = (x: number, y: number) => {
    if (awareness && localUser) {
      awareness.setLocalState({
        user: {
          ...localUser,
          cursor: { x, y }
        }
      })
    }
  }

  return {
    users: Array.from(users.values()),
    localUser,
    updateCursor
  }
}

