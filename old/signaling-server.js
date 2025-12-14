#!/usr/bin/env node

/**
 * WebSocket server for y-websocket
 * Provides reliable real-time sync for the P2P Tldraw app
 */

const WebSocket = require('ws')
const http = require('http')
const Y = require('yjs')

const port = process.env.PORT || 1234

const server = http.createServer((request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/plain' })
  response.end('WebRTC Signaling Server Running\n')
})

const wss = new WebSocket.Server({ server })

const rooms = new Map()

wss.on('connection', (ws) => {
  console.log('[Signaling] New connection')
  
  let currentRoom = null
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message)
      
      if (data.type === 'subscribe' && data.topics) {
        // Subscribe to rooms
        data.topics.forEach(topic => {
          if (!rooms.has(topic)) {
            rooms.set(topic, new Set())
          }
          rooms.get(topic).add(ws)
          currentRoom = topic
          console.log(`[Signaling] Client subscribed to room: ${topic}`)
        })
      } else if (data.type === 'unsubscribe' && data.topics) {
        // Unsubscribe from rooms
        data.topics.forEach(topic => {
          if (rooms.has(topic)) {
            rooms.get(topic).delete(ws)
            if (rooms.get(topic).size === 0) {
              rooms.delete(topic)
            }
          }
        })
      } else if (data.type === 'publish' && data.topic) {
        // Broadcast to all peers in the room except sender
        const room = rooms.get(data.topic)
        if (room) {
          console.log(`[Signaling] Broadcasting to ${room.size - 1} peers in room: ${data.topic}`)
          room.forEach(client => {
            if (client !== ws && client.readyState === WebSocket.OPEN) {
              client.send(message)
            }
          })
        }
      } else if (data.type === 'ping') {
        // Respond to ping
        ws.send(JSON.stringify({ type: 'pong' }))
      }
    } catch (error) {
      console.error('[Signaling] Error processing message:', error)
    }
  })
  
  ws.on('close', () => {
    console.log('[Signaling] Connection closed')
    // Remove from all rooms
    rooms.forEach((clients, topic) => {
      clients.delete(ws)
      if (clients.size === 0) {
        rooms.delete(topic)
      }
    })
  })
  
  ws.on('error', (error) => {
    console.error('[Signaling] WebSocket error:', error)
  })
})

server.listen(port, () => {
  console.log(`[Signaling] WebRTC signaling server running on ws://localhost:${port}`)
  console.log(`[Signaling] HTTP health check available at http://localhost:${port}`)
})

