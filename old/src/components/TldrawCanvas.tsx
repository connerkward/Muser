import React, { useEffect, useState, useRef } from 'react'
import { Tldraw, TLRecord, TLStoreWithStatus, createTLStore, useEditor, TLComponents, DefaultGrid } from 'tldraw'
import 'tldraw/tldraw.css'
import * as Y from 'yjs'
import { useYjs } from '../yjs/YjsProvider'

interface TldrawCanvasProps {
  darkMode?: boolean
  showDotGrid?: boolean
}

const DarkModeController: React.FC<{ darkMode: boolean }> = ({ darkMode }) => {
  const editor = useEditor()
  
  useEffect(() => {
    if (editor) {
      editor.user.updateUserPreferences({ 
        colorScheme: darkMode ? 'dark' : 'light' 
      })
    }
  }, [editor, darkMode])
  
  return null
}


export const TldrawCanvas: React.FC<TldrawCanvasProps> = ({ 
  darkMode = false, 
  showDotGrid = false
}) => {
  const { doc } = useYjs()
  const [store, setStore] = useState<TLStoreWithStatus>({ status: 'loading' })
  const isSyncingRef = useRef(false)

  useEffect(() => {
    const yStore = doc.getMap<TLRecord>('tldraw')
    const tlStore = createTLStore()

    // Sync all records from Yjs to Tldraw
    const syncAllFromYjs = () => {
      const records: TLRecord[] = []
      yStore.forEach((value) => {
        records.push(value)
      })
      if (records.length > 0) {
        console.log('[TldrawCanvas] Syncing', records.length, 'records from Yjs to Tldraw')
        tlStore.mergeRemoteChanges(() => {
          tlStore.put(records)
        })
      }
    }

    // Sync specific changes from Yjs to Tldraw
    const syncYjsChanges = (events: Y.YMapEvent<TLRecord>[]) => {
      if (isSyncingRef.current) {
        console.log('[TldrawCanvas] Skipping Yjs sync (already syncing)')
        return
      }
      
      const records: TLRecord[] = []
      
      events.forEach((event) => {
        event.keysChanged.forEach((key) => {
          const change = event.changes.keys.get(key)
          if (change) {
            if (change.action === 'add' || change.action === 'update') {
              const record = yStore.get(key)
              if (record) {
                records.push(record)
              }
            } else if (change.action === 'delete') {
              // Handle deletion
              const removedRecord = change.oldValue
              if (removedRecord) {
                tlStore.mergeRemoteChanges(() => {
                  tlStore.remove([removedRecord.id])
                })
              }
            }
          }
        })
      })

      if (records.length > 0) {
        console.log('[TldrawCanvas] Syncing', records.length, 'changed records from Yjs')
        isSyncingRef.current = true
        tlStore.mergeRemoteChanges(() => {
          tlStore.put(records)
        })
        setTimeout(() => {
          isSyncingRef.current = false
        }, 10)
      }
    }

    // Sync Tldraw changes to Yjs
    const syncTldrawToYjs = () => {
      const unsub = tlStore.listen(
        ({ changes }) => {
          if (isSyncingRef.current) {
            console.log('[TldrawCanvas] Skipping Tldraw sync (already syncing)')
            return
          }
          
          const added = Object.values(changes.added)
          const updated = Object.values(changes.updated).map(([_, record]) => record)
          const removed = Object.values(changes.removed)
          
          if (added.length > 0 || updated.length > 0 || removed.length > 0) {
            console.log('[TldrawCanvas] Syncing to Yjs - added:', added.length, 'updated:', updated.length, 'removed:', removed.length)
            isSyncingRef.current = true
            doc.transact(() => {
              added.forEach((record) => {
                yStore.set(record.id, record)
              })
              updated.forEach((record) => {
                yStore.set(record.id, record)
              })
              removed.forEach((record) => {
                yStore.delete(record.id)
              })
            })
            setTimeout(() => {
              isSyncingRef.current = false
            }, 10)
          }
        },
        { source: 'user', scope: 'all' }
      )
      return unsub
    }

    // Initial sync - load all existing data from Yjs
    syncAllFromYjs()
    setStore({ status: 'synced-remote', store: tlStore })

    // Listen to Yjs changes
    const yObserver = (event: Y.YMapEvent<TLRecord>) => {
      console.log('[TldrawCanvas] Yjs map changed, keys:', Array.from(event.keysChanged))
      syncYjsChanges([event])
    }
    yStore.observe(yObserver)

    // Listen to Tldraw changes
    const unsubTldraw = syncTldrawToYjs()

    return () => {
      yStore.unobserve(yObserver)
      unsubTldraw()
    }
  }, [doc])


  if (store.status === 'loading') {
    return <div style={{ padding: 20 }}>Loading...</div>
  }

  if (store.status !== 'synced-remote' || !store.store) {
    return <div style={{ padding: 20 }}>Error: Store not ready</div>
  }

  const components: TLComponents = showDotGrid ? {
    Grid: DefaultGrid
  } : {
    Grid: () => null
  }

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Tldraw 
        store={store.store}
        persistenceKey="tldraw"
        hideUi={false}
        components={components}
      >
        <DarkModeController darkMode={darkMode} />
      </Tldraw>
    </div>
  )
}


