import { contextBridge } from 'electron'

contextBridge.exposeInMainWorld('electron', {
  // Add any APIs you want to expose to the renderer process
})

