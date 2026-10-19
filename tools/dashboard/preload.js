const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('dashboardAPI', {
  scanPairs: () => ipcRenderer.invoke('scan-pairs'),
  getConfig: () => ipcRenderer.invoke('get-config'),
  saveConfig: (config) => ipcRenderer.invoke('save-config', config),
  pathToUrl: (filePath) => `fsr://${filePath.replace(/\\/g, '/')}`
});
