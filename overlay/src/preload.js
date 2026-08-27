const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
    // Add any needed IPC here
});
