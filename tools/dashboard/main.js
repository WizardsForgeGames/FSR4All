// FSR4 Visual Validation Dashboard — Electron Main Process
const { app, BrowserWindow, ipcMain, protocol, net } = require('electron');
const path = require('path');
const fs = require('fs');
const { pathToFileURL } = require('url');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');

let mainWindow = null;

// ---------------------------------------------------------------------------
// Config persistence
// ---------------------------------------------------------------------------
function getConfigPath() {
  return path.join(app.getPath('userData'), 'config.json');
}

function loadConfig() {
  try {
    const p = getConfigPath();
    if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, 'utf-8'));
  } catch (err) { console.error('Config load error:', err); }
  return {};
}

function saveConfig(config) {
  try {
    fs.writeFileSync(getConfigPath(), JSON.stringify(config, null, 2));
  } catch (err) { console.error('Config save error:', err); }
}

// ---------------------------------------------------------------------------
// BMP header reader (26 bytes: width at offset 18, height at offset 22)
// ---------------------------------------------------------------------------
function readBmpDimensions(filePath) {
  try {
    const fd = fs.openSync(filePath, 'r');
    const buf = Buffer.alloc(26);
    fs.readSync(fd, buf, 0, 26, 0);
    fs.closeSync(fd);
    return {
      w: buf.readInt32LE(18),
      h: Math.abs(buf.readInt32LE(22))
    };
  } catch (err) {
    return { w: 0, h: 0 };
  }
}

// ---------------------------------------------------------------------------
// Pair scanner — discovers input/upscaled pairs in output/ directory
// ---------------------------------------------------------------------------
function scanPairs() {
  const outputDir = path.join(PROJECT_ROOT, 'output');
  const testImagesDir = path.join(PROJECT_ROOT, 'test_images');
  const pairs = [];

  // Collect reference images from test_images/
  const references = [];
  if (fs.existsSync(testImagesDir)) {
    for (const file of fs.readdirSync(testImagesDir)) {
      if (file.toLowerCase().endsWith('.bmp')) {
        const fullPath = path.join(testImagesDir, file);
        const size = readBmpDimensions(fullPath);
        if (size.w > 0) references.push({ path: fullPath, name: file, ...size });
      }
    }
  }

  function buildPair(name, inputPath, upscaledPath) {
    const inputSize = readBmpDimensions(inputPath);
    const upscaledSize = readBmpDimensions(upscaledPath);
    // Match reference by dimension equality with upscaled
    const ref = references.find(r => r.w === upscaledSize.w && r.h === upscaledSize.h);
    return {
      name,
      inputPath,
      upscaledPath,
      inputSize,
      upscaledSize,
      referencePath: ref ? ref.path : null,
      referenceSize: ref ? { w: ref.w, h: ref.h } : null
    };
  }

  if (!fs.existsSync(outputDir)) return pairs;

  // Check output/ root for input.bmp + upscaled.bmp
  const rootInput = path.join(outputDir, 'input.bmp');
  const rootUpscaled = path.join(outputDir, 'upscaled.bmp');
  if (fs.existsSync(rootInput) && fs.existsSync(rootUpscaled)) {
    pairs.push(buildPair('test_pattern', rootInput, rootUpscaled));
  }

  // Check subdirectories
  try {
    for (const entry of fs.readdirSync(outputDir, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const subDir = path.join(outputDir, entry.name);
      const subInput = path.join(subDir, 'input.bmp');
      const subUpscaled = path.join(subDir, 'upscaled.bmp');
      if (fs.existsSync(subInput) && fs.existsSync(subUpscaled)) {
        pairs.push(buildPair(entry.name, subInput, subUpscaled));
      }

      // Temporal frames: frame_001.bmp vs frame_030.bmp etc.
      // Add pairs comparing first frame to each later snapshot
      const frame001 = path.join(subDir, 'frame_001.bmp');
      if (fs.existsSync(frame001)) {
        for (const file of fs.readdirSync(subDir)) {
          const m = file.match(/^frame_(\d{3})\.bmp$/);
          if (m && m[1] !== '001') {
            const laterFrame = path.join(subDir, file);
            pairs.push(buildPair(
              `${entry.name} (f1→f${parseInt(m[1])})`,
              frame001, laterFrame
            ));
          }
        }
      }
    }
  } catch (err) { console.error('Scan error:', err); }

  return pairs;
}

// ---------------------------------------------------------------------------
// Custom protocol — serves local files via fsr:// scheme
// ---------------------------------------------------------------------------
protocol.registerSchemesAsPrivileged([
  { scheme: 'fsr', privileges: { standard: true, secure: true, supportFetchAPI: true } }
]);

// ---------------------------------------------------------------------------
// App lifecycle
// ---------------------------------------------------------------------------
app.whenReady().then(() => {
  // Register fsr:// handler
  protocol.handle('fsr', (request) => {
    const url = new URL(request.url);
    const filePath = decodeURIComponent(url.pathname);
    // On Windows, pathname is /G:/path/... — pathToFileURL handles this
    return net.fetch(pathToFileURL(filePath).href);
  });

  const config = loadConfig();

  mainWindow = new BrowserWindow({
    width: config.windowBounds?.width || 1400,
    height: config.windowBounds?.height || 900,
    x: config.windowBounds?.x,
    y: config.windowBounds?.y,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile('src/index.html');

  mainWindow.on('close', () => {
    const bounds = mainWindow.getBounds();
    saveConfig({ ...loadConfig(), windowBounds: bounds });
  });
});

app.on('window-all-closed', () => app.quit());

// ---------------------------------------------------------------------------
// IPC handlers
// ---------------------------------------------------------------------------
ipcMain.handle('scan-pairs', () => scanPairs());
ipcMain.handle('get-config', () => loadConfig());
ipcMain.handle('save-config', (_, config) => saveConfig(config));
