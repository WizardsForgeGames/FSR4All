// FSR4 Dashboard — Renderer (app orchestration)
(async function () {
  const viewer = new ImageViewer();
  const inspector = new PixelInspector();

  // State
  let pairs = [];
  let currentPair = null;
  let compareMode = 'input-vs-upscaled'; // or 'upscaled-vs-ref'

  // --- Elements ---
  const pairSelect = document.getElementById('pair-select');
  const btnSideBySide = document.getElementById('btn-side-by-side');
  const btnSwipe = document.getElementById('btn-swipe');
  const btnInputVsUpscaled = document.getElementById('btn-input-vs-upscaled');
  const btnUpscaledVsRef = document.getElementById('btn-upscaled-vs-ref');
  const btnFit = document.getElementById('btn-fit');
  const btn1x = document.getElementById('btn-1x');
  const zoomLabel = document.getElementById('zoom-label');
  const labelLeft = document.getElementById('label-left');
  const labelRight = document.getElementById('label-right');
  const swipeLabelLeft = document.getElementById('swipe-label-left');
  const swipeLabelRight = document.getElementById('swipe-label-right');

  // --- Wire up viewer callbacks ---
  viewer.onZoomChange = (z) => {
    zoomLabel.textContent = Math.round(z * 100) + '%';
  };

  viewer.onMouseMove = (imgX, imgY, pixel, image) => {
    inspector.update(imgX, imgY, pixel, image);
  };

  // --- Load pair into viewer ---
  async function loadCurrentPair() {
    if (!currentPair) return;

    let leftUrl, rightUrl, leftLabel, rightLabel;

    if (compareMode === 'input-vs-upscaled') {
      leftUrl = window.dashboardAPI.pathToUrl(currentPair.inputPath);
      rightUrl = window.dashboardAPI.pathToUrl(currentPair.upscaledPath);
      leftLabel = 'Input';
      rightLabel = 'Upscaled';
    } else {
      leftUrl = window.dashboardAPI.pathToUrl(currentPair.upscaledPath);
      rightUrl = window.dashboardAPI.pathToUrl(currentPair.referencePath);
      leftLabel = 'Upscaled';
      rightLabel = 'Reference';
    }

    labelLeft.textContent = leftLabel;
    labelRight.textContent = rightLabel;
    swipeLabelLeft.textContent = leftLabel;
    swipeLabelRight.textContent = rightLabel;

    await viewer.loadPair(leftUrl, rightUrl);
  }

  // --- Pair selector ---
  pairSelect.addEventListener('change', () => {
    const idx = pairSelect.selectedIndex;
    if (idx >= 0 && idx < pairs.length) {
      currentPair = pairs[idx];
      loadCurrentPair();
    }
  });

  // --- View mode buttons ---
  btnSideBySide.addEventListener('click', () => {
    viewer.setMode('side-by-side');
    btnSideBySide.classList.add('active');
    btnSwipe.classList.remove('active');
  });

  btnSwipe.addEventListener('click', () => {
    viewer.setMode('swipe');
    btnSwipe.classList.add('active');
    btnSideBySide.classList.remove('active');
  });

  // --- Compare mode buttons ---
  btnInputVsUpscaled.addEventListener('click', () => {
    compareMode = 'input-vs-upscaled';
    btnInputVsUpscaled.classList.add('active');
    btnUpscaledVsRef.classList.remove('active');
    loadCurrentPair();
  });

  btnUpscaledVsRef.addEventListener('click', () => {
    if (!currentPair?.referencePath) return;
    compareMode = 'upscaled-vs-ref';
    btnUpscaledVsRef.classList.add('active');
    btnInputVsUpscaled.classList.remove('active');
    loadCurrentPair();
  });

  // --- Zoom buttons ---
  btnFit.addEventListener('click', () => viewer.fitToView());
  btn1x.addEventListener('click', () => viewer.setZoom(1.0));

  // --- Keyboard shortcuts ---
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
    switch (e.key) {
      case 'f': case 'F': viewer.fitToView(); break;
      case 's': case 'S':
        if (viewer.mode === 'side-by-side') btnSwipe.click();
        else btnSideBySide.click();
        break;
      case '1': viewer.setZoom(1.0); break;
      case '2': viewer.setZoom(2.0); break;
      case '4': viewer.setZoom(4.0); break;
    }
  });

  // --- Scan and populate ---
  pairs = await window.dashboardAPI.scanPairs();

  if (pairs.length === 0) {
    const opt = document.createElement('option');
    opt.textContent = 'No pairs found — run test_inference first';
    pairSelect.appendChild(opt);
    return;
  }

  for (const p of pairs) {
    const opt = document.createElement('option');
    const inputRes = `${p.inputSize.w}×${p.inputSize.h}`;
    const upscaledRes = `${p.upscaledSize.w}×${p.upscaledSize.h}`;
    opt.textContent = `${p.name}  (${inputRes} → ${upscaledRes})`;
    pairSelect.appendChild(opt);
  }

  // Enable/disable reference button based on first pair
  currentPair = pairs[0];
  btnUpscaledVsRef.disabled = !currentPair.referencePath;

  // Update reference button when pair changes
  pairSelect.addEventListener('change', () => {
    btnUpscaledVsRef.disabled = !currentPair?.referencePath;
    if (!currentPair?.referencePath && compareMode === 'upscaled-vs-ref') {
      btnInputVsUpscaled.click();
    }
  });

  // Load config (restore last selected pair)
  const config = await window.dashboardAPI.getConfig();
  if (config.lastPair) {
    const idx = pairs.findIndex(p => p.name === config.lastPair);
    if (idx >= 0) {
      pairSelect.selectedIndex = idx;
      currentPair = pairs[idx];
      btnUpscaledVsRef.disabled = !currentPair.referencePath;
    }
  }

  // Save selected pair on change
  pairSelect.addEventListener('change', async () => {
    if (currentPair) {
      await window.dashboardAPI.saveConfig({
        ...(await window.dashboardAPI.getConfig()),
        lastPair: currentPair.name
      });
    }
  });

  // Default to swipe mode (comparison slider)
  viewer.setMode('swipe');
  btnSwipe.classList.add('active');
  btnSideBySide.classList.remove('active');

  // Load the first pair
  await loadCurrentPair();
})();
