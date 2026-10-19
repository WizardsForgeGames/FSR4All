// FSR4 Dashboard — Image Viewer (side-by-side + swipe with synced zoom/pan)
class ImageViewer {
  constructor() {
    // Canvases
    this.canvasL = document.getElementById('canvas-left');
    this.canvasR = document.getElementById('canvas-right');
    this.canvasSL = document.getElementById('canvas-swipe-left');
    this.canvasSR = document.getElementById('canvas-swipe-right');
    this.ctxL = this.canvasL.getContext('2d');
    this.ctxR = this.canvasR.getContext('2d');
    this.ctxSL = this.canvasSL.getContext('2d');
    this.ctxSR = this.canvasSR.getContext('2d');

    // Images
    this.leftImage = null;
    this.rightImage = null;

    // View state (in right-image pixel space)
    this.zoom = 1.0;
    this.panX = 0;
    this.panY = 0;

    // Mode
    this.mode = 'side-by-side'; // or 'swipe'
    this.swipeX = 0.5; // 0-1

    // Drag state
    this._dragging = false;
    this._dragStartX = 0;
    this._dragStartY = 0;
    this._dragStartPanX = 0;
    this._dragStartPanY = 0;
    this._swipeDragging = false;

    // Callbacks
    this.onZoomChange = null;
    this.onMouseMove = null;

    this._setupEvents();
    this._resizeObserver = new ResizeObserver(() => this._resize());
    this._resizeObserver.observe(document.getElementById('viewport'));
  }

  async loadPair(leftUrl, rightUrl) {
    const [l, r] = await Promise.all([this._loadImage(leftUrl), this._loadImage(rightUrl)]);
    this.leftImage = l;
    this.rightImage = r;
    this.fitToView();
  }

  setMode(mode) {
    this.mode = mode;
    document.getElementById('side-by-side').classList.toggle('active', mode === 'side-by-side');
    document.getElementById('swipe').classList.toggle('active', mode === 'swipe');
    document.getElementById('viewport').style.cursor = '';
    this._resize();
    this.render();
  }

  fitToView() {
    if (!this.rightImage) return;
    const vp = document.getElementById('viewport').getBoundingClientRect();
    const canvasW = this.mode === 'side-by-side' ? (vp.width - 3) / 2 : vp.width;
    const canvasH = vp.height;
    const scaleX = canvasW / this.rightImage.width;
    const scaleY = canvasH / this.rightImage.height;
    this.zoom = Math.min(scaleX, scaleY);
    this.panX = -(canvasW / this.zoom - this.rightImage.width) / 2;
    this.panY = -(canvasH / this.zoom - this.rightImage.height) / 2;
    if (this.onZoomChange) this.onZoomChange(this.zoom);
    this.render();
  }

  setZoom(z) {
    // Zoom centered on viewport
    const vp = document.getElementById('viewport').getBoundingClientRect();
    const canvasW = this.mode === 'side-by-side' ? (vp.width - 3) / 2 : vp.width;
    const canvasH = vp.height;
    const cx = canvasW / 2, cy = canvasH / 2;
    const imgX = cx / this.zoom + this.panX;
    const imgY = cy / this.zoom + this.panY;
    this.zoom = Math.max(0.05, Math.min(8.0, z));
    this.panX = imgX - cx / this.zoom;
    this.panY = imgY - cy / this.zoom;
    if (this.onZoomChange) this.onZoomChange(this.zoom);
    this.render();
  }

  render() {
    if (this.mode === 'side-by-side') {
      this._renderCanvas(this.ctxL, this.canvasL, this.leftImage);
      this._renderCanvas(this.ctxR, this.canvasR, this.rightImage);
    } else {
      this._renderCanvas(this.ctxSL, this.canvasSL, this.leftImage);
      this._renderCanvas(this.ctxSR, this.canvasSR, this.rightImage);
      this._updateSwipeClip();
    }
  }

  // Returns image-space coords for a mouse event on a given canvas
  imageCoords(e, canvas, image) {
    if (!image) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const scale = image === this.leftImage && this.rightImage
      ? this.rightImage.width / image.width : 1;
    const imgX = Math.floor((cx / this.zoom + this.panX) / scale);
    const imgY = Math.floor((cy / this.zoom + this.panY) / scale);
    return { imgX, imgY, canvasX: Math.floor(cx), canvasY: Math.floor(cy) };
  }

  // --- Private ---

  _loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = (e) => reject(new Error(`Failed to load: ${url}`));
      img.src = url;
    });
  }

  _renderCanvas(ctx, canvas, image) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!image) return;
    ctx.imageSmoothingEnabled = this.zoom < 1;

    // Scale factor for left image (may be lower res than right)
    const scale = image === this.leftImage && this.rightImage
      ? this.rightImage.width / image.width : 1;

    const sx = this.panX / scale;
    const sy = this.panY / scale;
    const sw = canvas.width / (this.zoom * scale);
    const sh = canvas.height / (this.zoom * scale);

    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, canvas.width, canvas.height);
  }

  _updateSwipeClip() {
    const w = this.canvasSR.width;
    const px = Math.round(w * this.swipeX);
    this.canvasSR.style.clipPath = `inset(0 0 0 ${px}px)`;
    const divider = document.getElementById('swipe-divider');
    divider.style.left = `${px}px`;
  }

  _resize() {
    const vp = document.getElementById('viewport').getBoundingClientRect();

    if (this.mode === 'side-by-side') {
      const pw = Math.floor((vp.width - 3) / 2);
      const ph = Math.floor(vp.height);
      for (const c of [this.canvasL, this.canvasR]) {
        c.width = pw; c.height = ph;
        c.style.width = pw + 'px'; c.style.height = ph + 'px';
      }
    } else {
      const w = Math.floor(vp.width);
      const h = Math.floor(vp.height);
      for (const c of [this.canvasSL, this.canvasSR]) {
        c.width = w; c.height = h;
        c.style.width = w + 'px'; c.style.height = h + 'px';
      }
    }
    this.fitToView();
  }

  _setupEvents() {
    const vp = document.getElementById('viewport');

    // Wheel zoom (cursor-centered)
    vp.addEventListener('wheel', (e) => {
      e.preventDefault();
      const rect = this._activeCanvas(e).getBoundingClientRect();
      const cx = e.clientX - rect.left;
      const cy = e.clientY - rect.top;
      const imgX = cx / this.zoom + this.panX;
      const imgY = cy / this.zoom + this.panY;
      const factor = e.deltaY < 0 ? 1.1 : 1 / 1.1;
      this.zoom = Math.max(0.05, Math.min(8.0, this.zoom * factor));
      this.panX = imgX - cx / this.zoom;
      this.panY = imgY - cy / this.zoom;
      if (this.onZoomChange) this.onZoomChange(this.zoom);
      this.render();
    }, { passive: false });

    // Mouse interaction:
    //   Swipe mode:  left-drag = move divider, middle-drag or Alt+drag = pan
    //   Side-by-side: left-drag = pan
    vp.addEventListener('mousedown', (e) => {
      const isPan = e.button === 1 || (e.button === 0 && e.altKey);
      if (e.button === 0 && !e.altKey && this.mode === 'swipe') {
        // Swipe mode: left-click always moves divider
        this._swipeDragging = true;
        const vpRect = vp.getBoundingClientRect();
        this.swipeX = Math.max(0, Math.min(1, (e.clientX - vpRect.left) / vpRect.width));
        this._updateSwipeClip();
        e.preventDefault();
        return;
      }
      if (e.button === 0 || isPan) {
        this._dragging = true;
        this._dragStartX = e.clientX;
        this._dragStartY = e.clientY;
        this._dragStartPanX = this.panX;
        this._dragStartPanY = this.panY;
        e.preventDefault();
      }
    });

    window.addEventListener('mousemove', (e) => {
      if (this._swipeDragging) {
        const vpRect = vp.getBoundingClientRect();
        this.swipeX = Math.max(0, Math.min(1, (e.clientX - vpRect.left) / vpRect.width));
        this._updateSwipeClip();
        return;
      }
      if (this._dragging) {
        this.panX = this._dragStartPanX - (e.clientX - this._dragStartX) / this.zoom;
        this.panY = this._dragStartPanY - (e.clientY - this._dragStartY) / this.zoom;
        this.render();
        return;
      }
      // Swipe mode: always show ew-resize cursor (left-drag moves divider)
      if (this.mode === 'swipe') {
        vp.style.cursor = 'ew-resize';
      }
      // Pixel inspector callback
      if (this.onMouseMove) {
        const canvas = this._activeCanvas(e);
        const image = this._activeImage(e);
        if (canvas && image) {
          const coords = this.imageCoords(e, canvas, image);
          if (coords) {
            const ctx = canvas.getContext('2d');
            const pixel = ctx.getImageData(coords.canvasX, coords.canvasY, 1, 1).data;
            this.onMouseMove(coords.imgX, coords.imgY, pixel, image);
          }
        }
      }
    });

    window.addEventListener('mouseup', () => {
      this._dragging = false;
      this._swipeDragging = false;
    });
  }

  _activeCanvas(e) {
    if (this.mode === 'swipe') return this.canvasSL;
    // Determine which panel the mouse is over
    const rect = this.canvasL.getBoundingClientRect();
    return e.clientX < rect.right ? this.canvasL : this.canvasR;
  }

  _activeImage(e) {
    if (this.mode === 'swipe') {
      // Left of divider = left image, right = right image
      const vp = document.getElementById('viewport').getBoundingClientRect();
      const relX = (e.clientX - vp.left) / vp.width;
      return relX < this.swipeX ? this.leftImage : this.rightImage;
    }
    const rect = this.canvasL.getBoundingClientRect();
    return e.clientX < rect.right ? this.leftImage : this.rightImage;
  }
}
