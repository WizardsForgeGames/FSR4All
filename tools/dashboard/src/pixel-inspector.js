// FSR4 Dashboard — Pixel Inspector (RGB readout at cursor)
class PixelInspector {
  constructor() {
    this.el = document.getElementById('pixel-info');
  }

  update(imgX, imgY, pixel, image) {
    if (!pixel || imgX < 0 || imgY < 0 || imgX >= image.width || imgY >= image.height) {
      this.el.textContent = 'R:--- G:--- B:--- (---, ---)';
      return;
    }
    const r = pixel[0].toString().padStart(3, ' ');
    const g = pixel[1].toString().padStart(3, ' ');
    const b = pixel[2].toString().padStart(3, ' ');
    this.el.textContent = `R:${r} G:${g} B:${b} (${imgX}, ${imgY})`;
  }

  clear() {
    this.el.textContent = 'R:--- G:--- B:--- (---, ---)';
  }
}
