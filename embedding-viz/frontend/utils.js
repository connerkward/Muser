let tooltip = null;
let lightbox = null;
let lightboxImg = null;
let lightboxText = null;
let lightboxTitle = null;
let lightboxCluster = null;
let lightboxClose = null;

// Image warming cache
const warmImageCache = new Set();
const warmingImages = new Set();

function initUtils() {
  tooltip = document.getElementById('tooltip');
  lightbox = document.getElementById('lightbox');
  lightboxImg = document.getElementById('lightbox-img');
  lightboxText = document.getElementById('lightbox-text');
  lightboxTitle = document.getElementById('lightbox-title');
  lightboxCluster = document.getElementById('lightbox-cluster');
  lightboxClose = document.querySelector('.lightbox-close');
  
  if (lightboxClose) {
    lightboxClose.addEventListener('click', closeLightbox);
  }
  
  if (lightbox) {
    lightbox.addEventListener('click', (e) => {
      if (e.target === lightbox) {
        closeLightbox();
      }
    });
  }
  
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && lightbox?.classList.contains('visible')) {
      closeLightbox();
    }
  });
}

export function showItemTooltip(event, item, clusterLabel) {
  if (!tooltip) return;
  
  const mode = getMode();
  const isText = mode === 'text';
  
  let content = '';
  
  if (isText) {
    const preview = item.preview || item.full_text || item.id || '';
    const truncated = preview.length > 200 ? preview.slice(0, 200) + '…' : preview;
    content = `
      <strong>${item.id || 'Item'}</strong><br>
      <div style="margin-top: 0.5rem; font-size: 0.65rem; line-height: 1.4; color: #cbd5e1;">
        ${truncated.replace(/\n/g, '<br>')}
      </div>
      <div style="margin-top: 0.5rem; font-size: 0.6rem; color: #64748b;">
        ${clusterLabel}
      </div>
    `;
  } else {
    const imgUrl = getImageUrl(item.content);
    content = `
      ${imgUrl ? `<img src="${imgUrl}" alt="">` : ''}
      <strong>${item.id || 'Image'}</strong><br>
      <div style="margin-top: 0.25rem; font-size: 0.6rem; color: #64748b;">
        ${clusterLabel}
      </div>
    `;
  }
  
  tooltip.innerHTML = content;
  tooltip.classList.add('visible');
  updateTooltipPosition(event);
}

export function updateTooltipPosition(event) {
  if (!tooltip || !tooltip.classList.contains('visible')) return;
  
  const padding = 10;
  const tooltipRect = tooltip.getBoundingClientRect();
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  
  let x = event.clientX + padding;
  let y = event.clientY + padding;
  
  // Adjust if tooltip would go off right edge
  if (x + tooltipRect.width > viewportWidth) {
    x = event.clientX - tooltipRect.width - padding;
  }
  
  // Adjust if tooltip would go off bottom edge
  if (y + tooltipRect.height > viewportHeight) {
    y = event.clientY - tooltipRect.height - padding;
  }
  
  // Ensure tooltip doesn't go off left or top edges
  x = Math.max(padding, x);
  y = Math.max(padding, y);
  
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

export function hideTooltip() {
  if (!tooltip) return;
  tooltip.classList.remove('visible');
}

export function getImageUrl(content) {
  if (!content) return '';
  if (typeof content === 'string' && content.startsWith('http')) return content;
  if (typeof content === 'string') return `/images/${encodeURIComponent(content)}`;
  return '';
}

export function showLightbox(item, clusterLabel) {
  if (!lightbox) return;
  
  const mode = getMode();
  const isText = mode === 'text';
  
  if (isText) {
    lightboxImg.style.display = 'none';
    if (lightboxText) {
      lightboxText.style.display = 'block';
      const text = item.full_text || item.preview || item.id || '';
      lightboxText.textContent = text;
    }
  } else {
    if (lightboxText) lightboxText.style.display = 'none';
    lightboxImg.style.display = 'block';
    const imgUrl = getImageUrl(item.content);
    lightboxImg.src = imgUrl;
    lightboxImg.alt = item.id || 'Image';
  }
  
  if (lightboxTitle) {
    lightboxTitle.textContent = item.id || (isText ? 'Text Item' : 'Image');
  }
  
  if (lightboxCluster) {
    lightboxCluster.textContent = clusterLabel || 'Unclustered';
  }
  
  lightbox.classList.add('visible');
  document.body.style.overflow = 'hidden';
}

export function closeLightbox() {
  if (!lightbox) return;
  lightbox.classList.remove('visible');
  document.body.style.overflow = '';
  if (lightboxImg) lightboxImg.src = '';
  if (lightboxText) lightboxText.textContent = '';
}

export function calculateNodeSize(zoom, base = 4) {
  return Math.max(2, base / Math.sqrt(zoom));
}

export function shouldShowCards(zoom, nodeCount, viewWidth, viewHeight) {
  return zoom > 5;
}

export function getMode() {
  const toggle = document.getElementById('mode-toggle');
  return toggle?.checked ? 'text' : 'image';
}

export function isImageWarmUrl(url) {
  return url && warmImageCache.has(url);
}

export function warmImageUrl(url, { priority = 0 } = {}) {
  if (!url || warmImageCache.has(url) || warmingImages.has(url)) return;
  
  warmingImages.add(url);
  
  const img = new Image();
  img.onload = () => {
    warmImageCache.add(url);
    warmingImages.delete(url);
    window.dispatchEvent(new CustomEvent('image:warm', { detail: { url } }));
  };
  img.onerror = () => {
    warmingImages.delete(url);
  };
  img.src = url;
}

// Initialize on load
if (typeof document !== 'undefined') {
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initUtils);
  } else {
    initUtils();
  }
}
