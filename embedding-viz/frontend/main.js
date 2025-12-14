import { renderTopography } from './topography.js';
import { getMode } from './utils.js';
import './style.css';

let currentData = null;
let currentMode = 'image';

async function loadData(mode) {
  const filename = mode === 'text' ? 'embeddings_text.json' : 'embeddings_image.json';
  try {
    const response = await fetch(`/data/${filename}`);
    if (!response.ok) {
      throw new Error(`Failed to load ${filename}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error loading data:', error);
    // Fallback to embeddings.json if mode-specific file doesn't exist
    try {
      const response = await fetch('/data/embeddings.json');
      if (!response.ok) throw new Error('Failed to load embeddings.json');
      return await response.json();
    } catch (fallbackError) {
      console.error('Error loading fallback data:', fallbackError);
      return null;
    }
  }
}

function renderVisualization(data, mode) {
  const container = document.getElementById('topography');
  if (!container) {
    console.error('Topography container not found');
    return;
  }
  
  if (!data || !data.items || !data.items.length) {
    console.error('No data to render');
    return;
  }
  
  // Ensure mode is set
  data.mode = mode || data.mode || 'image';
  
  renderTopography(data, container);
}

function handleResize() {
  if (currentData) {
    renderVisualization(currentData, currentMode);
  }
}

async function init() {
  // Set up mode toggle
  const modeToggle = document.getElementById('mode-toggle');
  if (modeToggle) {
    modeToggle.addEventListener('change', async () => {
      const newMode = getMode();
      if (newMode !== currentMode) {
        currentMode = newMode;
        const data = await loadData(newMode);
        if (data) {
          currentData = data;
          renderVisualization(data, newMode);
        }
      }
    });
  }
  
  // Load initial data
  currentMode = getMode();
  const data = await loadData(currentMode);
  if (data) {
    currentData = data;
    renderVisualization(data, currentMode);
  }
  
  // Handle window resize
  let resizeTimeout;
  window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(handleResize, 250);
  });
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
