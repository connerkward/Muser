import * as d3 from 'd3';
import { showItemTooltip, hideTooltip, getImageUrl, showLightbox, updateTooltipPosition, calculateNodeSize, shouldShowCards, getMode, isImageWarmUrl, warmImageUrl } from './utils.js';

function sanitizeTextPreview(raw) {
  return String(raw || '')
    // cheap markdown-ish cleanup
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/\[[^\]]*\]\([^)]+\)/g, ' ')
    .replace(/[*_~>#|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

// Schedule at most once per animation frame, always using the latest args.
function rafThrottle(fn) {
  let scheduled = false;
  let lastArgs = null;
  return function(...args) {
    lastArgs = args;
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => {
      scheduled = false;
      fn(...lastArgs);
    });
  };
}

let lastCardsUpdateAt = 0;
let lastRegionUpdateAt = 0;
let lastRegionLabel = null;
let lastNodesUpdateAt = 0;
let lastLabelsUpdateAt = 0;

function truncateLabel(s, maxChars) {
  const str = (s ?? '').toString();
  if (!maxChars || str.length <= maxChars) return str;
  return str.slice(0, Math.max(0, maxChars - 1)).trimEnd() + '…';
}

function extractTopKeywordsFromText(items, { maxItems = 18, maxWords = 3 } = {}) {
  if (!items || !items.length) return [];
  const stop = new Set([
    'the','a','an','and','or','but','in','on','at','to','for','of','with','by','from',
    'is','are','was','were','be','been','being','have','has','had','do','does','did',
    'will','would','could','should','may','might','must','shall','can','this','that',
    'these','those','i','you','he','she','it','we','they','what','which','who','when',
    'where','why','how','all','each','every','both','few','more','most','other','some',
    'such','no','not','only','same','so','than','too','very','just','also','now','here',
    'there','then','if','as','because','until','while','about','into','through','during',
    'before','after','above','below'
  ]);

  const counts = new Map();
  const take = items.slice(0, maxItems);
  for (const it of take) {
    const text = (it?.preview || it?.full_text || it?.id || '').toString();
    // Basic tokenization; keep it cheap.
    const words = text
      .toLowerCase()
      .split(/[^a-z]+/g)
      .filter(Boolean);
    for (const w of words) {
      if (w.length < 4) continue;
      if (stop.has(w)) continue;
      counts.set(w, (counts.get(w) || 0) + 1);
    }
  }

  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, maxWords)
    .map(([w]) => w);
}

function buildClusterLabelLod(cluster, { mode, itemsForCluster } = {}) {
  const label = (cluster?.label || `Cluster ${cluster?.id ?? ''}`.trim());
  const broad = label;
  const mid = label;

  let detail = mid;
  if (mode === 'text') {
    const kws = extractTopKeywordsFromText(itemsForCluster, { maxItems: 20, maxWords: 3 })
      .map(w => w[0]?.toUpperCase() + w.slice(1));
    if (kws.length) detail = `${mid} · ${kws.join(', ')}`;
  }

  return { broad, mid, detail };
}

function labelTierForZoom(zoomK) {
  // Map-like: fewer, broader names when zoomed out; more specific as you zoom in.
  if (zoomK < 1.6) return 0; // broad
  if (zoomK < 3.2) return 1; // mid
  return 2; // detail
}

function smoothstep(edge0, edge1, x) {
  const t = Math.max(0, Math.min(1, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

function labelStyleForZoom(zoomK, baseSize) {
  // Smoothly interpolate styles across zoom so far→mid→near feels designed.
  const z = zoomK || 1;

  // Presets (no outlines; readability comes from opacity + subtle shadow).
  const far = {
    fontSize: Math.min(22, Math.max(11, baseSize * 1.22)),
    fontWeight: 720,
    letterSpacingEm: 0.18,
    fill: '#e2e8f0',
    fillOpacity: 0.84
  };
  const mid = {
    fontSize: Math.min(18, Math.max(10, baseSize * 1.04)),
    fontWeight: 620,
    letterSpacingEm: 0.12,
    fill: '#cbd5e1',
    fillOpacity: 0.70
  };
  const near = {
    fontSize: Math.min(14, Math.max(9, baseSize * 0.92)),
    fontWeight: 520,
    letterSpacingEm: 0.08,
    fill: '#94a3b8',
    fillOpacity: 0.62
  };

  // Blend far->mid around ~1.6, mid->near around ~3.2.
  const t01 = smoothstep(1.25, 1.95, z);
  const t12 = smoothstep(2.85, 3.75, z);

  const mixNum = (a, b, t) => a + (b - a) * t;
  const mixColor = (a, b, t) => d3.interpolateRgb(a, b)(t);

  const a = {
    fontSize: mixNum(far.fontSize, mid.fontSize, t01),
    fontWeight: mixNum(far.fontWeight, mid.fontWeight, t01),
    letterSpacingEm: mixNum(far.letterSpacingEm, mid.letterSpacingEm, t01),
    fill: mixColor(far.fill, mid.fill, t01),
    fillOpacity: mixNum(far.fillOpacity, mid.fillOpacity, t01)
  };

  const out = {
    fontSize: mixNum(a.fontSize, near.fontSize, t12),
    fontWeight: mixNum(a.fontWeight, near.fontWeight, t12),
    letterSpacingEm: mixNum(a.letterSpacingEm, near.letterSpacingEm, t12),
    fill: mixColor(a.fill, near.fill, t12),
    fillOpacity: mixNum(a.fillOpacity, near.fillOpacity, t12)
  };

  // Shadow fades out with zoom (prevents "outlined text" look).
  const shadowStrength = 1 - smoothstep(2.4, 4.0, z);
  const filter = shadowStrength > 0.55
    ? 'url(#label-shadow-strong)'
    : shadowStrength > 0.18
      ? 'url(#label-shadow-soft)'
      : null;

  return {
    fontSize: out.fontSize,
    fontWeight: Math.round(out.fontWeight),
    letterSpacing: `${out.letterSpacingEm.toFixed(3)}em`,
    fill: out.fill,
    fillOpacity: out.fillOpacity,
    filter
  };
}

function labelImportanceForRank(rank, total) {
  // 0 = "country", 1 = "state", 2 = "county"
  const n = Math.max(1, total || 1);
  // Keep it stable for small-ish cluster counts.
  const topCountries = Math.min(4, Math.max(2, Math.round(n * 0.12)));
  const topStates = Math.min(14, Math.max(topCountries + 4, Math.round(n * 0.42)));
  if (rank < topCountries) return 0;
  if (rank < topStates) return 1;
  return 2;
}

function applyImportanceToLabelStyle(style, importance) {
  const imp = importance ?? 2;
  if (imp === 0) {
    return {
      ...style,
      fontSize: style.fontSize * 1.65,
      fontWeight: Math.max(style.fontWeight, 760),
      fillOpacity: Math.min(1, style.fillOpacity + 0.12),
      letterSpacing: '0.20em'
    };
  }
  if (imp === 1) {
    return {
      ...style,
      fontSize: style.fontSize * 1.25,
      fontWeight: Math.max(style.fontWeight, 650),
      fillOpacity: Math.min(1, style.fillOpacity + 0.06),
      letterSpacing: '0.14em'
    };
  }
  return style;
}

function approxLabelBBoxScreen({ text, x, y, fontSize, letterSpacingEm = 0.1, zoomK = 1, pad = 0 }) {
  // Approximate screen-space bbox for a mono-ish font; good enough for collision culling.
  const t = (text || '').toString();
  const chars = t.length;
  const fsPx = (fontSize || 10) * (zoomK || 1);
  const lsPx = Math.max(0, letterSpacingEm) * fsPx;
  const charW = fsPx * 0.62; // IBM Plex Mono-ish
  const w = chars ? (chars * charW + Math.max(0, chars - 1) * lsPx) : 0;
  const h = fsPx * 1.18;
  return {
    x0: x - w / 2 - pad,
    x1: x + w / 2 + pad,
    y0: y - h / 2 - pad,
    y1: y + h / 2 + pad
  };
}

function bboxesOverlap(a, b) {
  return !(a.x1 < b.x0 || a.x0 > b.x1 || a.y1 < b.y0 || a.y0 > b.y1);
}

function pickLabelForZoom(d, zoomK, { isIndicator = false } = {}) {
  const tier = isIndicator ? 2 : labelTierForZoom(zoomK);
  const lod = d?.labelLod || { broad: d?.label, mid: d?.label, detail: d?.label };
  const raw = tier === 0 ? lod.broad : tier === 1 ? lod.mid : lod.detail;
  // Clamp lengths per tier so it behaves like map labels.
  const max = isIndicator ? 48 : tier === 0 ? 16 : tier === 1 ? 26 : 34;
  return truncateLabel((raw || '').toUpperCase(), max);
}

export function renderTopography(data, container) {
  const rect = container.getBoundingClientRect();
  const width = Math.max(0, rect.width || window.innerWidth);
  const height = Math.max(0, rect.height || window.innerHeight);
  const isCoarsePointer = typeof window !== 'undefined'
    && typeof window.matchMedia === 'function'
    && window.matchMedia('(pointer: coarse)').matches;
  const isMobile = isCoarsePointer || Math.min(width, height) <= 640;
  const margin = { top: 50, right: 50, bottom: 50, left: 50 };
  const mode = data.mode || 'image';
  
  container.innerHTML = '';
  
  const svg = d3.select(container)
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .style('touch-action', 'none')
    .style('cursor', 'grab');
  
  const g = svg.append('g').attr('class', 'zoom-layer');
  const contoursGroup = g.append('g');
  const cardsGroup = g.append('g').attr('class', 'cards-layer');
  const nodesGroup = g.append('g').attr('class', 'nodes-layer');
  const labelsGroup = g.append('g').attr('class', 'labels-layer');
  
  // Region indicator (fixed position, shows when zoomed in)
  // Add a feathered "scrim" behind the text to dim the area without looking like a flat black badge.
  const defs = svg.append('defs');
  // Shared card shadow (applied to card bg rects)
  if (defs.select('#card-shadow').empty()) {
    const f = defs.append('filter')
      .attr('id', 'card-shadow')
      .attr('x', '-40%')
      .attr('y', '-40%')
      .attr('width', '180%')
      .attr('height', '180%');
    f.append('feDropShadow')
      .attr('dx', 0)
      .attr('dy', 10)
      .attr('stdDeviation', 10)
      .attr('flood-color', '#000000')
      .attr('flood-opacity', 0.55);
  }

  // Label shadows: subtle readability without hard outlines.
  if (defs.select('#label-shadow-strong').empty()) {
    const f = defs.append('filter')
      .attr('id', 'label-shadow-strong')
      .attr('x', '-25%')
      .attr('y', '-25%')
      .attr('width', '150%')
      .attr('height', '150%');
    f.append('feDropShadow')
      .attr('dx', 0)
      .attr('dy', 1)
      .attr('stdDeviation', 2.2)
      .attr('flood-color', '#020617')
      .attr('flood-opacity', 0.65);
  }

  if (defs.select('#label-shadow-soft').empty()) {
    const f = defs.append('filter')
      .attr('id', 'label-shadow-soft')
      .attr('x', '-20%')
      .attr('y', '-20%')
      .attr('width', '140%')
      .attr('height', '140%');
    f.append('feDropShadow')
      .attr('dx', 0)
      .attr('dy', 1)
      .attr('stdDeviation', 1.4)
      .attr('flood-color', '#020617')
      .attr('flood-opacity', 0.45);
  }

  defs.append('radialGradient')
    .attr('id', 'region-indicator-scrim-gradient')
    .attr('cx', '50%')
    .attr('cy', '55%')
    .attr('r', '75%')
    .call(g => {
      g.append('stop').attr('offset', '0%').attr('stop-color', '#020617').attr('stop-opacity', 0.78);
      g.append('stop').attr('offset', '52%').attr('stop-color', '#020617').attr('stop-opacity', 0.4);
      g.append('stop').attr('offset', '74%').attr('stop-color', '#00d4ff').attr('stop-opacity', 0.1);
      g.append('stop').attr('offset', '100%').attr('stop-color', '#000000').attr('stop-opacity', 0);
    });

  const regionIndicatorWrap = svg.append('g')
    .attr('class', 'region-indicator-wrap')
    .attr('opacity', 0);

  // Base dim layer (covers full text area; avoids "circular spot" falloff missing the ends)
  const regionIndicatorDim = regionIndicatorWrap.append('rect')
    .attr('class', 'region-indicator-dim')
    .attr('fill', '#020617')
    .attr('rx', 10)
    .attr('ry', 10)
    .attr('width', 0)
    .attr('height', 0);

  // Feather layer (adds soft edges + subtle tint)
  const regionIndicatorScrim = regionIndicatorWrap.append('rect')
    .attr('class', 'region-indicator-scrim')
    .attr('fill', 'url(#region-indicator-scrim-gradient)')
    .attr('rx', 10)
    .attr('ry', 10)
    .attr('width', 0)
    .attr('height', 0);

  const regionIndicatorText = regionIndicatorWrap.append('text')
    .attr('class', 'region-indicator')
    .attr('x', 20)
    .attr('y', 30)
    .attr('font-family', 'IBM Plex Mono, monospace')
    .attr('font-size', 12)
    .attr('fill', '#94a3b8')
    .attr('letter-spacing', '0.15em')
    .attr('opacity', 1);

  function layoutRegionIndicatorScrim() {
    const node = regionIndicatorText.node();
    if (!node) return;
    const bbox = node.getBBox();
    const padX = 26;
    const padY = 12;
    const x = bbox.x - padX;
    const y = bbox.y - padY;
    const w = bbox.width + padX * 2;
    const h = bbox.height + padY * 2;
    regionIndicatorDim
      .attr('x', x)
      .attr('y', y)
      .attr('width', w)
      .attr('height', h);
    regionIndicatorScrim
      .attr('x', x)
      .attr('y', y)
      .attr('width', w)
      .attr('height', h);
  }
  
  // Scales
  const points = data.items.map(d => d.umap);
  const xExtent = d3.extent(points, p => p[0]);
  const yExtent = d3.extent(points, p => p[1]);
  const xPad = (xExtent[1] - xExtent[0]) * 0.08;
  const yPad = (yExtent[1] - yExtent[0]) * 0.08;
  
  const xScale = d3.scaleLinear()
    .domain([xExtent[0] - xPad, xExtent[1] + xPad])
    .range([margin.left, width - margin.right]);
  
  const yScale = d3.scaleLinear()
    .domain([yExtent[0] - yPad, yExtent[1] + yPad])
    .range([height - margin.bottom, margin.top]);
  
  // Contours
  const densityData = d3.contourDensity()
    .x(d => xScale(d.umap[0]))
    .y(d => yScale(d.umap[1]))
    .size([width, height])
    .bandwidth(22)
    .thresholds(14)
    (data.items);
  
  const colorScale = d3.scaleSequential()
    .domain([0, d3.max(densityData, d => d.value)])
    .interpolator(t => d3.interpolateRgb('#0a1628', '#00d4ff')(Math.pow(t, 0.6)));
  
  const contourPaths = contoursGroup.selectAll('path')
    .data(densityData)
    .join('path')
    .attr('class', 'contour-path')
    .attr('d', d3.geoPath())
    .attr('stroke', d => colorScale(d.value))
    .attr('stroke-opacity', 0.6)
    .attr('fill', d => colorScale(d.value))
    .attr('fill-opacity', 0.04);
  
  
  // Nodes
  const clusterLabelById = new Map((data.clusters || []).map(c => [c.id, c.label]));
  const getClusterLabelFast = (item) => {
    if (!item || item.cluster === -1) return 'Unclustered';
    return clusterLabelById.get(item.cluster) || `Cluster ${item.cluster}`;
  };

  const dataNodes = nodesGroup.selectAll('circle')
    .data(data.items)
    .join('circle')
    .attr('class', 'data-point')
    .attr('cx', d => xScale(d.umap[0]))
    .attr('cy', d => yScale(d.umap[1]))
    .attr('r', 12)
    .attr('fill', d => d.cluster === -1 ? '#475569' : d3.schemeTableau10[d.cluster % 10])
    .attr('fill-opacity', 0.4)
    .on('mouseenter', (event, d) => {
      showItemTooltip(event, d, getClusterLabelFast(d));
    })
    .on('mousemove', updateTooltipPosition)
    .on('mouseleave', hideTooltip)
    .on('click', (event, d) => {
      event.stopPropagation();
      showLightbox(d, getClusterLabelFast(d));
    });
  
  // Labels with collision detection (Google Maps style)
  const itemsByClusterId = new Map();
  for (const it of (data.items || [])) {
    if (!it || it.cluster == null || it.cluster === -1) continue;
    if (!itemsByClusterId.has(it.cluster)) itemsByClusterId.set(it.cluster, []);
    itemsByClusterId.get(it.cluster).push(it);
  }

  const labelData = data.clusters.map(c => {
    const itemsForCluster = itemsByClusterId.get(c.id) || [];
    return ({
      ...c,
      x: xScale(c.centroid[0]),
      y: yScale(c.centroid[1]) + 12,
      visible: true,
      labelLod: buildClusterLabelLod(c, { mode, itemsForCluster })
    });
  });
  
  // Sort by cluster size (larger clusters get priority)
  labelData.sort((a, b) => b.size - a.size);

  // Assign a stable "map hierarchy" importance level based on rank (size).
  for (let i = 0; i < labelData.length; i++) {
    labelData[i].rank = i;
    labelData[i].importance = labelImportanceForRank(i, labelData.length);
  }
  
  const clusterLabels = labelsGroup.selectAll('text')
    .data(labelData)
    .join('text')
    .attr('class', 'cluster-label')
    .attr('x', d => d.x)
    .attr('y', d => d.y)
    .attr('font-family', 'IBM Plex Mono, monospace')
    .attr('text-anchor', 'middle')
    .text(d => pickLabelForZoom(d, 1)) // overwritten in updateViz; set initial text
    .on('click', (event, cluster) => {
      event.stopPropagation();
      zoomToCluster(cluster, data.items, xScale, yScale, svg, zoom, width, height);
    });
  
  // Zoom limits
  const dataWidth = xScale(xExtent[1]) - xScale(xExtent[0]);
  const dataHeight = yScale(yExtent[0]) - yScale(yExtent[1]);
  // Fit-to-view zoom. Do NOT hard-floor near 1.0; that makes initial load feel "too zoomed in"
  // for wide/sparse datasets. Keep only a tiny floor for pathological cases.
  const fitSlack = 0.84; // smaller => zoom out more by default
  const minZoom = Math.max(0.05, Math.min(width / dataWidth, height / dataHeight) * fitSlack);
  const maxZoom = 25;

  function fitToDataTransform() {
    // Center the data bounds in the viewport at minZoom so the whole "map" is visible on first load.
    const midX = (xExtent[0] + xExtent[1]) / 2;
    const midY = (yExtent[0] + yExtent[1]) / 2;
    const cx = xScale(midX);
    const cy = yScale(midY);
    return d3.zoomIdentity
      .translate(width / 2, height / 2)
      .scale(minZoom)
      .translate(-cx, -cy);
  }

  function clampTransformToExtent(t) {
    const k = Math.max(minZoom, Math.min(maxZoom, t?.k ?? 1));
    // Keep the same world-space center, even if we clamp k or the viewport changed.
    const cx = width / 2;
    const cy = height / 2;
    const wx = (cx - (t?.x ?? 0)) / (t?.k ?? 1);
    const wy = (cy - (t?.y ?? 0)) / (t?.k ?? 1);
    const x = cx - wx * k;
    const y = cy - wy * k;
    return d3.zoomIdentity.translate(x, y).scale(k);
  }

  // Cards: cache base positions + quadtree once per render (scale depends on width/height).
  const cardCache = buildCardCache(data, mode, xScale, yScale);

  // Render: at most once per frame (latest transform wins).
  const renderFrame = rafThrottle((k, transform, isZooming) => {
    g.attr('transform', transform);
    svg.style('cursor', transform.k > 1 ? 'grabbing' : 'grab');
    updateViz(
      k,
      transform,
      data,
      dataNodes,
      nodesGroup,
      clusterLabels,
      labelData,
      cardsGroup,
      contourPaths,
      xScale,
      yScale,
      width,
      height,
      mode,
      regionIndicatorWrap,
      regionIndicatorText,
      layoutRegionIndicatorScrim,
      cardCache,
      getClusterLabelFast,
      minZoom,
      isZooming,
      isMobile
    );
  });

  // Zoom
  let latestTransform = d3.zoomIdentity;
  let zooming = false;
  let lastZoomK = 1;

  // Refresh cards when images become "warm" so skeletons swap to photos without interaction.
  // Avoid listener leaks across re-renders (resize calls renderTopography again).
  if (container._imageWarmHandler) {
    window.removeEventListener('image:warm', container._imageWarmHandler);
  }
  let warmRefreshScheduled = false;
  const onWarm = () => {
    if (warmRefreshScheduled) return;
    warmRefreshScheduled = true;
    requestAnimationFrame(() => {
      warmRefreshScheduled = false;
      if (!container.isConnected) return;
      renderFrame(lastZoomK, latestTransform, false);
    });
  };
  container._imageWarmHandler = onWarm;
  window.addEventListener('image:warm', onWarm);

  const zoom = d3.zoom()
    .scaleExtent([minZoom, maxZoom])
    .filter(event => !event.ctrlKey && !event.button)
    .on('zoom', (event) => {
      zooming = true;
      latestTransform = event.transform;
      lastZoomK = event.transform.k;
      container._savedZoomTransform = event.transform;
      renderFrame(event.transform.k, event.transform, true);
    })
    .on('end', () => {
      zooming = false;
      lastZoomK = latestTransform.k;
      container._savedZoomTransform = latestTransform;
      renderFrame(latestTransform.k, latestTransform, false);
    });
  
  // Preserve zoom across re-renders (e.g. resize), and avoid the "limits changed" feel.
  const initialTransform = container._savedZoomTransform
    ? clampTransformToExtent(container._savedZoomTransform)
    : fitToDataTransform();
  latestTransform = initialTransform;
  lastZoomK = initialTransform.k;
  container._savedZoomTransform = initialTransform;

  svg.call(zoom);
  // Set initial transform on the zoom behavior so wheel/pinch uses the same baseline.
  svg.call(zoom.transform, initialTransform);
  renderFrame(initialTransform.k, initialTransform, false);
}

function updateViz(
  zoom,
  transform,
  data,
  dataNodes,
  nodesGroup,
  clusterLabels,
  labelData,
  cardsGroup,
  contourPaths,
  xScale,
  yScale,
  width,
  height,
  mode,
  regionIndicatorWrap,
  regionIndicatorText,
  layoutRegionIndicatorScrim,
  cardCache,
  getClusterLabelFast,
  minZoom = 1,
  isZooming = false,
  isMobile = false
) {
  const now = performance.now();
  // Crossfade zone: labels fade out 3-5x, cards fade in 5-7x
  const labelOpacity = zoom < 3 ? Math.min(0.9, 2 / zoom) : Math.max(0, 1 - (zoom - 3) / 2);
  const cardOpacity = zoom < 5 ? 0 : Math.min(1, (zoom - 5) / 2);
  const showCards = zoom > 5;
  
  // Region indicator - show when labels are fading out
  const indicatorOpacity = zoom > 4 ? Math.min(1, (zoom - 4) / 1.6) : 0;
  if (indicatorOpacity > 0) {
    const shouldUpdateRegion = !isZooming || (now - lastRegionUpdateAt) > 90;
    if (shouldUpdateRegion) {
      lastRegionUpdateAt = now;

      // Find cluster nearest to viewport center
      const centerX = (width / 2 - transform.x) / transform.k;
      const centerY = (height / 2 - transform.y) / transform.k;

      let nearestCluster = null;
      let minDist = Infinity;
      for (const c of labelData) {
        const dist = Math.hypot(c.x - centerX, c.y - centerY);
        if (dist < minDist) {
          minDist = dist;
          nearestCluster = c;
        }
      }

      if (nearestCluster) {
        const label = pickLabelForZoom(nearestCluster, zoom, { isIndicator: true });
        regionIndicatorText.text(label);
        if (label !== lastRegionLabel) {
          lastRegionLabel = label;
          layoutRegionIndicatorScrim(); // getBBox: only when text changes
        }
        regionIndicatorWrap.attr('opacity', indicatorOpacity);
      }
    } else {
      regionIndicatorWrap.attr('opacity', indicatorOpacity);
    }
  } else {
    regionIndicatorWrap.attr('opacity', 0);
  }
  
  // Contour lines - thinner and fade as zoom increases
  const contourOpacity = Math.max(0.1, 0.6 - (zoom - 1) * 0.08);
  contourPaths
    .attr('stroke-width', 2 / Math.pow(zoom, 1.5))
    .attr('stroke-opacity', contourOpacity)
    .attr('fill-opacity', contourOpacity * 0.07);
  
  // Nodes: avoid touching every circle every frame while zooming.
  const nodeOpacity = showCards ? Math.max(0, 0.85 - cardOpacity) : 0.85;
  nodesGroup.attr('opacity', nodeOpacity);

  const shouldUpdateNodeR = !isZooming || (now - lastNodesUpdateAt) > 70;
  if (shouldUpdateNodeR) {
    lastNodesUpdateAt = now;
    // Nodes live inside the zoomed layer, so their *screen* radius is `r * zoom`.
    // Match the *old* screen size at max zoom-out (minZoom), then shrink slightly as you zoom in.
    const z0 = Math.max(0.0001, minZoom || 1);
    const oldR0 = Math.max(4, 12 / Math.pow(z0, 0.4)); // old data-space radius at z0
    const baseScreenR = oldR0 * z0; // old on-screen radius at z0
    const minScreenR = isMobile ? 3.6 : 4.2;
    const zRel = Math.max(0, (zoom / z0) - 1);
    const shrink = isMobile ? 0.7 : 0.9;
    const screenR = Math.max(minScreenR, baseScreenR - shrink * Math.log1p(zRel));
    const nodeR = screenR / Math.max(0.0001, zoom);
    dataNodes.attr('r', nodeR);
  }
  
  // Labels: recompute/cull less often while zooming.
  const shouldUpdateLabels = !isZooming || (now - lastLabelsUpdateAt) > 90;
  if (shouldUpdateLabels) {
    lastLabelsUpdateAt = now;

    const labelSize = Math.max(8, 11 / Math.pow(zoom, 0.25));
    // Keep mid-zoom from becoming a "wall of text".
    // Budget peaks a bit when you zoom in, but stays capped; mobile is stricter.
    const maxLabels = isMobile
      ? (zoom < 1.35 ? 6 : zoom < 2.0 ? 10 : zoom < 2.8 ? 14 : zoom < 3.6 ? 16 : 18)
      : (zoom < 1.35 ? 10 : zoom < 2.0 ? 16 : zoom < 2.8 ? 22 : zoom < 3.6 ? 26 : 32);
    const baseStyle = labelStyleForZoom(zoom, labelSize);
    // Let maxLabels + collision do the filtering. Importance is still used for styling/priority.

    // Reset visibility
    for (let i = 0; i < labelData.length; i++) {
      const d = labelData[i];
      d.visible = i < maxLabels && zoom < 6;
    }

    // Screen-space bbox collision culling (greedy, map-style): never overlap.
    // labelData is already sorted by priority (largest clusters first).
    const placed = [];
    for (let i = 0; i < labelData.length; i++) {
      const d = labelData[i];
      if (!d.visible) continue;

      const styled = applyImportanceToLabelStyle(baseStyle, d.importance);
      const lsEm = parseFloat(styled.letterSpacing) || 0.1;
      const screenX = transform.applyX(d.x);
      const screenY = transform.applyY(d.y);
      const pad = (d.importance ?? 2) === 0 ? 10 : (d.importance ?? 2) === 1 ? 8 : 7;

      const bbox = approxLabelBBoxScreen({
        text: pickLabelForZoom(d, zoom),
        x: screenX,
        y: screenY,
        fontSize: styled.fontSize,
        letterSpacingEm: lsEm,
        zoomK: zoom,
        pad
      });

      let hit = false;
      for (let j = 0; j < placed.length; j++) {
        if (bboxesOverlap(bbox, placed[j])) { hit = true; break; }
      }
      if (hit) {
        d.visible = false;
      } else {
        placed.push(bbox);
      }
    }

    clusterLabels
      .attr('font-size', d => applyImportanceToLabelStyle(baseStyle, d.importance).fontSize)
      .attr('font-weight', d => applyImportanceToLabelStyle(baseStyle, d.importance).fontWeight)
      .attr('letter-spacing', d => applyImportanceToLabelStyle(baseStyle, d.importance).letterSpacing)
      .attr('fill', baseStyle.fill)
      .attr('fill-opacity', d => applyImportanceToLabelStyle(baseStyle, d.importance).fillOpacity)
      .attr('filter', baseStyle.filter)
      .text(d => pickLabelForZoom(d, zoom))
      .attr('opacity', d => d.visible ? labelOpacity : 0);
  }
  
  // Cards - keep elements, just toggle opacity for smooth transitions
  const targetOpacity = showCards ? cardOpacity : 0;
  const shouldUpdateCards = targetOpacity <= 0.001 ? true : (!isZooming || (now - lastCardsUpdateAt) > 50);
  if (shouldUpdateCards) {
    lastCardsUpdateAt = now;
    renderCardsCulled(cardsGroup, cardCache, zoom, targetOpacity, width, height, mode, transform, getClusterLabelFast, isZooming, isMobile);
  } else {
    // Cheap: keep prior set, just fade as needed
    cardsGroup.style('display', targetOpacity <= 0.001 ? 'none' : null);
    cardsGroup.style('opacity', targetOpacity);
  }
}

function computeCardOffsets(data, mode) {
  // Compute collision offsets in normalized space
  const items = data.items;
  const n = items.length;
  const offsets = new Array(n);
  const isText = mode === 'text';
  
  // Initialize with zero offsets
  for (let i = 0; i < n; i++) {
    offsets[i] = { dx: 0, dy: 0 };
  }
  
  // Use UMAP coordinates directly for collision (scale-independent)
  const coords = items.map(d => ({ x: d.umap[0], y: d.umap[1] }));
  
  // Find data extent and compute repulsion radius
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const c of coords) {
    minX = Math.min(minX, c.x);
    maxX = Math.max(maxX, c.x);
    minY = Math.min(minY, c.y);
    maxY = Math.max(maxY, c.y);
  }
  const extent = Math.max(maxX - minX, maxY - minY);
  // Text needs more repulsion (bigger cards), images less (tighter grouping)
  const repulsionFactor = isText ? 1.5 : 0.8;
  const repulsion = extent / Math.sqrt(n) * repulsionFactor;
  
  // Spatial hash for O(n) collision
  const cellSize = repulsion * 2;
  
  // Single pass - fast
  for (let pass = 0; pass < 1; pass++) {
    const grid = new Map();
    
    // Build spatial hash
    for (let i = 0; i < n; i++) {
      const x = coords[i].x + offsets[i].dx;
      const y = coords[i].y + offsets[i].dy;
      const cell = `${Math.floor(x / cellSize)},${Math.floor(y / cellSize)}`;
      if (!grid.has(cell)) grid.set(cell, []);
      grid.get(cell).push(i);
    }
    
    // Check neighbors
    for (let i = 0; i < n; i++) {
      const xi = coords[i].x + offsets[i].dx;
      const yi = coords[i].y + offsets[i].dy;
      const cx = Math.floor(xi / cellSize);
      const cy = Math.floor(yi / cellSize);
      
      for (let dcx = -1; dcx <= 1; dcx++) {
        for (let dcy = -1; dcy <= 1; dcy++) {
          const neighbors = grid.get(`${cx + dcx},${cy + dcy}`);
          if (!neighbors) continue;
          
          for (const j of neighbors) {
            if (j <= i) continue;
            const xj = coords[j].x + offsets[j].dx;
            const yj = coords[j].y + offsets[j].dy;
            const dx = xj - xi;
            const dy = yj - yi;
            const dist = Math.sqrt(dx * dx + dy * dy);
            
            if (dist < repulsion && dist > 0.001) {
              const push = (repulsion - dist) / 2 * 0.5;
              const nx = dx / dist;
              const ny = dy / dist;
              offsets[i].dx -= nx * push;
              offsets[i].dy -= ny * push;
              offsets[j].dx += nx * push;
              offsets[j].dy += ny * push;
            }
          }
        }
      }
    }
    
    // Pull back to origin each pass to maintain clustering
    for (let i = 0; i < n; i++) {
      offsets[i].dx *= 0.9;
      offsets[i].dy *= 0.9;
    }
  }
  
  return offsets;
}

function buildCardCache(data, mode, xScale, yScale) {
  const isText = mode === 'text';
  const items = data.items || [];
  const n = items.length;
  const offsets = computeCardOffsets(data, mode);

  const base = new Array(n);
  for (let i = 0; i < n; i++) {
    const it = items[i];
    const ox = offsets[i]?.dx || 0;
    const oy = offsets[i]?.dy || 0;
    const raw = it.preview || it.full_text || it.id || '';
    const clean = sanitizeTextPreview(raw);
    base[i] = {
      _i: i,
      id: it.id,
      cluster: it.cluster,
      preview: it.preview,
      content: it.content,
      full_text: it.full_text,
      x: xScale(it.umap[0] + ox),
      y: yScale(it.umap[1] + oy),
      clipId: `clip-${mode}-${i}`,
      cardTitle: isText ? String(it.id || '').trim() : '',
      cardSnippet: isText ? clean.slice(0, 260) : '',
      _item: it
    };
  }

  const quadtree = d3.quadtree()
    .x(d => d.x)
    .y(d => d.y)
    .addAll(base);

  return { base, quadtree };
}

function quadtreeQueryLimited(quadtree, x0, y0, x1, y1, limit) {
  const results = [];
  if (!quadtree || !quadtree.root() || limit <= 0) return results;

  quadtree.visit((node, nx0, ny0, nx1, ny1) => {
    if (results.length >= limit) return true;
    if (nx1 < x0 || nx0 > x1 || ny1 < y0 || ny0 > y1) return true;

    if (!node.length) {
      let n = node;
      while (n) {
        const d = n.data;
        if (d && d.x >= x0 && d.x <= x1 && d.y >= y0 && d.y <= y1) {
          results.push(d);
          if (results.length >= limit) return true;
        }
        n = n.next;
      }
    }
    return false;
  });

  return results;
}

function renderCardsCulled(cardsGroup, cardCache, zoom, opacity, width, height, mode, transform, getClusterLabelFast, isZooming, isMobile) {
  if (!cardCache) return;
  if (opacity <= 0.001) {
    cardsGroup.style('display', 'none');
    return;
  }
  cardsGroup.style('display', null);
  cardsGroup.style('opacity', opacity);

  const isText = mode === 'text';
  // Keep card geometry in "screen px" and counter-scale by 1/k so we only update one transform per card.
  // This is much smoother on mobile than updating width/height/rx/stroke/font-size for every card each zoom tick.
  const dims = isText
    ? (isMobile ? { w: 118, h: 74, padX: 10, padY: 10, titleSize: 10, bodySize: 9 } : { w: 132, h: 82, padX: 12, padY: 12, titleSize: 11, bodySize: 9.5 })
    : (isMobile ? { w: 60, h: 60 } : { w: 72, h: 72 });
  const cardWidthPx = dims.w;
  const cardHeightPx = dims.h;
  const invK = 1 / (transform?.k || zoom || 1);
  
  // Visible query in base SVG coords (invert zoom transform)
  const screenMargin = 240;
  const margin = screenMargin / transform.k;
  const x0 = (0 - transform.x) / transform.k - margin;
  const x1 = (width - transform.x) / transform.k + margin;
  const y0 = (0 - transform.y) / transform.k - margin;
  const y1 = (height - transform.y) / transform.k + margin;

  // Ramp in card count as opacity increases to avoid a big "creation spike" at the threshold.
  const maxCardsFull = isMobile ? (isText ? 90 : 140) : (isText ? 200 : 320);
  const cap = Math.max(40, Math.floor(maxCardsFull * Math.min(1, opacity + 0.15)));
  const queried = quadtreeQueryLimited(cardCache.quadtree, x0, y0, x1, y1, cap);
  // Freeze membership mid-gesture to avoid DOM churn spikes during pinch/drag (especially on mobile).
  const visible = (isZooming && cardCache._lastVisible) ? cardCache._lastVisible : queried;
  if (!isZooming) cardCache._lastVisible = visible;
  const loadImages = !isZooming && opacity > 0.45;
  
  const cardClass = isText ? 'text-card' : 'image-card';
  const cards = cardsGroup.selectAll(`.${cardClass}`).data(visible, d => d.id);
  
  if (isZooming) {
    // Avoid DOM churn mid-gesture; prune on zoom end (isZooming=false).
    cards.exit().style('display', 'none');
  } else {
    cards.exit().remove();
  }
  
  const enter = cards.enter()
    .append('g')
    .attr('class', cardClass);
  
  if (isText) {
    // Create clipPath for rounded corners + HTML text clipping
    enter.append('clipPath')
      .attr('id', d => d.clipId)
      .append('rect')
      .attr('class', 'clip-rect');

    enter.append('rect').attr('class', 'card-bg');
    enter.append('rect').attr('class', 'card-border').attr('fill', 'none');

    const fo = enter.append('foreignObject')
      .attr('class', 'card-fo')
      .attr('clip-path', d => `url(#${d.clipId})`);

    const root = fo.append('xhtml:div').attr('class', 'text-card-html');
    root.append('div').attr('class', 'text-card-title');
    root.append('div').attr('class', 'text-card-snippet');
  } else {
    // Create clipPath for rounded corners
    enter.append('clipPath')
      .attr('id', d => d.clipId)
      .append('rect')
      .attr('class', 'clip-rect');

    enter.append('rect').attr('class', 'card-bg');

    // Skeleton while warming/decoding the photo.
    enter.append('rect')
      .attr('class', 'card-skeleton')
      .attr('clip-path', d => `url(#${d.clipId})`);

    enter.append('image')
      .attr('href', d => {
        const url = getImageUrl(d.content);
        if (!loadImages) return '';
        if (!isImageWarmUrl(url)) warmImageUrl(url, { priority: 0 });
        return isImageWarmUrl(url) ? url : '';
      })
      .attr('clip-path', d => `url(#${d.clipId})`)
      .attr('preserveAspectRatio', 'xMidYMid slice');
    enter.append('rect').attr('class', 'card-border').attr('fill', 'none');
  }
  
  enter.on('click', (event, d) => {
    event.stopPropagation();
    showLightbox(d._item || d, getClusterLabelFast(d._item || d));
  });
  
  const all = enter.merge(cards);
  // If an element was previously hidden via exit() during zoom, ensure it becomes visible again.
  all.style('display', null);
  
  // Keep screen-px sizing via counter-scale. One transform write per card.
  all.attr('transform', d => `translate(${d.x}, ${d.y}) scale(${invK}) translate(${-cardWidthPx / 2}, ${-cardHeightPx / 2})`);
  
  const cornerRadiusPx = isMobile ? 10 : 13; // screen px
  
  if (isText) {
    all.select('.card-bg')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('fill', '#070b14')
      .attr('fill-opacity', 0.92)
      .attr('filter', (!isMobile && !isZooming) ? 'url(#card-shadow)' : null)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx);

    all.select('.card-border')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('stroke', d => d.cluster === -1 ? '#334155' : d3.schemeTableau10[d.cluster % 10])
      .attr('stroke-opacity', 0.7)
      .attr('stroke-width', 1.2)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx);

    all.select('.clip-rect')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx);

    const padX = dims.padX || 10;
    const padY = dims.padY || 10;
    const titleSize = dims.titleSize || 10.5;
    const bodySize = dims.bodySize || 9;
    const titleMax = 44;

    all.select('foreignObject.card-fo')
      .attr('x', 0)
      .attr('y', 0)
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .each(function(d) {
        const sel = d3.select(this);
        const html = sel.select('div.text-card-html');
        
        const key = `${cardWidthPx}|${cardHeightPx}|${padX}|${padY}|${bodySize}|${titleSize}`;
        if (d._layoutKey !== key) {
          d._layoutKey = key;
          html
            .style('padding', `${padY}px ${padX}px`)
            .style('font-size', `${bodySize}px`);
        }

        // Always update text content, even if layout key matches
        const title = String(d.cardTitle || d.id || '')
          .replace(/\s+/g, ' ')
          .trim()
          .slice(0, titleMax);
        const snippet = String(d.cardSnippet || d.preview || d.id || '')
          .replace(/\s+/g, ' ')
          .trim();

        html.select('.text-card-title')
          .style('font-size', `${titleSize}px`)
          .text(title);
        html.select('.text-card-snippet')
          .text(snippet);
      });
  } else {
    
    // Update clipPath rect
    all.select('.clip-rect')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx);

    const skeletonSel = all.select('.card-skeleton')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx)
      .attr('fill', '#0b1220');
    // Avoid per-card warm checks during active zoom gesture.
    if (!isZooming) {
      skeletonSel.attr('opacity', d => {
        const url = getImageUrl(d.content);
        return isImageWarmUrl(url) ? 0 : 1;
      });
    }
    
    const imgSel = all.select('image')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('clip-path', d => `url(#${d.clipId})`);

    // Never blank out an already-loaded image mid-zoom; only set href when we decide to load.
    if (loadImages) {
      imgSel.attr('href', d => {
        const url = getImageUrl(d.content);
        if (!isImageWarmUrl(url)) warmImageUrl(url, { priority: 0 });
        return isImageWarmUrl(url) ? url : '';
      });
    }
    
    all.select('.card-border')
      .attr('width', cardWidthPx)
      .attr('height', cardHeightPx)
      .attr('stroke', d => d.cluster === -1 ? '#475569' : d3.schemeTableau10[d.cluster % 10])
      .attr('stroke-opacity', 0.5)
      .attr('stroke-width', 1.3)
      .attr('rx', cornerRadiusPx)
      .attr('ry', cornerRadiusPx);
  }
}

function zoomToCluster(cluster, items, xScale, yScale, svg, zoom, width, height) {
  const clusterItems = items.filter(item => item.cluster === cluster.id);
  if (!clusterItems.length) return;
  
  const xs = clusterItems.map(d => xScale(d.umap[0]));
  const ys = clusterItems.map(d => yScale(d.umap[1]));
  
  const pad = 60;
  const x0 = Math.min(...xs) - pad;
  const x1 = Math.max(...xs) + pad;
  const y0 = Math.min(...ys) - pad;
  const y1 = Math.max(...ys) + pad;
  
  const scale = Math.min(width / (x1 - x0), height / (y1 - y0)) * 0.85;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  
  svg.transition()
    .duration(600)
    .ease(d3.easeCubicInOut)
    .call(zoom.transform, d3.zoomIdentity.translate(width/2, height/2).scale(scale).translate(-cx, -cy));
}
