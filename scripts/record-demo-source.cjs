#!/usr/bin/env node
/**
 * record-demo-source.js — deterministic capture of a Muser interaction as the SOURCE footage
 * for a screenstudio-alt demo. Frame-steps a scripted flow (NOT recordVideo) so the output is
 * perfect CFR for the studio's camera/time-remap. Emits an events.jsonl (cursor moves, clicks,
 * keystrokes — each click/first-key carries a getBoundingClientRect bbox in VIDEO px) that the
 * studio uses for the synthetic cursor, keystroke chips, and element-aware auto-zoom.
 *
 * Flow: landing showcase (held) → cursor→click search → type "sunset over water" → Enter →
 * demo-filtered results (held idle) → cursor→click a result → image page → navigate BACK to the
 * landing showcase so the video LOOPS (last frame == first frame).
 *
 * Fixes baked in (learned the hard way, 2026-06):
 *  - demo route-intercept adds &demo=1 to /api/search AND /api/score (women/girls+NSFW filtered).
 *  - __MUSER_DEMO_SEED__ makes the showcase order deterministic → the loop-back grid == the opening.
 *  - clearTimeout(_searchTimer) after each keystroke → the gallery does NOT flicker mid-type
 *    (the deterministic stepper takes >800ms wall-clock/char, which would otherwise trip the debounce).
 *  - loop-back clears the query + calls showShowcase() and waits for every tile's .ld (decoded,
 *    paint-ready) so the loop has zero blank thumbnails.
 *
 * Usage: node record-demo-source.js <url> <out-dir> <events.jsonl> [fps=30] [query]
 */
const path = require('path');
function loadPlaywright(){ for (const p of ['playwright','/Users/conner/.npm/_npx/705bc6b22212b352/node_modules/playwright']){ try{return require(p);}catch(e){} } throw new Error('playwright not found'); }
const { chromium } = loadPlaywright();
const fs = require('fs');

const URL   = process.argv[2] || 'http://127.0.0.1:7777/#/search';
const DIR   = process.argv[3] || '/tmp/ssalt-muser/frames';
const EVT   = process.argv[4] || '/tmp/ssalt-muser/raw.events.jsonl';
const FPS   = parseInt(process.argv[5] || '30', 10);
const QUERY = process.argv[6] || 'sunset over water';
const VW = 1600, VH = 900, DSF = 3;        // native 16:9 → 4800x2700 source (crisp zooms)

const T_LANDING = 1.5, T_PRETYPE = 0.35, T_POSTTYPE = 0.5, T_IDLE = 3.0, T_IMAGE = 1.5;
const PER_CHAR = 0.12;
const addDemo = (u) => u + (u.includes('?') ? '&' : '?') + 'demo=1';

(async () => {
  fs.rmSync(DIR, { recursive: true, force: true }); fs.mkdirSync(DIR, { recursive: true });
  const browser = await chromium.launch({ headless: true });
  const ctx = await browser.newContext({ viewport: { width: VW, height: VH }, deviceScaleFactor: DSF });
  const page = await ctx.newPage();
  await page.addInitScript(() => { window.__MUSER_DEMO_SEED__ = 1337; });   // deterministic showcase
  await page.route('**/api/search*', r => r.continue({ url: addDemo(r.request().url()) }));
  await page.route('**/api/score*',  r => r.continue({ url: addDemo(r.request().url()) }));

  let frame = 0; const events = [];
  let curX = 120, curY = 760;                                  // parked cursor start (lower-left)
  const tNow = () => frame / FPS;
  const snap = async () => { await page.screenshot({ path: `${DIR}/f${String(frame).padStart(5,'0')}.jpg`, type: 'jpeg', quality: 92 }); frame++; };
  const holdFor = async (secs) => { const n = Math.max(1, Math.round(secs*FPS)); for (let i=0;i<n;i++) await snap(); };
  const moveCursor = async (tx, ty, secs) => {                 // eased cursor travel (emits move events)
    const n = Math.max(2, Math.round(secs*FPS)), x0 = curX, y0 = curY;
    for (let i=1;i<=n;i++){ const u=i/n, e=u<0.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
      const x=x0+(tx-x0)*e, y=y0+(ty-y0)*e;
      await page.mouse.move(x,y); events.push({type:'move',t:tNow(),x:Math.round(x),y:Math.round(y)}); await snap(); }
    curX=tx; curY=ty;
  };
  const cssToVid = (r) => [Math.round(r.x*DSF), Math.round(r.y*DSF), Math.round(r.width*DSF), Math.round(r.height*DSF)];

  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(700);

  // locate + clear the concept search input
  const sb = await page.evaluate(() => {
    window.scrollTo(0,0);
    const inp=[...document.querySelectorAll('input')].find(i=>/type a concept/i.test(i.placeholder||''));
    if(!inp) return null;
    inp.value=''; inp.dispatchEvent(new Event('input',{bubbles:true})); inp.setAttribute('data-rec-search','1');
    const r=inp.getBoundingClientRect(); return {x:r.x,y:r.y,w:r.width,h:r.height,cx:r.x+r.width/2,cy:r.y+r.height/2};
  });
  if(!sb){ console.error('search input not found'); process.exit(1); }
  process.stderr.write(`searchbar(css) cx=${sb.cx|0} cy=${sb.cy|0} w=${sb.w|0}\n`);

  // wait for the demo-filtered landing showcase thumbs to load
  await page.waitForFunction(() => {
    const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>80&&r.top>100&&r.top<880;});
    return imgs.length>=8 && imgs.every(im=>im.complete && im.naturalWidth>0);
  }, { timeout: 30000 }).catch(()=>process.stderr.write('landing thumb-wait timeout (continuing)\n'));
  await page.evaluate(()=>window.scrollTo(0,0));

  // measure the rendered query-text bbox (left-aligned in the wide input) for element-aware zoom
  const txtBox = await page.evaluate((q) => {
    const inp=document.querySelector('[data-rec-search]'); const cs=getComputedStyle(inp); const r=inp.getBoundingClientRect();
    const c=document.createElement('canvas').getContext('2d'); c.font=`${cs.fontSize} ${cs.fontFamily}`;
    const padL=parseFloat(cs.paddingLeft)||12, tw=c.measureText(q).width;
    return [r.x+padL, r.y+(r.height-parseFloat(cs.fontSize))/2, tw, parseFloat(cs.fontSize)];
  }, QUERY);
  const txtBoxV = cssToVid({x:txtBox[0],y:txtBox[1],width:txtBox[2],height:txtBox[3]});
  process.stderr.write(`querytext(css) bbox=${txtBox.map(n=>n|0)} → text w=${txtBox[2]|0}\n`);

  // ---- A. opening hold on the landing showcase ----
  await holdFor(T_LANDING);

  // ---- B. move cursor to the search box + click into it ----
  await moveCursor(sb.cx, sb.cy, 0.7);
  events.push({type:'down', t:tNow(), x:Math.round(sb.cx*DSF), y:Math.round(sb.cy*DSF), bbox: cssToVid({x:sb.x,y:sb.y,width:sb.w,height:sb.h})});
  await page.evaluate(()=>{ const i=document.querySelector('[data-rec-search]'); i.focus(); });
  await holdFor(T_PRETYPE);

  // ---- C. type the query (kill the live-search debounce each char so the grid doesn't flicker) ----
  let typed='';
  for (let ci=0; ci<QUERY.length; ci++){
    typed += QUERY[ci];
    events.push(ci===0 ? {type:'key', t:tNow(), key:QUERY[ci], bbox:txtBoxV} : {type:'key', t:tNow(), key:QUERY[ci]});
    await page.evaluate((val)=>{
      const i=document.querySelector('[data-rec-search]');
      i.value=val; i.dispatchEvent(new Event('input',{bubbles:true}));
      try{ clearTimeout(_searchTimer); }catch(e){}     // no mid-type gallery flicker
    }, typed);
    await holdFor(PER_CHAR);
  }
  await holdFor(T_POSTTYPE);

  // ---- D. Enter → submit; block on full results decode before capturing ----
  events.push({type:'key', t:tNow(), key:'↵'});
  await page.evaluate(()=>{ const i=document.querySelector('[data-rec-search]'); i.focus(); });
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => {
    const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>80&&r.top>100&&r.top<880;});
    return imgs.length>=8 && imgs.every(im=>im.complete && im.naturalWidth>0);
  }, { timeout: 30000 }).catch(()=>process.stderr.write('results thumb-wait timeout (continuing)\n'));
  await page.evaluate(async ()=>{ const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>80&&r.top<900;}); await Promise.allSettled(imgs.map(im=>im.decode?im.decode().catch(()=>{}):null)); });

  // ---- E. BROWSE the results by SCROLLING — moving content is what the auto-fast-forward finale
  // speeds through (a static grid doesn't read as "fast-forward"; a scroll zipping by does). ----
  await holdFor(1.4);                                          // static look at top results — room for the camera PAN beat
  {
    const n=Math.round(3.4*FPS), maxY=1700;                    // ease-scroll down through the gallery
    for(let i=1;i<=n;i++){ const u=i/n, e=u<0.5?2*u*u:1-Math.pow(-2*u+2,2)/2;
      await page.evaluate(y=>window.scrollTo(0,y), Math.round(maxY*e));
      events.push({type:'move', t:tNow(), x:Math.round(curX), y:Math.round(curY)});
      await snap();
    }
    // let any lazily-loaded tiles revealed by the scroll finish decoding before the loop-back
    await page.evaluate(async ()=>{ const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>80&&r.top>-200&&r.top<1100;}); await Promise.allSettled(imgs.map(im=>im.decode?im.decode().catch(()=>{}):null)); });
  }
  await holdFor(0.4);

  // ---- F. RETURN to the landing showcase so the video LOOPS (deterministic seed → identical grid).
  // Muser persists the last query (concept-bar state survives a soft reload), so a plain goto comes
  // back on the RESULTS. Wipe storage first → fresh load shows the showcase with an EMPTY bar, and
  // the deterministic seed makes the tiles identical to the opening. (No image-click beat — that 2nd
  // zoom was deemed unnecessary, and the click opened a lightbox rather than navigating cleanly.) ----
  await page.evaluate(()=>{ try{ localStorage.clear(); sessionStorage.clear(); }catch(e){} });
  await page.goto(URL, { waitUntil: 'networkidle' });
  await page.evaluate(()=>{                                  // ensure showcase mode + EMPTY bar (clean loop seam)
    window.scrollTo(0,0);
    // The actual query lives in #q (the concept bar builds it there) — clearing only the visible
    // 'type a concept' entry left the green query chip rendered. Clear #q too.
    const q=document.querySelector('#q'); if(q){ q.value=''; q.dispatchEvent(new Event('input',{bubbles:true})); }
    const ce=[...document.querySelectorAll('input')].find(i=>/type a concept/i.test(i.placeholder||''));
    if(ce){ ce.value=''; ce.dispatchEvent(new Event('input',{bubbles:true})); }
    if(typeof showShowcase==='function') showShowcase();
  });
  await page.waitForFunction(() => {                          // every visible tile faded in (.ld = decoded)
    const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>80 && r.top>100 && r.top<900;});
    return imgs.length>=8 && imgs.every(im=>im.complete && im.naturalWidth>0 && im.classList.contains('ld'));
  }, { timeout: 30000 }).catch(()=>process.stderr.write('loop showcase .ld-wait timeout (continuing)\n'));
  await page.evaluate(async ()=>{ const imgs=[...document.querySelectorAll('#results img')].filter(im=>{const r=im.getBoundingClientRect();return r.width>60 && r.top<940;}); await Promise.allSettled(imgs.map(im=>im.decode?im.decode().catch(()=>{}):null)); window.scrollTo(0,0); });
  await page.waitForTimeout(400);
  await moveCursor(120, 760, 0.6);                            // cursor back toward the parked opening position
  await holdFor(T_LANDING);

  fs.writeFileSync(EVT, events.map(e=>JSON.stringify(e)).join('\n')+'\n');
  await browser.close();
  process.stdout.write(JSON.stringify({frames:frame, dur:frame/FPS, fps:FPS, searchbar:sb, querytext:txtBoxV, events:events.length})+'\n');
})().catch(e => { console.error(e); process.exit(1); });
