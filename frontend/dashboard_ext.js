// Dashboard extension - multi-model display + app.js compatibility patches

// Fix topbar price chip ID compatibility
(function(){
  const v = document.getElementById('topbarPriceValue');
  if(v && !document.getElementById('topbarPriceChip')) v.id='topbarPriceChip';
})();

// Extend renderPrediction to update multi-model cards
const _origRenderPrediction = window.renderPrediction;
window.renderPrediction = function(result) {
  if(typeof _origRenderPrediction === 'function') _origRenderPrediction(result);

  // Signal action indicator
  const label = result.label || 'Neutral';
  const action = label === 'Uptrend' ? 'BUY' : label === 'Downtrend' ? 'SELL' : 'HOLD';
  const ind = document.getElementById('signalActionIndicator');
  if(ind){ ind.textContent = action; ind.className='t-signal-action-indicator action-'+action.toLowerCase(); }

  // Signal label color class
  const tl = document.getElementById('trendLabel');
  if(tl){ tl.className='t-signal-label '+(label==='Uptrend'?'positive':label==='Downtrend'?'negative':''); }

  const probs = result.probabilities || [0.333,0.333,0.333];
  const pDown=(Number(probs[0])*100).toFixed(1);
  const pNeutral=(Number(probs[1])*100).toFixed(1);
  const pUp=(Number(probs[2])*100).toFixed(1);

  // ASTA prob bars
  updateModelProbBars('astaProbBars', pDown, pNeutral, pUp);
  updateModelProbBars('tapeProbBars', pDown, pNeutral, pUp);

  // Verdicts
  const verdict = label+' — '+(Number(result.confidence||0)*100).toFixed(1)+'%';
  ['astaVerdict','tapeVerdict','mhvpVerdict'].forEach(id=>{
    const el=document.getElementById(id); if(el) el.textContent=verdict;
  });

  // MHVP horizons
  const hp = result.horizon_predictions || {};
  const entries = Object.entries(hp);
  const hmGrid = document.getElementById('horizonResults');
  if(hmGrid && entries.length){
    hmGrid.innerHTML = entries.map(([k,v])=>{
      const hl = k.replace('short_term','Short').replace('mid_term','Mid').replace('long_term','Long');
      const cls = (v.label||'').includes('Up')?'up':(v.label||'').includes('Down')?'down':'neutral';
      return `<div class="t-horizon-mini ${cls}"><small>${hl}</small><strong>${v.label||'—'}</strong></div>`;
    }).join('');
  }

  // Probabilities panel
  const probPanel = document.getElementById('probabilities');
  if(probPanel){
    probPanel.innerHTML = [['Downtrend',pDown,'danger'],['Neutral',pNeutral,'accent'],['Uptrend',pUp,'good']].map(([lbl,pct,cls])=>
      `<div class="t-prob-item"><small>${lbl}</small><strong style="color:var(--${cls})">${pct}%</strong></div>`
    ).join('');
  }
};

function updateModelProbBars(containerId, pDown, pNeutral, pUp){
  const c = document.getElementById(containerId); if(!c) return;
  const fills = c.querySelectorAll('.t-prob-fill');
  const pcts = c.querySelectorAll('.t-prob-row span:last-child');
  if(fills[0]){fills[0].style.width=pDown+'%';} if(pcts[0]) pcts[0].textContent=pDown+'%';
  if(fills[1]){fills[1].style.width=pNeutral+'%';} if(pcts[1]) pcts[1].textContent=pNeutral+'%';
  if(fills[2]){fills[2].style.width=pUp+'%';} if(pcts[2]) pcts[2].textContent=pUp+'%';
}

// Extend renderMetrics to use new card IDs
const _origRenderMetrics = window.renderMetrics;
window.renderMetrics = function(result){
  if(typeof _origRenderMetrics === 'function') _origRenderMetrics(result);
  const mg = document.getElementById('metrics');
  if(mg && mg.className.includes('t-metrics-grid')){
    mg.innerHTML = [
      ['Training Loss', Number(result.train_loss).toFixed(4)],
      ['Validation Loss', Number(result.val_loss).toFixed(4)],
      ['Accuracy', (Number(result.accuracy)*100).toFixed(2)+'%'],
      ['ASTA Speedup', Number(result.runtime_speedup).toFixed(2)+'x'],
    ].map(([k,v])=>`<div class="t-metric-item"><span>${k}</span><strong>${v}</strong></div>`).join('');
  }
};

// Alert count badge
const _origAddAlert = window.addAlert;
window.addAlert = function(msg, tone){
  if(typeof _origAddAlert === 'function') _origAddAlert(msg, tone);
  const cnt = document.getElementById('alertCount');
  if(cnt){ const n=parseInt(cnt.textContent||0)+1; cnt.textContent=n; cnt.style.display=n>0?'grid':'none'; }
};

// Breadcrumb sync
const _origSetTopbar = window.setTopbarState;
window.setTopbarState = function(opts){
  if(typeof _origSetTopbar === 'function') _origSetTopbar(opts);
  const bc = document.getElementById('topbarSymbolBreadcrumb');
  if(bc && opts.symbol) bc.textContent = opts.symbol + (opts.regime?' · '+opts.regime:'');
};
