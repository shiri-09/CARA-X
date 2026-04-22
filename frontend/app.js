/**
 * CARA-X Dashboard — Frontend Application
 * Connects to the FastAPI backend and renders all 6 cognitive layers.
 */

const API = 'http://localhost:8000/api';
let cy = null; // Cytoscape instance
let competenceChart = null;
let learningChart = null;

// ═══ Navigation ═══
function showPage(page) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    document.getElementById('page-' + page).classList.add('active');
    document.querySelector(`[data-page="${page}"]`).classList.add('active');

    // Load data for the page
    if (page === 'dashboard') loadDashboard();
    if (page === 'graph') loadGraph();
    if (page === 'experiments') loadExperiments();
    if (page === 'memory') loadMemory();
    if (page === 'metacognition') loadMetacognition();
}

// ═══ API Helpers ═══
async function api(path, opts = {}) {
    try {
        const res = await fetch(API + path, {
            headers: { 'Content-Type': 'application/json' },
            ...opts,
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || res.statusText);
        }
        return res.json();
    } catch (e) {
        if (e.message.includes('Failed to fetch')) {
            updateStatus(false);
            toast('Backend offline — start with: python -m cara.main', 'error');
        }
        console.error('API Error:', e);
        return null;
    }
}

function toast(msg, type = 'success') {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function addLog(msg) {
    const log = document.getElementById('activity-log');
    const now = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-time">${now}</span><span class="log-msg">${msg}</span>`;
    log.prepend(entry);
    if (log.children.length > 30) log.lastChild.remove();
}

function updateStatus(online) {
    const el = document.getElementById('system-status');
    if (online) {
        el.innerHTML = '<div class="status-dot online"></div><span>System Online</span>';
    } else {
        el.innerHTML = '<div class="status-dot" style="background:#f43f5e;box-shadow:0 0 8px rgba(244,63,94,0.5)"></div><span>Offline</span>';
    }
}

// ═══ Dashboard ═══
async function loadDashboard() {
    const data = await api('/status');
    if (!data) return;
    updateStatus(true);

    document.getElementById('val-steps').textContent = data.engine.total_steps.toLocaleString();
    document.getElementById('val-nodes').textContent = data.world_model.node_count;
    document.getElementById('val-edges').textContent = data.world_model.edge_count;
    document.getElementById('val-memories').textContent = data.memory.episodic.total_memories;

    const avgConf = data.world_model.avg_confidence || 0;
    document.getElementById('val-confidence').textContent = (avgConf * 100).toFixed(0) + '%';
    document.getElementById('val-hypotheses').textContent = data.reasoning.hypotheses.total || 0;

    // Competence chart
    const comp = data.metacognition.competence.summary;
    renderCompetenceChart(comp.high_confidence_edges, comp.medium_confidence_edges, comp.low_confidence_edges);
}

function renderCompetenceChart(high, med, low) {
    const ctx = document.getElementById('competence-chart');
    if (!ctx) return;
    
    if (competenceChart) competenceChart.destroy();
    
    competenceChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Confidence', 'Uncertain', 'Unknown'],
            datasets: [{
                data: [high || 0, med || 0, low || 0],
                backgroundColor: ['rgba(16,185,129,0.7)', 'rgba(245,158,11,0.7)', 'rgba(244,63,94,0.5)'],
                borderColor: ['#10b981', '#f59e0b', '#f43f5e'],
                borderWidth: 1.5,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            cutout: '65%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#9898b0', font: { size: 11, family: 'Inter' }, padding: 12 }
                }
            }
        }
    });
}

// ═══ Causal Graph ═══
async function loadGraph() {
    const data = await api('/graph');
    if (!data) return;
    renderGraph(data);
}

function renderGraph(graphData) {
    const container = document.getElementById('causal-graph');
    const elements = [];

    // Nodes
    if (graphData.nodes) {
        graphData.nodes.forEach(node => {
            const obs = node.metadata ? node.metadata.observation_count : 0;
            const unc = node.metadata ? node.metadata.uncertainty_level : 0.5;
            elements.push({
                data: {
                    id: node.id,
                    label: node.id.replace(/_/g, '\n'),
                    size: Math.max(25, Math.min(50, 20 + obs * 0.3)),
                    uncertainty: unc,
                }
            });
        });
    }

    // Edges
    if (graphData.edges) {
        graphData.edges.forEach(edge => {
            elements.push({
                data: {
                    id: edge.source + '-' + edge.target,
                    source: edge.source,
                    target: edge.target,
                    confidence: edge.metadata ? edge.metadata.confidence : 0.5,
                    evidenceType: edge.metadata ? edge.metadata.evidence_type : 'observational',
                }
            });
        });
    }

    if (cy) cy.destroy();

    cy = cytoscape({
        container: container,
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'width': 'data(size)',
                    'height': 'data(size)',
                    'background-color': function(ele) {
                        const u = ele.data('uncertainty') || 0.5;
                        if (u < 0.3) return '#10b981';
                        if (u < 0.6) return '#f59e0b';
                        return '#7c3aed';
                    },
                    'border-width': 2,
                    'border-color': 'rgba(255,255,255,0.15)',
                    'color': '#e8e8f0',
                    'font-size': '9px',
                    'font-family': 'Inter, sans-serif',
                    'font-weight': 600,
                    'text-wrap': 'wrap',
                    'text-max-width': '70px',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-outline-width': 2,
                    'text-outline-color': '#0a0a0f',
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': function(ele) { return 1 + (ele.data('confidence') || 0.5) * 3; },
                    'line-color': function(ele) {
                        return ele.data('evidenceType') === 'interventional' ? '#10b981' : '#7c3aed';
                    },
                    'target-arrow-color': function(ele) {
                        return ele.data('evidenceType') === 'interventional' ? '#10b981' : '#7c3aed';
                    },
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': function(ele) { return 0.3 + (ele.data('confidence') || 0.5) * 0.7; },
                    'arrow-scale': 0.8,
                }
            },
            {
                selector: 'node:selected',
                style: {
                    'border-color': '#a78bfa',
                    'border-width': 3,
                    'overlay-opacity': 0,
                }
            }
        ],
        layout: { name: 'cose', animate: true, animationDuration: 800, nodeRepulsion: 8000, idealEdgeLength: 100 },
    });

    // Click handler for node details
    cy.on('tap', 'node', function(evt) {
        const node = evt.target;
        showNodeInfo(node.data());
    });

    cy.on('tap', function(evt) {
        if (evt.target === cy) {
            document.getElementById('node-info').classList.add('hidden');
        }
    });
}

function showNodeInfo(nodeData) {
    const panel = document.getElementById('node-info');
    const title = document.getElementById('node-info-title');
    const content = document.getElementById('node-info-content');

    title.textContent = nodeData.id;
    content.innerHTML = `
        <div style="font-size:0.75rem; color: var(--text-secondary)">
            <div style="margin-bottom:0.5rem"><strong>Uncertainty:</strong> ${(nodeData.uncertainty * 100).toFixed(0)}%</div>
            <div class="conf-bar"><div class="conf-bar-fill" style="width:${(1 - nodeData.uncertainty) * 100}%;background:${nodeData.uncertainty < 0.3 ? 'var(--emerald)' : nodeData.uncertainty < 0.6 ? 'var(--amber)' : 'var(--purple)'}"></div></div>
        </div>
    `;
    panel.classList.remove('hidden');
}

function fitGraph() { if (cy) cy.fit(null, 30); }

function refreshGraph() { loadGraph(); toast('Graph refreshed'); }

function changeLayout() {
    if (!cy) return;
    const layout = document.getElementById('graph-layout').value;
    cy.layout({ name: layout, animate: true, animationDuration: 600, nodeRepulsion: 8000 }).run();
}

// ═══ Experiments ═══
async function loadExperiments() {
    const queue = await api('/hypotheses/queue');
    const all = await api('/hypotheses');

    const queueEl = document.getElementById('hypothesis-queue');
    if (queue && queue.length) {
        queueEl.innerHTML = queue.map(h => `
            <div class="data-item">
                <div class="data-item-title">${h.cause} → ${h.effect}</div>
                <div class="data-item-meta">${h.mechanism || 'No mechanism specified'}</div>
                <div class="data-item-meta">Status: ${h.status} | Confidence: ${(h.confidence * 100).toFixed(0)}% | Source: ${h.source}</div>
                <div class="conf-bar"><div class="conf-bar-fill" style="width:${h.confidence * 100}%;background:var(--purple)"></div></div>
            </div>
        `).join('');
    } else {
        queueEl.innerHTML = '<div class="data-item"><div class="data-item-meta">No hypotheses in queue. Run Discovery first.</div></div>';
    }

    const histEl = document.getElementById('experiment-history');
    if (all && all.length) {
        const confirmed = all.filter(h => h.status === 'confirmed');
        const refuted = all.filter(h => h.status === 'refuted');
        histEl.innerHTML = `
            <div class="data-item">
                <div class="data-item-title">Summary</div>
                <div class="data-item-meta">Total: ${all.length} | Confirmed: ${confirmed.length} | Refuted: ${refuted.length}</div>
            </div>
            ${all.slice(0, 10).map(h => `
                <div class="data-item">
                    <div class="data-item-title">${h.cause} → ${h.effect}</div>
                    <div class="data-item-meta">${h.status} (${(h.confidence * 100).toFixed(0)}%) — ${h.n_tests} tests</div>
                </div>
            `).join('')}
        `;
    } else {
        histEl.innerHTML = '<div class="data-item"><div class="data-item-meta">No experiments yet.</div></div>';
    }
}

async function testNextHypothesis() {
    addLog('Testing next hypothesis...');
    const result = await api('/hypotheses/test', { method: 'POST' });
    if (result) {
        const verb = result.supports ? 'SUPPORTS' : 'REFUTES';
        toast(`Test result: ${verb} ${result.hypothesis.cause} → ${result.hypothesis.effect}`);
        addLog(`Hypothesis test: ${verb} (${result.hypothesis.cause} → ${result.hypothesis.effect})`);
        loadExperiments();
    }
}

// ═══ Memory ═══
async function loadMemory() {
    const episodic = await api('/memory/episodic?n=15');
    const semantic = await api('/memory/semantic');
    const procedural = await api('/memory/procedural');

    // Episodic
    const epEl = document.getElementById('episodic-memories');
    if (episodic && episodic.memories.length) {
        epEl.innerHTML = `
            <div class="data-item" style="border-color:rgba(59,130,246,0.2)">
                <div class="data-item-title">Total: ${episodic.total}</div>
            </div>
            ${episodic.memories.slice(0, 10).map(m => `
                <div class="data-item">
                    <div class="data-item-title">${m.action || 'observe'}</div>
                    <div class="data-item-meta">${m.action_type} | Reward: ${(m.reward || 0).toFixed(3)} | Importance: ${(m.importance * 100).toFixed(0)}%</div>
                    <div class="data-item-meta">${m.consolidated ? '✓ Consolidated' : '○ Unconsolidated'} | Age: ${(m.age_hours || 0).toFixed(1)}h</div>
                </div>
            `).join('')}
        `;
    } else {
        epEl.innerHTML = '<div class="data-item"><div class="data-item-meta">No episodic memories. Run an episode first.</div></div>';
    }

    // Semantic
    const semEl = document.getElementById('semantic-summary');
    if (semantic) {
        semEl.innerHTML = `
            <div class="eval-grid">
                <div class="eval-metric"><div class="eval-metric-value" style="color:var(--emerald)">${semantic.high_confidence_edges || 0}</div><div class="eval-metric-label">High Conf</div></div>
                <div class="eval-metric"><div class="eval-metric-value" style="color:var(--amber)">${semantic.low_confidence_edges || 0}</div><div class="eval-metric-label">Low Conf</div></div>
                <div class="eval-metric"><div class="eval-metric-value" style="color:var(--purple-light)">${semantic.interventionally_verified || 0}</div><div class="eval-metric-label">Interventional</div></div>
                <div class="eval-metric"><div class="eval-metric-value">${semantic.edge_count || 0}</div><div class="eval-metric-label">Total Edges</div></div>
            </div>
        `;
    }

    // Procedural
    const procEl = document.getElementById('procedural-memories');
    if (procedural && procedural.procedures.length) {
        procEl.innerHTML = procedural.procedures.map(p => `
            <div class="data-item">
                <div class="data-item-title">${p.name}</div>
                <div class="data-item-meta">${p.goal}</div>
                <div class="data-item-meta">Success: ${(p.success_rate * 100).toFixed(0)}% | Executions: ${p.total_executions}</div>
            </div>
        `).join('');
    } else {
        procEl.innerHTML = '<div class="data-item"><div class="data-item-meta">No procedures learned yet. Run consolidation.</div></div>';
    }
}

// ═══ Metacognition ═══
async function loadMetacognition() {
    const priorities = await api('/metacognition/exploration');
    const curve = await api('/metacognition/learning-curve');

    const priEl = document.getElementById('exploration-priorities');
    if (priorities && priorities.length) {
        priEl.innerHTML = priorities.map((p, i) => `
            <div class="data-item">
                <div class="data-item-title">#${i + 1} ${p.edge}</div>
                <div class="data-item-meta">Intervene on: ${p.intervene_on} | Entropy: ${p.entropy.toFixed(3)} | Evidence: ${p.evidence_count}</div>
                <div class="data-item-meta">Priority Score: ${p.priority_score.toFixed(3)}</div>
                <div class="conf-bar"><div class="conf-bar-fill" style="width:${p.uncertainty * 100}%;background:var(--amber)"></div></div>
            </div>
        `).join('');
    } else {
        priEl.innerHTML = '<div class="data-item"><div class="data-item-meta">No exploration data yet.</div></div>';
    }

    // Learning curve
    if (curve && curve.length > 1) {
        const ctx = document.getElementById('learning-curve-chart');
        if (learningChart) learningChart.destroy();
        learningChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: curve.map(p => p.step),
                datasets: [{
                    label: 'MAE',
                    data: curve.map(p => p.mae),
                    borderColor: '#7c3aed',
                    backgroundColor: 'rgba(124,58,237,0.1)',
                    fill: true,
                    tension: 0.4,
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { labels: { color: '#9898b0', font: { family: 'Inter' } } } },
                scales: {
                    x: { ticks: { color: '#686880' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { ticks: { color: '#686880' }, grid: { color: 'rgba(255,255,255,0.03)' } },
                }
            }
        });
    }
}

// ═══ Actions ═══
async function runEpisode() {
    const steps = parseInt(document.getElementById('ctrl-steps')?.value || 50);
    const exploreEl = document.getElementById('ctrl-explore');
    const explore = exploreEl ? exploreEl.value / 100 : 0.3;

    addLog(`Running episode: ${steps} steps, explore=${(explore).toFixed(2)}`);
    toast('Running episode...');

    const result = await api('/environment/episode', {
        method: 'POST',
        body: JSON.stringify({ n_steps: steps, explore_rate: explore }),
    });
    if (result) {
        toast(`Episode complete! Avg reward: ${result.avg_reward.toFixed(3)}`);
        addLog(`Episode ${result.episode}: ${result.observations_stored} observations, reward=${result.avg_reward.toFixed(3)}`);
        loadDashboard();
    }
}

async function runDiscovery() {
    addLog('Running causal discovery...');
    toast('Running causal discovery...');
    const result = await api('/discovery/run', { method: 'POST', body: '{}' });
    if (result) {
        toast(`Discovered ${result.edges_discovered} edges using ${result.algorithms_used.join(', ')}`);
        addLog(`Discovery: ${result.edges_discovered} edges, ${result.hypotheses_generated} hypotheses`);
        loadDashboard();
    }
}

async function runConsolidation() {
    addLog('Running memory consolidation...');
    const result = await api('/consolidate', { method: 'POST', body: '{}' });
    if (result) {
        toast(`Consolidated: ${result.patterns_promoted} patterns promoted`);
        addLog(`Consolidation: ${result.patterns_promoted} promoted, ${result.memories_pruned} pruned`);
        loadDashboard();
    }
}

async function runIntervention() {
    const variable = document.getElementById('ctrl-var')?.value;
    const value = parseFloat(document.getElementById('ctrl-val')?.value || 0);
    if (!variable) { toast('Enter a variable name', 'error'); return; }

    addLog(`Intervening: do(${variable}=${value})`);
    const result = await api('/intervene', {
        method: 'POST',
        body: JSON.stringify({ variable, value }),
    });
    if (result) {
        toast(`Intervention complete: do(${variable}=${value})`);
        addLog(`Intervention result: ${JSON.stringify(result).substring(0, 100)}...`);
    }
}

async function evaluateGraph() {
    const result = await api('/graph/evaluate');
    if (result) {
        const el = document.getElementById('evaluation-results');
        el.innerHTML = `
            <div class="eval-metric"><div class="eval-metric-value" style="color:var(--emerald)">${(result.precision * 100).toFixed(1)}%</div><div class="eval-metric-label">Precision</div></div>
            <div class="eval-metric"><div class="eval-metric-value" style="color:var(--blue)">${(result.recall * 100).toFixed(1)}%</div><div class="eval-metric-label">Recall</div></div>
            <div class="eval-metric"><div class="eval-metric-value" style="color:var(--purple-light)">${(result.f1 * 100).toFixed(1)}%</div><div class="eval-metric-label">F1 Score</div></div>
            <div class="eval-metric"><div class="eval-metric-value" style="color:var(--amber)">${result.shd}</div><div class="eval-metric-label">SHD</div></div>
            <div class="eval-metric"><div class="eval-metric-value">${result.true_positives}</div><div class="eval-metric-label">True Pos</div></div>
            <div class="eval-metric"><div class="eval-metric-value">${result.false_positives}</div><div class="eval-metric-label">False Pos</div></div>
        `;
        toast(`Evaluation: F1=${(result.f1 * 100).toFixed(1)}%, SHD=${result.shd}`);
    }
}

// ═══ Explore rate slider ═══
document.addEventListener('DOMContentLoaded', () => {
    const slider = document.getElementById('ctrl-explore');
    const valEl = document.getElementById('ctrl-explore-val');
    if (slider && valEl) {
        slider.addEventListener('input', () => {
            valEl.textContent = (slider.value / 100).toFixed(2);
        });
    }

    // Initial load
    loadDashboard();
});
