from .base import render_template

from servers.templates.object_detection import _EXTRA_CSS

_SIGN_EXTRA_CSS = '''
.log-panel {
    font-family: 'Courier New', Courier, monospace;
    font-size: 11px;
    background: var(--bg-sidebar);
    border-radius: 4px;
    padding: 8px;
    max-height: 220px;
    overflow-y: auto;
    display: flex;
    flex-direction: column-reverse;
}
.log-line {
    padding: 1px 0;
    color: var(--text-muted);
    white-space: pre;
    overflow: hidden;
    text-overflow: ellipsis;
}
.log-ts   { color: #484f58; margin-right: 6px; }
.log-warn { color: #d6a63a; }
.log-ok   { color: var(--accent-green); }
.log-info { color: #58a6ff; }
.log-state{ color: #c77dff; }
'''

CONTENT = '''
    <div class="container">
        <div class="video-section">
            <div class="video-wrapper">
                <div id="stop-banner" class="stop-banner"></div>
                <img src="{{ url_for('video') }}" id="stream-img" class="stream">
            </div>
        </div>

        <div class="controls-section">
            <div class="card">
                <div class="card-header">Drive Control</div>
                <div style="display:flex;align-items:center;gap:16px;margin-bottom:12px">
                    <span id="run-indicator" style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#e74c3c;flex-shrink:0"></span>
                    <span id="run-label" style="font-size:14px;font-weight:600;color:var(--text-secondary)">STOPPED</span>
                </div>
                <div style="display:flex;gap:10px;margin-bottom:8px">
                    <button onclick="driveStart()" class="button success" style="flex:1">Start</button>
                    <button onclick="driveStop()"  class="button" style="flex:1;background:var(--accent-orange,#e67e22)">Stop</button>
                </div>
                {% if virtual %}
                <div style="display:flex;gap:10px;margin-bottom:8px">
                    <button id="mode-btn" onclick="toggleMode()" class="button" style="flex:1;background:#555">Manual</button>
                    <button onclick="resetPosition()" class="button" style="flex:1;background:#444">Reset</button>
                </div>
                {% else %}
                <div style="margin-bottom:8px">
                    <button id="mode-btn" onclick="toggleMode()" class="button" style="width:100%;background:#555">Manual</button>
                </div>
                {% endif %}
                <div id="key-panel" style="display:none">
                    <div class="key-display">
                        <div class="key-box key-up"    id="key-up">&#9650;</div>
                        <div class="key-box key-left"  id="key-left">&#9664;</div>
                        <div class="key-box key-down"  id="key-down">&#9660;</div>
                        <div class="key-box key-right" id="key-right">&#9654;</div>
                    </div>
                    <p style="text-align:center;font-size:11px;color:var(--text-muted);margin:4px 0 0">Arrow keys or WASD</p>
                </div>
            </div>

            {% if virtual %}
            <div class="card">
                <div class="card-header">Scene Objects</div>
                <button onclick="removeObjects('duckie')" class="button" style="width:100%;background:#555">Remove Duck &amp; Continue</button>
            </div>
            <div class="card">
                <div class="card-header">Map</div>
                <button id="scene-btn" onclick="switchScene()" class="button" style="width:100%;background:#446">Switch to Free Drive</button>
            </div>
            {% endif %}

            <div class="card trt-build-card" id="trt-build-card">
                <div class="card-header" id="trt-header">Building TensorRT Engine</div>
                <p class="trt-build-hint">Camera and driving work now — detection starts when done.</p>
                <div class="trt-ready" id="trt-ready" style="display:none">Detection started!</div>
            </div>

            <div class="card">
                <div class="card-header">Confidence Threshold</div>
                <div style="display:flex;align-items:center;gap:10px">
                    <input id="conf-slider" type="range" min="0" max="1" step="0.01" value="0.5"
                        style="flex:1" oninput="onThresholdChange(this.value)">
                    <span id="conf-value" style="font-size:13px;font-variant-numeric:tabular-nums;min-width:32px">0.50</span>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    Detection
                    <span style="font-size:11px;font-weight:400;color:var(--text-muted)" id="det-count"></span>
                </div>
                <div id="model-status" class="model-status ok">Loading…</div>
                <div id="detections" class="detections-list">
                    <div class="empty-state">Waiting for frames…</div>
                </div>
            </div>

            <div class="card">
                <div class="card-header">
                    Bot Logs
                    <span id="sign-state-badge" style="font-size:11px;font-weight:400;color:#c77dff"></span>
                    <button onclick="clearLogs()" style="font-size:10px;padding:2px 8px;background:#222;border:1px solid #444;color:#888;border-radius:3px;cursor:pointer;margin-left:auto">Clear</button>
                </div>
                <div id="bot-logs" class="log-panel">
                    <div class="log-line" style="color:var(--text-muted)">Waiting for data…</div>
                </div>
            </div>
        </div>
    </div>
'''

EXTRA_JS = '''
    function setRunningUI(isRunning) {
        const indicator = document.getElementById('run-indicator');
        const label     = document.getElementById('run-label');
        indicator.style.background = isRunning ? '#2ecc71' : '#e74c3c';
        label.textContent = isRunning ? 'RUNNING' : 'STOPPED';
        label.style.color = isRunning ? '#2ecc71' : 'var(--text-secondary)';
    }

    function driveStart() {
        postJSON('/start', {}).then(() => {
            setRunningUI(true);
            addLog('[Server] started — running', 'ok');
        });
    }

    function driveStop() {
        postJSON('/stop', {}).then(() => {
            setRunningUI(false);
            addLog('[Server] stopped', 'warn');
        });
    }

    function removeObjects(filter) {
        postJSON('/remove_objects', {filter: filter});
    }

    function resetPosition() {
        postJSON('/reset', {}).then(data => {
            if (data && data.running !== undefined) setRunningUI(data.running);
            addLog('[Server] position reset', 'info');
        });
    }

    let _currentScene = 'object_detection';
    function switchScene() {
        const target = _currentScene === 'object_detection' ? 'introduction' : 'object_detection';
        postJSON('/switch_scene', {scene: target}).then(data => {
            if (data && data.scene) {
                _currentScene = data.scene;
                _updateSceneBtn();
                _manualMode = !!data.manual_mode;
                const btn = document.getElementById('mode-btn');
                const panel = document.getElementById('key-panel');
                if (btn) btn.textContent = _manualMode ? 'Auto' : 'Manual';
                if (panel) panel.style.display = _manualMode ? 'block' : 'none';
            }
        });
    }
    function _updateSceneBtn() {
        const btn = document.getElementById('scene-btn');
        if (!btn) return;
        if (_currentScene === 'introduction') {
            btn.textContent = 'Switch to Lane Follow';
        } else {
            btn.textContent = 'Switch to Free Drive';
        }
    }

    let _manualMode = false;
    const keyState = {up: false, down: false, left: false, right: false};
    const keyMap = {
        'ArrowUp': 'up', 'w': 'up', 'W': 'up',
        'ArrowDown': 'down', 's': 'down', 'S': 'down',
        'ArrowLeft': 'left', 'a': 'left', 'A': 'left',
        'ArrowRight': 'right', 'd': 'right', 'D': 'right',
    };

    function updateKeyDisplay() {
        for (const [key, active] of Object.entries(keyState)) {
            const el = document.getElementById('key-' + key);
            if (el) el.classList.toggle('active', active);
        }
    }

    function sendKeys() {
        fetch('/keys', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(keyState)}).catch(() => {});
    }

    function toggleMode() {
        _manualMode = !_manualMode;
        postJSON('/set_mode', {mode: _manualMode ? 'manual' : 'auto'});
        const btn = document.getElementById('mode-btn');
        const panel = document.getElementById('key-panel');
        if (btn)   btn.textContent = _manualMode ? 'Auto' : 'Manual';
        if (panel) panel.style.display = _manualMode ? 'block' : 'none';
    }

    document.addEventListener('keydown', e => {
        const dir = keyMap[e.key];
        if (dir && !keyState[dir]) { e.preventDefault(); keyState[dir] = true; updateKeyDisplay(); if (_manualMode) sendKeys(); }
    });
    document.addEventListener('keyup', e => {
        const dir = keyMap[e.key];
        if (dir && keyState[dir]) { e.preventDefault(); keyState[dir] = false; updateKeyDisplay(); if (_manualMode) sendKeys(); }
    });
    window.addEventListener('blur', () => {
        Object.keys(keyState).forEach(k => keyState[k] = false);
        updateKeyDisplay(); if (_manualMode) sendKeys();
    });
    setInterval(() => { if (_manualMode && Object.values(keyState).some(Boolean)) sendKeys(); }, 150);

    let _sliderDirty = false;
    function onThresholdChange(value) {
        document.getElementById('conf-value').textContent = parseFloat(value).toFixed(2);
        _sliderDirty = true;
        postJSON('/set_threshold', {value: parseFloat(value)}).then(() => { _sliderDirty = false; });
    }

    // ---- Log panel ----
    const _MAX_LOG = 120;
    const _logLines = [];

    function _ts() {
        const n = new Date();
        return n.toTimeString().slice(0, 8);
    }

    function _esc(s) {
        return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    }

    function addLog(msg, cls) {
        _logLines.push({ts: _ts(), msg, cls: cls || ''});
        if (_logLines.length > _MAX_LOG) _logLines.shift();
        _renderLogs();
    }

    function clearLogs() {
        _logLines.length = 0;
        _renderLogs();
    }

    function _renderLogs() {
        const el = document.getElementById('bot-logs');
        if (!el) return;
        if (_logLines.length === 0) {
            el.innerHTML = '<div class="log-line" style="color:var(--text-muted)">No logs yet.</div>';
            return;
        }
        el.innerHTML = _logLines.slice().reverse().map(e =>
            '<div class="log-line ' + (e.cls ? 'log-' + e.cls : '') + '">' +
            '<span class="log-ts">' + e.ts + '</span>' + _esc(e.msg) + '</div>'
        ).join('');
    }

    // ---- Sign FSM state tracking ----
    let _prevSignState   = null;
    let _prevConfirmed   = [];
    let _prevSavedTag    = null;
    let _prevRedLine     = false;
    let _prevRedLocked   = false;
    let _prevDetSig      = '';
    let _prevRunning     = null;
    let _prevStoppedDet  = false;

    function _checkSignLogs(data) {
        const state    = data.sign_state || null;
        const dbg      = data.sign_debug || {};
        const confirmed = dbg.confirmed_tags || [];
        const savedTag  = dbg.saved_tag != null ? dbg.saved_tag : null;
        const redLine   = !!dbg.red_line;
        const redLocked = !!dbg.red_locked;
        const chosen    = dbg.chosen_turn || 'forward';
        const slowF     = dbg.slow_factor != null ? dbg.slow_factor.toFixed(3) : null;

        // State transitions
        if (state !== null && _prevSignState !== null && state !== _prevSignState) {
            if (state === 'STOPPED') {
                addLog('[SignBehavior] >>> STOPPED at red line', 'warn');
            } else if (state === 'SLOWING') {
                addLog('[SignBehavior] >>> SLOWING — red line close, braking', 'warn');
            } else if (state === 'YIELD_CREEP') {
                addLog('[SignBehavior] >>> YIELD_CREEP', 'info');
            } else if (state === 'CHECKPATH') {
                addLog('[SignBehavior] >>> CHECKPATH — sweeping for cars', 'info');
            } else if (state === 'POST_STOP') {
                addLog('[SignBehavior] >>> path clear — POST_STOP', 'ok');
            } else if (state === 'INTERSECT') {
                addLog('[SignBehavior] >>> INTERSECT — direction: ' + chosen, 'state');
            } else if (state === 'PRE_TURN') {
                addLog("[SignBehavior] >>> PRE_TURN for '" + chosen + "'", 'state');
            } else if (state === 'TURNING') {
                addLog("[SignBehavior] >>> TURNING '" + chosen + "'", 'state');
            } else if (state === 'EXITING') {
                addLog("[SignBehavior] >>> TURNING done — EXITING", 'state');
            } else if (state === 'MOVING' && (_prevSignState === 'POST_STOP' || _prevSignState === 'EXITING')) {
                addLog('[SignBehavior] >>> POST_STOP done — MOVING (red-line unlocked)', 'ok');
            } else if (state === 'MOVING') {
                addLog('[SignBehavior] >>> MOVING', 'ok');
            }
        }
        if (state !== null) _prevSignState = state;

        // Update badge
        if (state) {
            const badge = document.getElementById('sign-state-badge');
            if (badge) badge.textContent = state;
        }

        // Red line detected
        if (redLine && !_prevRedLine) {
            addLog('[SignBehavior] RED LINE DETECTED CLOSE', 'warn');
        }
        if (!redLine && _prevRedLine) {
            addLog('[SignBehavior] red line gone', '');
        }
        _prevRedLine = redLine;

        // Red lock released
        if (_prevRedLocked && !redLocked && _prevSignState === 'MOVING') {
            addLog('[SignBehavior] red-line lock released', 'ok');
        }
        _prevRedLocked = redLocked;

        // New confirmed tags
        const newTags = confirmed.filter(t => !_prevConfirmed.includes(t));
        if (newTags.length > 0) {
            addLog('[SignBehavior] confirmed tags: [' + confirmed.join(', ') + ']', 'info');
        }
        _prevConfirmed = confirmed.slice();

        // Saved tag changed
        if (savedTag !== _prevSavedTag) {
            if (savedTag !== null) {
                addLog('[SignBehavior] sign saved: tag ID ' + savedTag, 'info');
            }
            _prevSavedTag = savedTag;
        }
    }

    async function pollStatus() {
        try {
            const data = await fetch('/status').then(r => r.json());

            // Running state
            setRunningUI(data.running);
            if (_prevRunning !== null && data.running !== _prevRunning) {
                addLog(data.running ? '[Server] bot started' : '[Server] bot stopped',
                       data.running ? 'ok' : 'warn');
            }
            _prevRunning = data.running;

            // Scene sync
            if (data.current_scene && data.current_scene !== _currentScene) {
                _currentScene = data.current_scene;
                _updateSceneBtn();
            }

            // Stop banner + red outline on video
            const banner    = document.getElementById('stop-banner');
            const streamImg = document.getElementById('stream-img');
            if (data.stopped_by_detection) {
                banner.textContent = 'STOPPED — ' + data.stop_reason;
                banner.classList.add('active');
                streamImg.classList.add('stopped');
                if (!_prevStoppedDet) {
                    addLog('[SignBehavior] >>> vehicle seen — STOPPED: ' + data.stop_reason, 'warn');
                }
            } else {
                banner.classList.remove('active');
                streamImg.classList.remove('stopped');
                if (_prevStoppedDet) {
                    addLog('[SignBehavior] vehicle gone — resuming', 'ok');
                }
            }
            _prevStoppedDet = !!data.stopped_by_detection;

            // TRT build card
            const trtCard = document.getElementById('trt-build-card');
            if (data.trt_building) {
                trtCard.style.display = 'block';
            } else if (trtCard._wasBuilding) {
                document.getElementById('trt-header').textContent = 'TensorRT Engine Ready';
                document.getElementById('trt-ready').style.display = 'block';
                setTimeout(() => { trtCard.style.display = 'none'; trtCard._wasBuilding = false; }, 4000);
            } else {
                trtCard.style.display = 'none';
            }
            trtCard._wasBuilding = data.trt_building;

            // Model status chip
            const status = document.getElementById('model-status');
            if (data.status === 'not_initialized') {
                status.className = 'model-status building';
                status.textContent = 'Initializing…';
            } else if (data.trt_building) {
                status.className = 'model-status building';
                status.textContent = 'Building TensorRT engine…';
            } else if (data.model_loaded) {
                status.className = 'model-status ok';
                status.textContent = 'Model loaded';
            } else {
                status.className = 'model-status err';
                status.textContent = data.load_error || 'Model not loaded';
            }

            // Sync slider only when user isn't actively dragging
            if (!_sliderDirty && data.conf_threshold != null) {
                const slider = document.getElementById('conf-slider');
                slider.value = data.conf_threshold;
                document.getElementById('conf-value').textContent = data.conf_threshold.toFixed(2);
            }

            // Detections list
            const dets = data.detections || [];
            document.getElementById('det-count').textContent = dets.length ? dets.length + ' found' : '';

            const list = document.getElementById('detections');
            list.innerHTML = dets.length === 0
                ? '<div class="empty-state">No detections</div>'
                : dets.map(d =>
                    '<div class="det-row">' +
                    '<span class="det-class">' + d.class + '</span>' +
                    '<span class="det-score">' + d.score.toFixed(2) + '</span>' +
                    '<span class="det-bbox">[' + d.bbox.join(', ') + ']</span>' +
                    '</div>').join('');

            // New detections log
            const detSig = dets.map(d => d.class + ':' + d.score.toFixed(2)).join('|');
            if (detSig !== _prevDetSig && dets.length > 0) {
                dets.forEach(d => addLog('[Detection] ' + d.class + ' score=' + d.score.toFixed(2), 'info'));
            }
            _prevDetSig = detSig;

            // Sign FSM logs (only from virtual server which exposes sign_state)
            if (data.sign_state !== undefined) {
                _checkSignLogs(data);
            }

        } catch (e) {}
    }

    setInterval(pollStatus, 300);
    pollStatus();
'''

SIGN_DETECTION_TEMPLATE = render_template(
    'Final Project - Sign Detection',
    '{{ hostname }} — Drive',
    CONTENT,
    extra_css=_EXTRA_CSS + _SIGN_EXTRA_CSS,
    extra_js=EXTRA_JS,
)
