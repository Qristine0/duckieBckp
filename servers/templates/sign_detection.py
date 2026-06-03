from .base import render_template

from servers.templates.object_detection import _EXTRA_CSS

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
        postJSON('/start', {}).then(() => setRunningUI(true));
    }

    function driveStop() {
        postJSON('/stop', {}).then(() => setRunningUI(false));
    }

    function removeObjects(filter) {
        postJSON('/remove_objects', {filter: filter});
    }

    function resetPosition() {
        postJSON('/reset', {}).then(data => {
            if (data && data.running !== undefined) setRunningUI(data.running);
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
  
'''

SIGN_DETECTION_TEMPLATE = render_template(
    'Final Project - Sign Detection',
    '{{ hostname }} — Drive',
    CONTENT,
    extra_css=_EXTRA_CSS,
    extra_js=EXTRA_JS,
)
