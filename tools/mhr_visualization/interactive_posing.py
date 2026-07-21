#!/usr/bin/env python3
"""Interactive MHR model posing via web browser.

Run:  pixi run python tools/mhr_visualization/interactive_posing.py
Then open http://localhost:5000 in your browser.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import json
import torch
from flask import Flask, request, jsonify, render_template_string

import tools.mhr_visualization.MHR_vis as vis

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "assets", "mhr_model.pt")

# --- Load model and compute metadata ---
vis.load_model(MODEL_PATH)

joint_names = list(vis._mhr_model.get_joint_names())
num_joints = len(joint_names)
pose_parameter_names = list(vis._mhr_model.get_parameter_names()[:-45])

influence_matrix = vis._mhr_model.get_parameter_transform().numpy().astype(bool).T
influence_mapping = {}
for pname, jmask in zip(pose_parameter_names, influence_matrix):
    influence_mapping[pname] = []
    indices = np.nonzero(jmask.reshape(num_joints, 7).sum(1))[0].tolist()
    for idx in indices:
        influence_mapping[pname].append(joint_names[idx])

pose_parameter_limits_list = vis._mhr_model.get_parameter_limits()
# Convert to simple float pairs
pose_parameter_limits = []
for item in pose_parameter_limits_list:
    pose_parameter_limits.append([float(item[0]), float(item[1])])

num_pca_comp = vis._mhr_model.get_num_identity_blendshapes()

# Global state
parameter_values = {param: 0.0 for param in pose_parameter_names}
current_parameter = pose_parameter_names[0]

# --- Flask app ---
app = Flask(__name__)

MAIN_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MHR Interactive Posing</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: Arial, sans-serif; margin: 0; background: #f0f2f5; }
  .header { background: #1a73e8; color: #fff; padding: 12px 20px; font-size: 18px; }
  .container { display: flex; height: calc(100vh - 48px); }
  .controls { width: 380px; padding: 16px; background: #fff; border-right: 1px solid #ddd;
              overflow-y: auto; display: flex; flex-direction: column; gap: 12px; }
  .viz-area { flex: 1; position: relative; background: #e8eaed; }
  .viz-area iframe { width: 100%; height: 100%; border: none; }
  .viz-area .loading { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        font-size: 16px; color: #666; }
  label { font-weight: bold; font-size: 13px; color: #333; }
  select, input[type=range] { width: 100%; }
  select { padding: 6px; font-size: 13px; }
  input[type=range] { accent-color: #1a73e8; }
  .btn-row { display: flex; gap: 8px; }
  .btn-row button { flex: 1; padding: 8px; border: none; border-radius: 4px;
                    font-size: 13px; cursor: pointer; }
  .btn-warning { background: #f0ad4e; color: #fff; }
  .btn-danger { background: #d9534f; color: #fff; }
  .btn-warning:hover { background: #ec971f; }
  .btn-danger:hover { background: #c9302c; }
  .info-box { background: #f8f9fa; border: 1px solid #ddd; border-radius: 4px;
              padding: 12px; font-size: 12px; line-height: 1.6; }
  .info-box .param-name { color: #1a73e8; font-size: 14px; font-weight: bold; }
  .info-box .param-val { color: #e8710a; font-size: 16px; font-weight: bold; }
  .info-box .joints-list { color: #2e7d32; }
  .slider-val { text-align: center; font-size: 12px; color: #666; }
</style>
</head>
<body>
<div class="header">MHR Interactive Posing</div>
<div class="container">
  <div class="controls">
    <label>Parameter:</label>
    <select id="param-dropdown"></select>

    <label>Value:</label>
    <input type="range" id="param-slider" min="-3" max="3" step="0.01" value="0">
    <div class="slider-val" id="slider-display">0.00</div>

    <div class="btn-row">
      <button class="btn-warning" id="reset-current">Reset Current</button>
      <button class="btn-danger" id="reset-all">Reset All</button>
    </div>

    <div class="info-box" id="info-box">
      <div class="param-name" id="info-name">—</div>
      <div>Min: <span id="info-min">—</span> &nbsp; Max: <span id="info-max">—</span></div>
      <div>Current: <span class="param-val" id="info-val">0.0000</span></div>
      <div style="margin-top:6px;"><strong>Affected joints:</strong></div>
      <div class="joints-list" id="info-joints">—</div>
    </div>
  </div>
  <div class="viz-area">
    <div class="loading" id="loading-text">Loading visualization…</div>
    <iframe id="viz-iframe" src="/visualization" style="display:none;"></iframe>
  </div>
</div>

<script>
  const params = {{ params_json | safe }};
  const limits = {{ limits_json | safe }};
  const initialParam = "{{ initial_param }}";

  const dropdown = document.getElementById('param-dropdown');
  const slider = document.getElementById('param-slider');
  const sliderDisplay = document.getElementById('slider-display');
  const infoName = document.getElementById('info-name');
  const infoMin = document.getElementById('info-min');
  const infoMax = document.getElementById('info-max');
  const infoVal = document.getElementById('info-val');
  const infoJoints = document.getElementById('info-joints');
  const vizIframe = document.getElementById('viz-iframe');
  const loadingText = document.getElementById('loading-text');

  // Populate dropdown
  params.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p; opt.textContent = p;
    dropdown.appendChild(opt);
  });
  dropdown.value = initialParam;

  function getCurrentLimits() {
    const idx = params.indexOf(dropdown.value);
    return limits[idx] || [-3, 3];
  }

  function updateSliderRange() {
    const [min, max] = getCurrentLimits();
    slider.min = min; slider.max = max; slider.step = 0.01;
  }

  function fetchParamInfo() {
    fetch('/param_info?param=' + encodeURIComponent(dropdown.value))
      .then(r => r.json())
      .then(data => {
        infoName.textContent = data.name;
        infoMin.textContent = data.min.toFixed(4);
        infoMax.textContent = data.max.toFixed(4);
        infoVal.textContent = data.value.toFixed(4);
        infoJoints.textContent = data.joints.length > 0 ? data.joints.join(', ') : 'None';
      });
  }

  function updateVisualization() {
    loadingText.style.display = 'block';
    vizIframe.style.display = 'none';
    vizIframe.src = '/visualization?t=' + Date.now();
  }

  vizIframe.addEventListener('load', () => {
    loadingText.style.display = 'none';
    vizIframe.style.display = 'block';
  });

  dropdown.addEventListener('change', () => {
    updateSliderRange();
    // Fetch current stored value for this param
    fetch('/param_info?param=' + encodeURIComponent(dropdown.value))
      .then(r => r.json())
      .then(data => {
        slider.value = data.value;
        sliderDisplay.textContent = data.value.toFixed(2);
      });
    fetchParamInfo();
    // Tell server which param is selected (for affected joints highlighting)
    fetch('/set_current?param=' + encodeURIComponent(dropdown.value));
    updateVisualization();
  });

  slider.addEventListener('input', () => {
    sliderDisplay.textContent = parseFloat(slider.value).toFixed(2);
    fetch('/update?param=' + encodeURIComponent(dropdown.value) +
          '&value=' + slider.value);
    fetchParamInfo();
  });

  slider.addEventListener('change', () => {
    updateVisualization();
  });

  document.getElementById('reset-current').addEventListener('click', () => {
    fetch('/reset_current?param=' + encodeURIComponent(dropdown.value));
    slider.value = 0;
    sliderDisplay.textContent = '0.00';
    fetchParamInfo();
    updateVisualization();
  });

  document.getElementById('reset-all').addEventListener('click', () => {
    fetch('/reset_all');
    slider.value = 0;
    sliderDisplay.textContent = '0.00';
    fetchParamInfo();
    updateVisualization();
  });

  // Initial setup
  updateSliderRange();
  fetchParamInfo();
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(
        MAIN_PAGE,
        params_json=json.dumps(pose_parameter_names),
        limits_json=json.dumps(pose_parameter_limits),
        initial_param=current_parameter,
    )


@app.route("/visualization")
def visualization():
    affected = influence_mapping.get(current_parameter, [])
    pose_parm_values = np.array(
        [parameter_values[p] for p in pose_parameter_names]
    ).astype(np.float32)[np.newaxis, ...]
    html = vis.visualize_posed_mhr_model(
        pose_parameters=pose_parm_values,
        affected_joints_names=affected,
        output="html",
    )
    return html


@app.route("/param_info")
def param_info():
    param = request.args.get("param", current_parameter)
    if param not in pose_parameter_names:
        param = current_parameter
    idx = pose_parameter_names.index(param)
    min_val, max_val = pose_parameter_limits[idx]
    return jsonify({
        "name": param,
        "min": min_val,
        "max": max_val,
        "value": parameter_values.get(param, 0.0),
        "joints": influence_mapping.get(param, []),
    })


@app.route("/update")
def update():
    param = request.args.get("param")
    value = float(request.args.get("value", 0.0))
    parameter_values[param] = value
    return jsonify(success=True)


@app.route("/set_current")
def set_current():
    global current_parameter
    current_parameter = request.args.get("param")
    return jsonify(success=True)


@app.route("/reset_current")
def reset_current():
    param = request.args.get("param", current_parameter)
    parameter_values[param] = 0.0
    return jsonify(success=True)


@app.route("/reset_all")
def reset_all():
    for p in pose_parameter_names:
        parameter_values[p] = 0.0
    return jsonify(success=True)


if __name__ == "__main__":
    import webbrowser
    import threading

    port = 5000
    url = f"http://localhost:{port}"

    def open_browser():
        webbrowser.open(url)

    threading.Timer(1.5, open_browser).start()

    print(f"Starting MHR Interactive Posing server at {url}")
    print("Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
