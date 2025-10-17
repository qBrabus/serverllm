#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flask-based management interface for vLLM model servers.

The original script has been lightly adapted to make it container friendly:
- the model storage root can be configured via the ``MODELS_ROOT`` environment variable;
- the Flask host/port/debug settings can be configured with ``FLASK_HOST``, ``FLASK_PORT``
  and ``FLASK_DEBUG``.

All the logic of the original script remains the same so that the behaviour is
unchanged for operators running it directly on bare metal.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request
from huggingface_hub import snapshot_download

# Rendre la console lisible (mais on garde des prints utiles)
logging.getLogger("werkzeug").setLevel(logging.WARNING)

app = Flask(__name__)

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
MODELS_ROOT = Path(
    os.environ.get("MODELS_ROOT", "/raid/workspace/qladane/llm/models")
)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)

# État global
running_servers: Dict[str, Dict[str, Any]] = {}
download_progress: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------
# Helpers backend
# ---------------------------------------------------------------------
def get_gpu_info() -> List[Dict[str, Any]]:
    """Récupère les infos GPU via ``nvidia-smi``.

    ``nvidia-smi`` peut parfois ne pas être dans le PATH lorsque l'application
    est conteneurisée. Pour cette raison plusieurs emplacements standards sont
    essayés avant d'abandonner.
    """

    try:
        smi = shutil.which("nvidia-smi")
        if not smi:
            for cand in (
                "/usr/bin/nvidia-smi",
                "/usr/local/bin/nvidia-smi",
                "/usr/lib/nvidia-smi/nvidia-smi",
            ):
                if os.path.exists(cand):
                    smi = cand
                    break

        if not smi:
            print("[GPU] nvidia-smi introuvable (PATH et chemins connus).")
            return []

        cmd = [
            smi,
            "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
        if result.returncode != 0:
            print(
                f"[GPU] nvidia-smi rc={result.returncode} stderr={result.stderr.strip()}"
            )
            return []

        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
        gpus: List[Dict[str, Any]] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            gpus.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used": int(parts[2]),
                    "memory_total": int(parts[3]),
                    "utilization": int(parts[4]),
                    "temperature": int(parts[5]),
                    "in_use": False,
                }
            )
        if not gpus:
            print("[GPU] nvidia-smi n'a retourné aucune ligne exploitable.")
        else:
            print(f"[GPU] détectés: {len(gpus)} -> {[g['name'] for g in gpus]}")
        return gpus
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"[GPU] Exception: {exc}")
        return []


def get_dir_size(path: Path) -> float:
    total = 0
    try:
        for entry in path.rglob("*"):
            if entry.is_file():
                total += entry.stat().st_size
    except Exception:
        pass
    return round(total / (1024**3), 2)


def get_available_models() -> List[Dict[str, Any]]:
    """Liste tous les modèles téléchargés sous ``MODELS_ROOT``."""

    models: List[Dict[str, Any]] = []
    try:
        if not MODELS_ROOT.exists():
            print(f"[MODELS] Dossier inexistant: {MODELS_ROOT}")
            return models

        vendors = 0
        found = 0
        for vendor_dir in MODELS_ROOT.iterdir():
            if not vendor_dir.is_dir():
                continue
            vendors += 1
            for model_dir in vendor_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                has_config = (model_dir / "config.json").exists()
                has_nemo = any(model_dir.glob("*.nemo"))
                has_safetensors = any(model_dir.glob("*.safetensors"))
                if has_config or has_nemo or has_safetensors:
                    models.append(
                        {
                            "name": f"{vendor_dir.name}/{model_dir.name}",
                            "path": str(model_dir),
                            "size": get_dir_size(model_dir),
                        }
                    )
                    found += 1
        print(f"[MODELS] Vendors: {vendors} | Modèles détectés: {found}")
        return models
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"[MODELS] Exception: {exc}")
        return models


def download_model_task(model_id: str, task_id: str, hf_token: str | None = None) -> None:
    """Télécharge un modèle depuis Hugging Face dans un thread séparé."""

    try:
        download_progress[task_id] = {
            "status": "downloading",
            "progress": 0,
            "message": "Initialisation...",
        }
        vendor, model_name = model_id.split("/", 1)
        target_dir = MODELS_ROOT / vendor / model_name
        download_progress[task_id]["message"] = (
            f"Téléchargement vers {target_dir}..."
        )

        kwargs = {
            "repo_id": model_id,
            "local_dir": str(target_dir),
            "local_dir_use_symlinks": False,
            "ignore_patterns": ["*.msgpack", "*.h5", "*.onnx", "*/original/*"],
            "revision": None,
        }
        if hf_token:
            kwargs["token"] = hf_token

        snapshot_download(**kwargs)
        download_progress[task_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Téléchargement terminé",
        }
    except Exception as exc:  # pragma: no cover - diagnostic helper
        download_progress[task_id] = {
            "status": "error",
            "progress": 0,
            "message": str(exc),
        }


def start_vllm_server(
    server_id: str,
    model_path: str,
    port: int,
    gpus: List[int],
    tp_size: int,
    max_len: int,
    dtype: str,
) -> bool:
    """Lance un serveur vLLM."""

    try:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
        env["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        cmd = [
            "vllm",
            "serve",
            model_path,
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
            "--served-model-name",
            server_id,
            "--dtype",
            dtype,
            "--tensor-parallel-size",
            str(tp_size),
            "--max-model-len",
            str(max_len),
            "--gpu-memory-utilization",
            "0.90",
            "--trust-remote-code",
        ]

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        running_servers[server_id] = {
            "process": process,
            "config": {
                "model_path": model_path,
                "port": port,
                "gpus": gpus,
                "tp_size": tp_size,
                "max_len": max_len,
                "dtype": dtype,
            },
            "stats": {
                "start_time": datetime.now().isoformat(),
                "tokens_processed": 0,
                "requests": 0,
            },
            "log_lines": [],
        }

        def read_logs() -> None:
            for line in process.stdout:
                if server_id in running_servers:
                    running_servers[server_id]["log_lines"].append(line.strip())
                    if len(running_servers[server_id]["log_lines"]) > 200:
                        running_servers[server_id]["log_lines"].pop(0)

        threading.Thread(target=read_logs, daemon=True).start()
        return True
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print(f"[SERVER] Erreur démarrage: {exc}")
        return False


def stop_vllm_server(server_id: str) -> bool:
    if server_id in running_servers:
        try:
            process = running_servers[server_id]["process"]
            process.terminate()
            process.wait(timeout=10)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        finally:
            del running_servers[server_id]
        return True
    return False


# ---------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------
HTML_TEMPLATE = r"""
<!DOCTYPE html
<html>
<head>
    <title>vLLM Manager</title>
    <meta charset="utf-8">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: #0f172a; color: #e2e8f0; padding: 20px; }
        .container { max-width: 1600px; margin: 0 auto; }
        h1 { color: #60a5fa; margin-bottom: 10px; font-size: 2em; }
        #statusbar { margin-bottom: 20px; padding: 8px 12px; border-radius: 6px; background: #0b1222; color: #93c5fd; font-size: 13px; display:none; }
        h2 { color: #818cf8; margin: 30px 0 15px; font-size: 1.5em; }
        .section { background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }
        .card { background: #334155; padding: 15px; border-radius: 6px; border-left: 4px solid #60a5fa; }
        .gpu-card { border-left-color: #34d399; }
        .gpu-card.in-use { border-left-color: #f59e0b; background: #422006; }
        .server-card { border-left-color: #8b5cf6; }
        input, select, button { padding: 10px; border-radius: 4px; border: 1px solid #475569; background: #1e293b; color: #e2e8f0; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #60a5fa; }
        button { background: #3b82f6; color: white; cursor: pointer; border: none; font-weight: 600; transition: all 0.2s; }
        button:hover { background: #2563eb; transform: translateY(-1px); }
        button:disabled { background: #475569; cursor: not-allowed; }
        button.danger { background: #ef4444; }
        button.danger:hover { background: #dc2626; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; color: #94a3b8; font-weight: 500; }
        .status { display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: 600; }
        .status.running { background: #065f46; color: #34d399; }
        .status.downloading { background: #422006; color: #fb923c; }
        .progress { background: #1e293b; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .progress-bar { background: linear-gradient(90deg, #3b82f6, #8b5cf6); height: 100%; transition: width 0.3s; }
        .stat { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #334155; }
        .stat:last-child { border-bottom: none; }
        .gpu-checkbox { margin-right: 10px; }
        .refresh-btn { float: right; padding: 5px 15px; font-size: 12px; }
        code { background: #0b1222; padding: 2px 6px; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 vLLM Model Manager</h1>
        <div id="statusbar"></div>

        <div class="section">
            <h2>💻 GPUs Disponibles <button class="refresh-btn" onclick="manualRefresh()">🔄 Rafraîchir</button></h2>
            <div class="grid" id="gpu-grid"></div>
        </div>

        <div class="section">
            <h2>📥 Télécharger un Modèle</h2>
            <div class="form-group">
                <label>URL HuggingFace (ex: Qwen/Qwen3-VL-30B-A3B-Instruct)</label>
                <input type="text" id="model-url" placeholder="organization/model-name" style="width: 100%;">
            </div>
            <div class="form-group">
                <label>Token HuggingFace (optionnel, pour modèles privés/gated)</label>
                <input type="password" id="hf-token" placeholder="hf_..." style="width: 100%;">
            </div>
            <button onclick="downloadModel()">📥 Télécharger</button>
            <div id="download-status"></div>
        </div>

        <div class="section">
            <h2>📚 Modèles Disponibles</h2>
            <div class="grid" id="models-grid"></div>
        </div>

        <div class="section">
            <h2>▶️ Lancer un Serveur</h2>
            <div class="form-group">
                <label>Modèle</label>
                <select id="server-model" style="width: 100%;"></select>
            </div>
            <div class="form-group">
                <label>Nom du serveur</label>
                <input type="text" id="server-name" placeholder="my-model-server">
            </div>
            <div class="form-group">
                <label>Port</label>
                <input type="number" id="server-port" value="8000">
            </div>
            <div class="form-group">
                <label>Max Model Length</label>
                <input type="number" id="server-maxlen" value="256000">
            </div>
            <div class="form-group">
                <label>Dtype</label>
                <select id="server-dtype">
                    <option value="bfloat16">bfloat16</option>
                    <option value="float16">float16</option>
                    <option value="auto">auto</option>
                </select>
            </div>
            <div class="form-group">
                <label>Tensor Parallel Size</label>
                <input type="number" id="server-tp" value="1" min="1">
            </div>
            <div class="form-group">
                <label>GPUs à utiliser (sélectionnez-en)</label>
                <div id="gpu-selector"></div>
            </div>
            <button onclick="startServer()">▶️ Démarrer le Serveur</button>
        </div>

        <div class="section">
            <h2>🟢 Serveurs Actifs</h2>
            <div class="grid" id="servers-grid"></div>
        </div>
    </div>

    <script>
        // --- état global front
        let gpuData = [];
        let modelsData = [];
        let serversData = {};
        let isEditingServerForm = false;
        let editingTimeout = null;
        let pollTimer = null;

        const statusbar = document.getElementById('statusbar');

        function showStatus(msg, isError=false) {
            statusbar.style.display = 'block';
            statusbar.style.color = isError ? '#fecaca' : '#93c5fd';
            statusbar.style.background = isError ? '#3f1e1e' : '#0b1222';
            statusbar.textContent = msg;
        }
        function clearStatus() {
            statusbar.style.display = 'none';
            statusbar.textContent = '';
        }

        function armEditingGuard(ms = 4000) {
            isEditingServerForm = true;
            if (editingTimeout) clearTimeout(editingTimeout);
            editingTimeout = setTimeout(() => { isEditingServerForm = false; }, ms);
        }

        function readServerFormState() {
            const gpusSel = new Set(
                Array.from(document.querySelectorAll('.gpu-checkbox:checked'))
                     .map(cb => parseInt(cb.value))
            );
            return {
                model: (document.getElementById('server-model') || {}).value || "",
                name: (document.getElementById('server-name') || {}).value || "",
                port: (document.getElementById('server-port') || {}).value || "",
                maxlen: (document.getElementById('server-maxlen') || {}).value || "",
                dtype: (document.getElementById('server-dtype') || {}).value || "",
                tp: (document.getElementById('server-tp') || {}).value || "",
                gpus: gpusSel
            };
        }
        function restoreServerFormState(state) {
            if (!state) return;
            const select = document.getElementById('server-model');
            if (select && state.model && Array.from(select.options).some(o => o.value === state.model)) {
                select.value = state.model;
            }
            const name = document.getElementById('server-name'); if (name) name.value = state.name;
            const port = document.getElementById('server-port'); if (port) port.value = state.port;
            const maxlen = document.getElementById('server-maxlen'); if (maxlen) maxlen.value = state.maxlen;
            const dtype = document.getElementById('server-dtype'); if (dtype) dtype.value = state.dtype;
            const tp = document.getElementById('server-tp'); if (tp) tp.value = state.tp;

            const boxes = document.querySelectorAll('.gpu-checkbox');
            boxes.forEach(cb => {
                const idx = parseInt(cb.value);
                if (!cb.disabled) cb.checked = state.gpus.has(idx);
            });
        }

        async function fetchJson(url) {
            const res = await fetch(url);
            if (!res.ok) throw new Error(url + " -> HTTP " + res.status);
            return res.json();
        }

        async function fetchData() {
            try {
                const [gpus, models, servers] = await Promise.all([
                    fetchJson('/api/gpus'),
                    fetchJson('/api/models'),
                    fetchJson('/api/servers')
                ]);
                gpuData = gpus || [];
                modelsData = models || [];
                serversData = servers || {};
                clearStatus();
                updateUI();
            } catch (e) {
                showStatus("Erreur de mise à jour: " + (e.message || e), true);
                console.error('Erreur fetch:', e);
            }
        }

        function updateUI() {
            const prevState = readServerFormState();
            updateGPUGrid();
            updateModelsGrid();
            updateServersGrid();

            if (!isEditingServerForm) {
                updateModelSelector();
                updateGPUSelector();
            }
            restoreServerFormState(prevState);
            attachEditingGuards();
        }

        function updateGPUGrid() {
            const grid = document.getElementById('gpu-grid');
            const usedGPUs = new Set();
            Object.values(serversData).forEach(s => s.config.gpus.forEach(g => usedGPUs.add(g)));

            if (!gpuData || gpuData.length === 0) {
                grid.innerHTML = '<p style="color:#94a3b8;">Aucun GPU détecté. Vérifie que <code>nvidia-smi</code> est accessible pour ce process (PATH, droits, drivers).</p>';
                return;
            }
            grid.innerHTML = gpuData.map(gpu => {
                const inUse = usedGPUs.has(gpu.index);
                return `
                    <div class="card gpu-card ${inUse ? 'in-use' : ''}">
                        <div class="stat"><strong>GPU ${gpu.index}</strong><span>${inUse ? '🔴 Utilisé' : '🟢 Libre'}</span></div>
                        <div class="stat"><span>Nom</span><span>${gpu.name}</span></div>
                        <div class="stat"><span>Mémoire</span><span>${gpu.memory_used} / ${gpu.memory_total} MB</span></div>
                        <div class="stat"><span>Utilisation</span><span>${gpu.utilization}%</span></div>
                        <div class="stat"><span>Température</span><span>${gpu.temperature}°C</span></div>
                    </div>
                `;
            }).join('');
        }

        function updateModelsGrid() {
            const grid = document.getElementById('models-grid');
            if (!modelsData || modelsData.length === 0) {
                grid.innerHTML = '<p style="color: #94a3b8;">Aucun modèle détecté dans <code>{{ models_root }}</code>.</p>';
                return;
            }
            grid.innerHTML = modelsData.map(model => `
                <div class="card">
                    <div class="stat"><strong>${model.name}</strong></div>
                    <div class="stat"><span>Taille</span><span>${model.size} GB</span></div>
                    <div class="stat"><span>Chemin</span><span style="font-size: 11px; word-break: break-all;">${model.path}</span></div>
                </div>
            `).join('');
        }

        function updateServersGrid() {
            const grid = document.getElementById('servers-grid');
            const servers = Object.entries(serversData);
            if (servers.length === 0) {
                grid.innerHTML = '<p style="color: #94a3b8;">Aucun serveur actif</p>';
                return;
            }
            grid.innerHTML = servers.map(([id, server]) => {
                const uptime = Math.floor((Date.now() - new Date(server.stats.start_time)) / 1000);
                return `
                    <div class="card server-card">
                        <div class="stat"><strong>${id}</strong><span class="status running">RUNNING</span></div>
                        <div class="stat"><span>Port</span><span>${server.config.port}</span></div>
                        <div class="stat"><span>GPUs</span><span>${server.config.gpus.join(', ')}</span></div>
                        <div class="stat"><span>TP Size</span><span>${server.config.tp_size}</span></div>
                        <div class="stat"><span>Uptime</span><span>${uptime}s</span></div>
                        <div class="stat"><span>API</span><span>http://0.0.0.0:${server.config.port}/v1</span></div>
                        <button class="danger" style="width: 100%; margin-top: 10px;" onclick="stopServer('${id}')">⏹️ Arrêter</button>
                    </div>
                `;
            }).join('');
        }

        function updateModelSelector() {
            const select = document.getElementById('server-model');
            const prev = select.value;
            select.innerHTML = modelsData.map(m => `<option value="${m.path}">${m.name}</option>`).join('');
            if (Array.from(select.options).some(o => o.value === prev)) {
                select.value = prev;
            }
            if (!select.__guarded) {
                select.addEventListener('focusin', () => armEditingGuard(4000));
                select.addEventListener('change', () => armEditingGuard(4000));
                select.__guarded = true;
            }
        }

        function updateGPUSelector() {
            const div = document.getElementById('gpu-selector');
            const usedGPUs = new Set();
            Object.values(serversData).forEach(s => s.config.gpus.forEach(g => usedGPUs.add(g)));

            div.innerHTML = gpuData.map(gpu => {
                const disabled = usedGPUs.has(gpu.index) ? 'disabled' : '';
                const usedLabel = usedGPUs.has(gpu.index) ? '(Utilisé)' : '';
                return `
                    <label style="display: block; margin: 5px 0;">
                        <input type="checkbox" class="gpu-checkbox" value="${gpu.index}" ${disabled}>
                        GPU ${gpu.index} - ${gpu.name} ${usedLabel}
                    </label>
                `;
            }).join('');

            if (!div.__guarded) {
                div.addEventListener('mousedown', () => armEditingGuard(4000));
                div.addEventListener('change', () => armEditingGuard(4000));
                div.__guarded = true;
            }
        }

        function attachEditingGuards() {
            const ids = ['server-name','server-port','server-maxlen','server-dtype','server-tp'];
            ids.forEach(id => {
                const el = document.getElementById(id);
                if (!el || el.__guarded) return;
                el.addEventListener('focusin', () => armEditingGuard(5000));
                el.addEventListener('change', () => armEditingGuard(5000));
                el.addEventListener('input', () => armEditingGuard(5000));
                el.__guarded = true;
            });
        }

        async function downloadModel() {
            const url = document.getElementById('model-url').value.trim();
            const token = document.getElementById('hf-token').value.trim();
            if (!url) { alert('Entrez une URL'); return; }

            try {
                const res = await fetch('/api/download', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_id: url, hf_token: token || null})
                });
                const data = await res.json();
                if (data.success) {
                    alert('Téléchargement démarré !');
                    pollDownloadStatus(data.task_id);
                } else {
                    alert('Erreur: ' + data.message);
                }
            } catch (e) {
                showStatus("Erreur démarrage téléchargement: " + (e.message || e), true);
            }
        }

        async function pollDownloadStatus(taskId) {
            const statusDiv = document.getElementById('download-status');
            const interval = setInterval(async () => {
                try {
                    const res = await fetch(`/api/download/${taskId}`);
                    const data = await res.json();
                    statusDiv.innerHTML = `
                        <div style="margin-top: 15px;">
                            <span class="status downloading">${data.status}</span>
                            <div class="progress"><div class="progress-bar" style="width: ${data.progress}%"></div></div>
                            <p style="color: #94a3b8; margin-top: 5px;">${data.message}</p>
                        </div>
                    `;
                    if (data.status === 'completed' || data.status === 'error') {
                        clearInterval(interval);
                        setTimeout(() => fetchData(), 1500);
                    }
                } catch (e) {
                    clearInterval(interval);
                    showStatus("Erreur de suivi téléchargement: " + (e.message || e), true);
                }
            }, 1000);
        }

        async function startServer() {
            const model = document.getElementById('server-model').value;
            const name = document.getElementById('server-name').value.trim();
            const port = parseInt(document.getElementById('server-port').value);
            const maxlen = parseInt(document.getElementById('server-maxlen').value);
            const dtype = document.getElementById('server-dtype').value;
            const tp = parseInt(document.getElementById('server-tp').value);
            const gpus = Array.from(document.querySelectorAll('.gpu-checkbox:checked')).map(cb => parseInt(cb.value));

            if (!name || !model || gpus.length === 0) {
                alert('Remplissez tous les champs et sélectionnez au moins un GPU');
                return;
            }

            try {
                const res = await fetch('/api/servers/start', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({server_id: name, model_path: model, port, gpus, tp_size: tp, max_len: maxlen, dtype})
                });
                const data = await res.json();

                if (data.success) {
                    alert('Serveur démarré !');
                    isEditingServerForm = false;
                    fetchData();
                } else {
                    alert('Erreur: ' + data.message);
                }
            } catch (e) {
                showStatus("Erreur démarrage serveur: " + (e.message || e), true);
            }
        }

        async function stopServer(serverId) {
            if (!confirm(`Arrêter le serveur ${serverId} ?`)) return;
            try {
                const res = await fetch(`/api/servers/${serverId}`, {method: 'DELETE'});
                const data = await res.json();
                if (data.success) {
                    alert('Serveur arrêté');
                    fetchData();
                }
            } catch (e) {
                showStatus("Erreur arrêt serveur: " + (e.message || e), true);
            }
        }

        function manualRefresh() {
            if (!isEditingServerForm) fetchData();
        }

        function startPolling() {
            stopPolling();
            pollTimer = setInterval(() => {
                if (!isEditingServerForm) fetchData();
            }, 5000);
        }
        function stopPolling() { if (pollTimer) { clearInterval(pollTimer); pollTimer = null; } }

        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                if (!isEditingServerForm) fetchData();
                startPolling();
            } else {
                stopPolling();
            }
        });

        fetchData();
        startPolling();
    </script>
</body>
</html>
"""


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE, models_root=MODELS_ROOT)


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "time": time.time()})


@app.route("/api/gpus")
def api_gpus():
    gpus = get_gpu_info()
    for server in running_servers.values():
        for gpu_idx in server["config"]["gpus"]:
            for gpu in gpus:
                if gpu["index"] == gpu_idx:
                    gpu["in_use"] = True
    print(f"[API] /api/gpus -> {len(gpus)} GPU(s)")
    return jsonify(gpus)


@app.route("/api/models")
def api_models():
    models = get_available_models()
    print(f"[API] /api/models -> {len(models)} modèle(s)")
    return jsonify(models)


@app.route("/api/servers")
def api_servers():
    result = {}
    for sid, server in running_servers.items():
        result[sid] = {"config": server["config"], "stats": server["stats"]}
    return jsonify(result)


@app.route("/api/download", methods=["POST"])
def api_download():
    data = request.json or {}
    model_id = data.get("model_id")
    hf_token = data.get("hf_token")
    if not model_id:
        return jsonify({"success": False, "message": "model_id requis"})

    task_id = f"download_{int(time.time())}"
    thread = threading.Thread(
        target=download_model_task, args=(model_id, task_id, hf_token)
    )
    thread.start()

    return jsonify({"success": True, "task_id": task_id})


@app.route("/api/download/<task_id>")
def api_download_status(task_id: str):
    if task_id in download_progress:
        return jsonify(download_progress[task_id])
    return jsonify({"status": "unknown", "progress": 0, "message": "Tâche inconnue"})


@app.route("/api/servers/start", methods=["POST"])
def api_start_server():
    data = request.json or {}
    server_id = data.get("server_id")
    model_path = data.get("model_path")
    port = data.get("port", 8000)
    gpus = data.get("gpus", [])
    tp_size = data.get("tp_size", 1)
    max_len = data.get("max_len", 256000)
    dtype = data.get("dtype", "bfloat16")

    if not server_id or not model_path or not gpus:
        return jsonify({"success": False, "message": "Paramètres insuffisants"})

    if server_id in running_servers:
        return jsonify({"success": False, "message": "Serveur déjà actif"})

    success = start_vllm_server(server_id, model_path, port, gpus, tp_size, max_len, dtype)
    return jsonify({"success": success, "message": "Serveur démarré" if success else "Erreur démarrage"})


@app.route("/api/servers/<server_id>", methods=["DELETE"])
def api_stop_server(server_id: str):
    success = stop_vllm_server(server_id)
    return jsonify({"success": success, "message": "Serveur arrêté" if success else "Serveur introuvable"})


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "0.0.0.0")
    port = int(os.environ.get("FLASK_PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print(f"[i] Démarrage vLLM Manager sur http://{host}:{port}")
    print("[i] Modèles stockés dans:", MODELS_ROOT)
    app.run(host=host, port=port, debug=debug, threaded=True)
