"""Settings > AI Models: environment, memory options, model downloads, and
the Unload / Clear caches buttons."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from vertexcomp.ai.catalog import MODELS
from vertexcomp.ai.client import candidate_pythons, default_models_dir, get_manager

MEMORY_MODES = ["low vram (model offload)", "minimal (sequential offload)", "auto", "full GPU"]
MEMORY_TIPS = {
    "low vram (model offload)": "Only the model part that is running sits on the GPU (~5-7 GB peak at 1024px). "
                                "Little speed loss. Recommended for 8-16 GB cards.",
    "minimal (sequential offload)": "Streams layers to the GPU one by one (~2-3 GB peak). Slow, for 4-6 GB cards.",
    "auto": "Full GPU when the card has >= 15 GB and enough free memory, otherwise model offload.",
    "full GPU": "Everything on the GPU (~10-12 GB). Fastest, needs a big card.",
}


class AISettingsPage(QWidget):
    def __init__(self, settings, on_change=None):
        super().__init__()
        self.st = settings
        self.mgr = get_manager(settings)
        self.on_change = on_change
        self.rows: dict[tuple, QLabel] = {}
        lay = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # Python environment
        self.py = QComboBox()
        self.py.setEditable(True)
        cands = list(dict.fromkeys(candidate_pythons()))
        self.py.addItems(cands)
        self.py.setCurrentText(self.st.get("ai", "python") or "")
        self.py.lineEdit().setPlaceholderText(f"auto: {cands[0]}")
        self.py.lineEdit().editingFinished.connect(lambda: self._set("python", self.py.currentText().strip()))
        self.py.activated.connect(lambda _: self._set("python", self.py.currentText().strip()))
        pyrow = QHBoxLayout()
        pyrow.addWidget(self.py, 1)
        b = QPushButton("...")
        b.setFixedWidth(28)
        b.clicked.connect(self._browse_python)
        chk = QPushButton("Check")
        chk.clicked.connect(self._check_env)
        pyrow.addWidget(b)
        pyrow.addWidget(chk)
        self._row(form, "python", "AI Python (torch + diffusers)", pyrow)
        self.env = QLabel("Press Check to test the AI environment.")
        self.env.setWordWrap(True)
        self.env.setObjectName("knobLabel")
        form.addRow("", self.env)
        root = Path(sys.prefix).parent
        bat = root / "install_ai_windows.bat"
        if sys.platform.startswith("win") and bat.exists():
            inst = QPushButton("Install AI environment (.venv-ai)...")
            inst.setToolTip(str(bat))
            inst.clicked.connect(lambda: subprocess.Popen(["cmd", "/c", "start", "", str(bat)], cwd=str(root)))
            form.addRow("", inst)

        # Models folder
        self.mdir = QLineEdit(self.st.get("ai", "models_dir") or "")
        self.mdir.setPlaceholderText(str(default_models_dir()))
        self.mdir.editingFinished.connect(lambda: (self._set("models_dir", self.mdir.text().strip()), self.refresh()))
        drow = QHBoxLayout()
        drow.addWidget(self.mdir, 1)
        b2 = QPushButton("...")
        b2.setFixedWidth(28)
        b2.clicked.connect(self._browse_dir)
        op = QPushButton("Open")
        op.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._ensure_dir()))))
        drow.addWidget(b2)
        drow.addWidget(op)
        self._row(form, "models_dir", "models folder", drow)

        self.mem = QComboBox()
        self.mem.addItems(MEMORY_MODES)
        self.mem.setCurrentText(self.st.get("ai", "memory_mode"))
        self.mem.setToolTip("\n".join(f"{k}: {v}" for k, v in MEMORY_TIPS.items()))
        self.mem.currentTextChanged.connect(lambda v: self._set("memory_mode", v, restart=True))
        self._row(form, "memory_mode", "GPU memory mode", self.mem)
        self.mem_tip = QLabel(MEMORY_TIPS.get(self.mem.currentText(), ""))
        self.mem_tip.setWordWrap(True)
        self.mem_tip.setObjectName("knobLabel")
        self.mem.currentTextChanged.connect(lambda v: self.mem_tip.setText(MEMORY_TIPS.get(v, "")))
        form.addRow("", self.mem_tip)

        self.prec = QComboBox()
        self.prec.addItems(["fp16", "bf16"])
        self.prec.setCurrentText(self.st.get("ai", "precision"))
        self.prec.currentTextChanged.connect(lambda v: self._set("precision", v, restart=True))
        self._row(form, "precision", "precision", self.prec)

        self.tiling = QCheckBox()
        self.tiling.setChecked(bool(self.st.get("ai", "vae_tiling")))
        self.tiling.toggled.connect(lambda v: self._set("vae_tiling", v, restart=True))
        self._row(form, "vae_tiling", "VAE tiling / slicing (less VRAM)", self.tiling)

        self.idle = QSpinBox()
        self.idle.setRange(0, 600)
        self.idle.setValue(int(self.st.get("ai", "idle_unload_minutes") or 0))
        self.idle.valueChanged.connect(lambda v: self._set("idle_unload_minutes", v))
        self._row(form, "idle_unload_minutes", "unload after idle (min, 0 = never)", self.idle)

        self.tok = QLineEdit(self.st.get("ai", "hf_token") or "")
        self.tok.setEchoMode(QLineEdit.EchoMode.Password)
        self.tok.setPlaceholderText("only needed for gated models")
        self.tok.editingFinished.connect(lambda: self._set("hf_token", self.tok.text().strip()))
        self._row(form, "hf_token", "Hugging Face token", self.tok)
        lay.addLayout(form)

        # Models
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["model", "size", "licence", "status", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        lay.addWidget(self.table, 1)

        # Memory actions
        act = QHBoxLayout()
        self.state = QLabel("")
        self.state.setObjectName("knobLabel")
        unload = QPushButton("Unload models from RAM")
        unload.setToolTip("Stops the AI worker process: frees all RAM and VRAM used by the models. "
                          "They load again on the next Generate.")
        unload.clicked.connect(self._unload)
        clear = QPushButton("Clear caches")
        clear.setToolTip("Clears the render cache, temporary AI files, unused stored images and the GPU cache.")
        clear.clicked.connect(self._clear)
        act.addWidget(self.state, 1)
        act.addWidget(unload)
        act.addWidget(clear)
        lay.addLayout(act)

        self.mgr.listeners.append(self._event)
        self.destroyed.connect(lambda *_: self._detach())
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(500)
        self.refresh()

    # ------------------------------------------------------------ helpers
    def _row(self, form, key, label, widget):
        lab = QLabel(label)
        lab.setObjectName("knobLabel")
        if isinstance(widget, QHBoxLayout):
            w = QWidget()
            widget.setContentsMargins(0, 0, 0, 0)
            w.setLayout(widget)
            widget = w
        form.addRow(lab, widget)
        self.rows[("ai", key)] = lab

    def _set(self, key, value, restart=False):
        self.st.set("ai", key, value)
        self.st.save()
        if restart and self.mgr.running():
            self.mgr.stop()  # next Generate starts with the new options
        if self.on_change:
            self.on_change("ai", key)

    def _detach(self):
        try:
            self.mgr.listeners.remove(self._event)
        except ValueError:
            pass

    def _event(self, kind, payload):
        self.refresh()

    def _ensure_dir(self) -> Path:
        d = self.mgr.models_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Models folder", str(self.mgr.models_dir()))
        if d:
            self.mdir.setText(d)
            self._set("models_dir", d)
            self.refresh()

    def _browse_python(self):
        filt = "python.exe (python.exe)" if sys.platform.startswith("win") else "python (python*)"
        p, _ = QFileDialog.getOpenFileName(self, "Python with torch + diffusers", "", filt)
        if p:
            self.py.setCurrentText(p)
            self._set("python", p, restart=True)

    def _check_env(self):
        self.env.setText("checking (the first import of torch can take a while)...")
        self.env.repaint()
        r = self.mgr.check_environment(self.py.currentText().strip() or None)
        if "error" in r:
            self.env.setText(f"<span style='color:#d06a5f'>{r['error']}</span>")
            return
        parts = [f"Python {r.get('python')}"]
        parts.append(f"torch {r['torch']}" if r.get("torch") else
                     f"<span style='color:#d06a5f'>torch missing: {r.get('torch_error', '')}</span>")
        parts.append(f"diffusers {r['diffusers']}" if r.get("diffusers") else
                     "<span style='color:#d06a5f'>diffusers missing</span>")
        if r.get("torch"):
            parts.append(f"GPU {r['gpu']} ({r['vram_gb']} GB)" if r.get("cuda") else
                         "<span style='color:#e8a33c'>no CUDA GPU: generation runs on the CPU (slow)</span>")
        if not r.get("peft"):
            parts.append("<span style='color:#e8a33c'>peft missing (needed for LoRAs)</span>")
        self.env.setText(" | ".join(parts))

    def _delete(self, mid):
        from PySide6.QtWidgets import QMessageBox
        m = MODELS[mid]
        r = QMessageBox.question(self, "Delete model", f"Delete {m.label} ({m.size_mb / 1024:.2f} GB) from\n"
                                                       f"{self.mgr.model_dir(mid)}?")
        if r == QMessageBox.StandardButton.Yes:
            if self.mgr.running():
                self.mgr.stop()
            self.mgr.delete_model(mid)

    def _unload(self):
        self.mgr.unload()
        self.state.setText("models unloaded")

    def _clear(self):
        self.state.setText(self.mgr.clear_caches())

    # ------------------------------------------------------------ table
    def refresh(self):
        ids = list(MODELS)
        self.table.setRowCount(len(ids))
        self._bars = {}
        for r, mid in enumerate(ids):
            m = MODELS[mid]
            name = QTableWidgetItem(m.label)
            name.setToolTip(f"{m.repo}\n{m.note}")
            self.table.setItem(r, 0, name)
            self.table.setItem(r, 1, QTableWidgetItem(f"{m.size_mb / 1024:.2f} GB" if m.size_mb >= 100 else
                                                      f"{m.size_mb} MB"))
            lic = QTableWidgetItem(m.license.split(" (")[0])
            lic.setToolTip(m.license)
            self.table.setItem(r, 2, lic)
            dl = self.mgr.downloads.get(mid)
            if dl and not dl.get("finished"):
                bar = QProgressBar()
                bar.setMaximum(1000)
                self._bars[mid] = bar
                self.table.setCellWidget(r, 3, bar)
                btn = QPushButton("Cancel")
                btn.clicked.connect(lambda _=False, i=mid: self.mgr.cancel_download(i))
            elif self.mgr.installed(mid):
                self.table.removeCellWidget(r, 3)
                self.table.setItem(r, 3, QTableWidgetItem("installed"))
                btn = QPushButton("Delete")
                btn.clicked.connect(lambda _=False, i=mid: self._delete(i))
            else:
                self.table.removeCellWidget(r, 3)
                err = dl.get("error") if dl else ""
                it = QTableWidgetItem(f"failed: {err}" if err else "not downloaded")
                it.setToolTip(err)
                self.table.setItem(r, 3, it)
                btn = QPushButton("Download")
                btn.clicked.connect(lambda _=False, i=mid: (self.mgr.download(i), self.refresh()))
            self.table.setCellWidget(r, 4, btn)
        self.table.resizeColumnsToContents()
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tick()

    def _tick(self):
        for mid, bar in getattr(self, "_bars", {}).items():
            dl = self.mgr.downloads.get(mid)
            if dl:
                bar.setValue(int(1000 * dl["done"] / max(1, dl["total"])))
                bar.setFormat(f"{dl['done'] / 2**30:.2f} / {dl['total'] / 2**30:.2f} GB")
        running = self.mgr.running()
        info = self.mgr.env_info
        txt = "AI worker: " + ("running" if running else "stopped")
        if running and info.get("gpu"):
            txt += f" on {info['gpu']}"
        if not self.state.text() or self.state.text().startswith("AI worker"):
            self.state.setText(txt)
