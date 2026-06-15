"""
Factory Vision Control Dashboard
A modern futuristic industrial PyQt5 dashboard for factory control rooms.
Requirements: pip install PyQt5 opencv-python
"""

import sys
import time
from datetime import datetime, timedelta
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QSizePolicy, QGraphicsDropShadowEffect
)
from PyQt5.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation,
    QEasingCurve, QRect, pyqtProperty, QSequentialAnimationGroup,
    QParallelAnimationGroup
)
from PyQt5.QtGui import (
    QColor, QPainter, QPen, QBrush, QFont, QFontDatabase,
    QLinearGradient, QRadialGradient, QPainterPath, QPixmap,
    QImage
)
import math
import random


# ─── Color Palette ────────────────────────────────────────────────────────────
# Backgrounds
BG_DEEP        = "#040917"   # Main background
BG_SECONDARY   = "#0A1022"   # Secondary background
BG_CARD        = "#162B4F"   # Card background
BG_CARD_HOVER  = "#1D3660"   # Card hover (slightly lighter than card)
BG_DARK_PANEL  = "#101A33"   # Dark panel
BG_SIDEBAR     = "#07101F"   # Sidebar / action bar

# Neon Blue Accents
ACCENT_BLUE    = "#00A8FF"   # Primary neon blue
ACCENT_CYAN    = "#00D4FF"   # Bright cyan blue (video glow)
ELECTRIC_BLUE  = "#1E90FF"   # Electric blue glow
BORDER_BLUE    = "#162BFF"   # Dashboard border blue
CARD_BLUE      = "#3B82F6"   # Card highlight blue

# Success / Live Status
LIVE_GREEN     = "#20DB14"   # Live indicator green
SUCCESS_GREEN  = "#209D24"   # Success green
BTN_START      = "#00C853"   # Start button green
NEON_GREEN     = "#00FF88"   # Bright neon green accent

# Warning / Overlap
ORANGE_ACCENT  = "#9B6611"   # Orange accent
AMBER_GLOW     = "#FFB300"   # Amber glow (overlap card accent)
GOLD_HIGH      = "#FFC107"   # Gold highlight

# Stop / Danger
BTN_STOP       = "#FF4D4F"   # Stop button red
DARK_RED       = "#D32F2F"   # Dark red
NEON_RED       = "#FF1744"   # Neon red glow

# Text
TEXT_PRIMARY   = "#E8ECF0"   # Primary text
TEXT_SECONDARY = "#A3B4CB"   # Secondary text
TEXT_MUTED     = "#9CA2A8"   # Muted text
TEXT_DISABLED  = "#76889A"   # Disabled text

# Industrial Box Colors (for bounding box overlays)
BOX_BROWN      = "#674630"   # Box brown
BOX_DARK_BROWN = "#583226"   # Dark box brown

# Convenience aliases kept for backward-compat with widget code
ACCENT_GREEN   = NEON_GREEN
ACCENT_AMBER   = AMBER_GLOW
ACCENT_RED     = BTN_STOP
BORDER_DIM     = "#1A2B4A"
BORDER_GLOW    = "#1E3A6F"


# ─── Utility: drop-shadow helper ──────────────────────────────────────────────
def make_shadow(color: str, blur: int = 24, x: int = 0, y: int = 4) -> QGraphicsDropShadowEffect:
    eff = QGraphicsDropShadowEffect()
    eff.setBlurRadius(blur)
    eff.setOffset(x, y)
    eff.setColor(QColor(color))
    return eff


# ─── Fake Video Frame Generator (replace with real OpenCV capture) ─────────────
class FakeVideoThread(QThread):
    """Generates synthetic factory-camera frames for demo purposes.
       Swap this class body with real cv2.VideoCapture logic."""
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._running = False
        self._t = 0.0

    def run(self):
        self._running = True
        while self._running:
            img = self._generate_frame()
            self.frame_ready.emit(img)
            self.msleep(33)   # ~30 fps

    def stop(self):
        self._running = False
        self.wait()

    def _generate_frame(self) -> QImage:
        """Draw a synthetic industrial camera view."""
        W, H = 640, 640
        img = QImage(W, H, QImage.Format_RGB888)
        img.fill(QColor(BG_DARK_PANEL))

        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        self._t += 0.04

        # Conveyor belt lines — dark panel grid
        for i in range(0, W, 40):
            alpha = int(100 + 70 * math.sin(self._t + i * 0.1))
            c = QColor(22, 43, 79, alpha)   # BG_CARD tint
            p.setPen(QPen(c, 1))
            offset = int((self._t * 30) % 40)
            p.drawLine(i + offset, 0, i + offset, H)

        # Horizontal scan lines — electric blue
        for j in range(0, H, 80):
            alpha = int(50 + 35 * math.sin(self._t * 0.5 + j * 0.02))
            p.setPen(QPen(QColor(0, 168, 255, alpha), 1))   # ACCENT_BLUE
            y = int(j + (self._t * 15) % 80)
            if y < H:
                p.drawLine(0, y, W, y)

        # Tracked objects (boxes) — industrial palette
        objects = [
            (200 + int(60 * math.sin(self._t * 0.7)), 180, 90, 90, LIVE_GREEN),
            (420 + int(30 * math.cos(self._t * 0.5)), 320, 80, 100, ACCENT_BLUE),
            (150 + int(20 * math.sin(self._t * 1.1)), 430, 70, 70, AMBER_GLOW),
            (350 + int(40 * math.cos(self._t * 0.9)), 200, 100, 85, LIVE_GREEN),
            (480 + int(15 * math.sin(self._t * 0.6)), 460, 65, 65, BTN_STOP),
        ]

        for (x, y, w, h, color) in objects:
            # Glow aura
            glow = QColor(color)
            glow.setAlpha(30)
            p.setPen(Qt.NoPen)
            p.setBrush(glow)
            p.drawRoundedRect(x - 6, y - 6, w + 12, h + 12, 6, 6)

            # Box border
            box_color = QColor(color)
            box_color.setAlpha(220)
            pen = QPen(box_color, 2)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRect(x, y, w, h)

            # Corner ticks
            tick = 10
            p.setPen(QPen(QColor(color), 3))
            for (cx, cy, dx, dy) in [
                (x, y, 1, 1), (x+w, y, -1, 1),
                (x, y+h, 1, -1), (x+w, y+h, -1, -1)
            ]:
                p.drawLine(cx, cy, cx + dx * tick, cy)
                p.drawLine(cx, cy, cx, cy + dy * tick)

            # Label chip
            label_color = QColor(color)
            label_color.setAlpha(180)
            p.setBrush(label_color)
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(x, y - 22, 52, 18, 4, 4)
            p.setPen(QColor("#000000"))
            p.setFont(QFont("Consolas", 8, QFont.Bold))
            p.drawText(x + 4, y - 8, "ID:" + str(objects.index((x, y, w, h, color)) + 1))

        # Crosshair overlay — electric blue
        cx, cy = W // 2, H // 2
        p.setPen(QPen(QColor(30, 144, 255, 55), 1))   # ELECTRIC_BLUE
        p.drawLine(0, cy, W, cy)
        p.drawLine(cx, 0, cx, H)
        p.drawEllipse(cx - 40, cy - 40, 80, 80)

        # Resolution watermark
        p.setPen(QColor(TEXT_SECONDARY))
        p.setFont(QFont("Consolas", 8))
        p.drawText(8, H - 8, "CAM-01 | 640×640 | 30fps")

        p.end()
        return img


# ─── Video Display Widget ──────────────────────────────────────────────────────
class VideoWidget(QWidget):
    def __init__(self):
        super().__init__()
        self._pixmap = None
        self._glow_alpha = 180
        self._glow_dir = -2
        self.setMinimumSize(640, 640)

        # Glow pulse timer
        self._glow_timer = QTimer(self)
        self._glow_timer.timeout.connect(self._pulse_glow)
        self._glow_timer.start(40)

    def _pulse_glow(self):
        self._glow_alpha += self._glow_dir
        if self._glow_alpha <= 80 or self._glow_alpha >= 220:
            self._glow_dir *= -1
        self.update()

    def set_frame(self, img: QImage):
        self._pixmap = QPixmap.fromImage(img)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        W, H = self.width(), self.height()
        radius = 24

        # Outer glow halo — cyan blue
        for i in range(8, 0, -1):
            alpha = int(self._glow_alpha * (i / 8) * 0.22)
            pen = QPen(QColor(0, 212, 255, alpha), i * 2)   # ACCENT_CYAN
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W - i * 2, H - i * 2, radius + i, radius + i)

        # Clip inner area
        path = QPainterPath()
        path.addRoundedRect(6, 6, W - 12, H - 12, radius, radius)
        p.setClipPath(path)

        if self._pixmap:
            scaled = self._pixmap.scaled(W - 12, H - 12,
                                         Qt.KeepAspectRatioByExpanding,
                                         Qt.SmoothTransformation)
            x_off = (scaled.width() - (W - 12)) // 2
            y_off = (scaled.height() - (H - 12)) // 2
            p.drawPixmap(6 - x_off, 6 - y_off, scaled)
        else:
            # Placeholder
            p.fillRect(6, 6, W - 12, H - 12, QColor(BG_DARK_PANEL))
            p.setPen(QColor(TEXT_SECONDARY))
            p.setFont(QFont("Inter", 14))
            p.drawText(self.rect(), Qt.AlignCenter, "No Signal")

        p.setClipping(False)

        # Inner border ring — cyan blue
        pen = QPen(QColor(0, 212, 255, self._glow_alpha), 2)   # ACCENT_CYAN
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(6, 6, W - 12, H - 12, radius, radius)

        p.end()


# ─── Live Badge ───────────────────────────────────────────────────────────────
class LiveBadge(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(90, 28)
        self._dot_alpha = 255
        self._dot_dir = -8
        t = QTimer(self)
        t.timeout.connect(self._blink)
        t.start(50)

    def _blink(self):
        self._dot_alpha += self._dot_dir
        if self._dot_alpha <= 60 or self._dot_alpha >= 255:
            self._dot_dir *= -1
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        # Pill background
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(32, 219, 20, 28))   # LIVE_GREEN tint
        p.drawRoundedRect(0, 0, self.width(), self.height(), 14, 14)
        # Dot
        dot_color = QColor(LIVE_GREEN)
        dot_color.setAlpha(self._dot_alpha)
        p.setBrush(dot_color)
        p.drawEllipse(10, 9, 10, 10)
        # Text
        p.setPen(QColor(LIVE_GREEN))
        p.setFont(QFont("Inter", 9, QFont.Bold))
        p.drawText(28, 19, "● LIVE")
        p.end()


# ─── KPI Card ─────────────────────────────────────────────────────────────────
class KPICard(QWidget):
    def __init__(self, title: str, accent: str, parent=None):
        super().__init__(parent)
        self._accent = accent
        self._hover = False
        self._hover_anim = 0.0
        self.setMouseTracking(True)
        self.setAttribute(Qt.WA_Hover)

        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(20, 16, 20, 16)
        self._layout.setSpacing(6)

        # Title
        self._title_label = QLabel(title.upper())
        self._title_label.setFont(QFont("Inter", 9, 57))
        self._title_label.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        self._layout.addWidget(self._title_label)

        # Value
        self._value_label = QLabel("0")
        self._value_label.setFont(QFont("Inter", 44, QFont.Bold))
        self._value_label.setStyleSheet(f"color: {accent}; background: transparent;")
        self._layout.addWidget(self._value_label)

        # Hover timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate_hover)
        self._anim_timer.start(16)

    def set_value(self, v):
        self._value_label.setText(str(v))

    def _animate_hover(self):
        target = 1.0 if self._hover else 0.0
        self._hover_anim += (target - self._hover_anim) * 0.12
        self.update()

    def enterEvent(self, e):
        self._hover = True

    def leaveEvent(self, e):
        self._hover = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        a = self._hover_anim
        W, H = self.width(), self.height()

        # Card background with glassmorphism
        bg = QColor(BG_CARD)
        hover_bg = QColor(BG_CARD_HOVER)
        r = int(bg.red()   + (hover_bg.red()   - bg.red())   * a)
        g = int(bg.green() + (hover_bg.green() - bg.green()) * a)
        b = int(bg.blue()  + (hover_bg.blue()  - bg.blue())  * a)

        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        # Accent glow border
        accent = QColor(self._accent)
        border_alpha = int(60 + 120 * a)
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), border_alpha), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        # Top accent bar
        bar_alpha = int(80 + 120 * a)
        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), bar_alpha))
        grad.setColorAt(1, QColor(accent.red(), accent.green(), accent.blue(), 0))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)

        # Elevation shadow on hover
        if a > 0.05:
            shadow = QRadialGradient(W / 2, H + 10, W * 0.6)
            shadow.setColorAt(0, QColor(accent.red(), accent.green(), accent.blue(), int(30 * a)))
            shadow.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(shadow)
            p.drawEllipse(-W // 4, H - 10, W + W // 2, 40)

        p.end()
        super().paintEvent(event)


# ─── Session Info Card ────────────────────────────────────────────────────────
class SessionCard(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._hover = False
        self._hover_anim = 0.0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)

        title = QLabel("SESSION INFO")
        title.setFont(QFont("Inter", 9, 57))
        title.setStyleSheet(f"color: {TEXT_SECONDARY}; letter-spacing: 2px; background: transparent;")
        layout.addWidget(title)

        self._rows: dict[str, QLabel] = {}
        for key, default in [
            ("Start Time", "--:--:--"),
            ("Elapsed", "00:00:00"),
            ("Status", "Idle"),
        ]:
            row = QHBoxLayout()
            row.setSpacing(8)
            k_lbl = QLabel(key)
            k_lbl.setFont(QFont("Inter", 10))
            k_lbl.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
            v_lbl = QLabel(default)
            v_lbl.setFont(QFont("Inter", 10, QFont.DemiBold))
            v_lbl.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent;")
            v_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            row.addWidget(k_lbl)
            row.addStretch()
            row.addWidget(v_lbl)
            layout.addLayout(row)
            self._rows[key] = v_lbl

        layout.addStretch()

        # Hover
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._animate)
        self._anim_timer.start(16)

    def set_value(self, key: str, val: str):
        if key in self._rows:
            self._rows[key].setText(val)

    def set_status(self, running: bool):
        if running:
            self._rows["Status"].setText("● Running")
            self._rows["Status"].setStyleSheet(f"color: {ACCENT_GREEN}; background: transparent; font-weight: 600;")
        else:
            self._rows["Status"].setText("● Idle")
            self._rows["Status"].setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; font-weight: 600;")

    def _animate(self):
        target = 1.0 if self._hover else 0.0
        self._hover_anim += (target - self._hover_anim) * 0.12
        self.update()

    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        a = self._hover_anim
        W, H = self.width(), self.height()

        bg = QColor(BG_CARD)
        hov = QColor(BG_CARD_HOVER)
        r = int(bg.red()   + (hov.red()   - bg.red())   * a)
        g = int(bg.green() + (hov.green() - bg.green()) * a)
        b = int(bg.blue()  + (hov.blue()  - bg.blue())  * a)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(r, g, b))
        p.drawRoundedRect(0, 0, W, H, 16, 16)

        border_alpha = int(40 + 80 * a)
        p.setPen(QPen(QColor(59, 130, 246, border_alpha), 1))   # CARD_BLUE
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 15, 15)

        # Top bar — card highlight blue
        grad = QLinearGradient(0, 0, W, 0)
        grad.setColorAt(0, QColor(59, 130, 246, int(70 + 90 * a)))   # CARD_BLUE
        grad.setColorAt(1, QColor(59, 130, 246, 0))
        p.setPen(Qt.NoPen)
        p.setBrush(grad)
        p.drawRoundedRect(0, 0, W, 3, 2, 2)
        p.end()
        super().paintEvent(event)


# ─── Glow Button ──────────────────────────────────────────────────────────────
class GlowButton(QPushButton):
    def __init__(self, text: str, accent: str, parent=None):
        super().__init__(text, parent)
        self._accent = accent
        self._hover_anim = 0.0
        self._pulse_anim = 0.0
        self._is_active = False
        self.setFixedHeight(52)
        self.setCursor(Qt.PointingHandCursor)
        self.setFont(QFont("Inter", 12, QFont.DemiBold))

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick)
        self._anim_timer.start(16)
        self._pulse_t = 0.0

    def set_active(self, v: bool):
        self._is_active = v

    def _tick(self):
        target_hover = 1.0 if self.underMouse() else 0.0
        self._hover_anim += (target_hover - self._hover_anim) * 0.15

        if self._is_active:
            self._pulse_t += 0.05
            self._pulse_anim = 0.5 + 0.5 * math.sin(self._pulse_t)
        else:
            self._pulse_anim = 0.0
            self._pulse_t = 0.0

        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        W, H = self.width(), self.height()
        a = self._hover_anim
        pulse = self._pulse_anim
        accent = QColor(self._accent)

        # Outer pulse ring
        if pulse > 0:
            ring_alpha = int(60 * pulse)
            spread = int(8 * pulse)
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), ring_alpha), 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(-spread, -spread, W + spread * 2, H + spread * 2, 14 + spread, 14 + spread)

        # Glow halo
        for i in range(6, 0, -1):
            alpha = int((a * 0.6 + pulse * 0.4) * 35 * (i / 6))
            p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), alpha), i * 2))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(i, i, W - i * 2, H - i * 2, 12, 12)

        # Button fill
        fill_alpha = int(30 + 60 * a + 30 * pulse)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(accent.red(), accent.green(), accent.blue(), fill_alpha))
        p.drawRoundedRect(0, 0, W, H, 12, 12)

        # Border
        border_alpha = int(140 + 115 * a)
        p.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), border_alpha), 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(1, 1, W - 2, H - 2, 11, 11)

        # Label
        text_alpha = int(180 + 75 * a)
        p.setPen(QColor(accent.red(), accent.green(), accent.blue(), text_alpha))
        p.setFont(QFont("Inter", 12, QFont.DemiBold))
        p.drawText(self.rect(), Qt.AlignCenter, self.text())
        p.end()


# ─── Main Window ──────────────────────────────────────────────────────────────
class FactoryDashboard(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factory Vision · Control Dashboard")
        self.setMinimumSize(1280, 780)
        self.resize(1920, 1080)

        self._counting = False
        self._start_time: datetime | None = None
        self._total_count = 0
        self._overlap_count = 0

        self._setup_ui()
        self._setup_timers()
        self._video_thread = FakeVideoThread()
        self._video_thread.frame_ready.connect(self._on_frame)

    # ── UI Construction ────────────────────────────────────────────────────────
    def _setup_ui(self):
        # Root
        root = QWidget()
        root.setStyleSheet(f"background: {BG_DEEP};")
        self.setCentralWidget(root)

        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ── Header ────────────────────────────────────────────────────────────
        header = self._build_header()
        outer.addWidget(header)

        # ── Content row ───────────────────────────────────────────────────────
        content = QWidget()
        content.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(content)
        content_layout.setContentsMargins(28, 16, 28, 0)
        content_layout.setSpacing(24)

        # Left — video
        left = self._build_left_panel()
        content_layout.addWidget(left, 70)

        # Right — KPI cards
        right = self._build_right_panel()
        content_layout.addWidget(right, 30)

        outer.addWidget(content, 1)

        # ── Action bar ────────────────────────────────────────────────────────
        action_bar = self._build_action_bar()
        outer.addWidget(action_bar)

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setFixedHeight(64)
        header.setStyleSheet(f"""
            background: {BG_CARD};
            border-bottom: 1px solid {BORDER_DIM};
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(28, 0, 28, 0)

        # Logo + title
        logo_dot = QLabel("◈")
        logo_dot.setFont(QFont("Inter", 18))
        logo_dot.setStyleSheet(f"color: {ACCENT_BLUE}; background: transparent;")

        title = QLabel("FactoryVision")
        title.setFont(QFont("Inter", 16, QFont.Bold))
        title.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; letter-spacing: 1px;")

        subtitle = QLabel("Control Dashboard")
        subtitle.setFont(QFont("Inter", 10))
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")

        layout.addWidget(logo_dot)
        layout.addSpacing(8)
        layout.addWidget(title)
        layout.addSpacing(12)
        layout.addWidget(subtitle)
        layout.addStretch()

        # System time
        self._clock_label = QLabel()
        self._clock_label.setFont(QFont("Inter", 13, 57))
        self._clock_label.setStyleSheet(f"color: {TEXT_PRIMARY}; background: transparent; font-variant: tabular-nums;")
        layout.addWidget(self._clock_label)

        # Status pill
        self._status_pill = QLabel("  SYSTEM READY  ")
        self._status_pill.setFont(QFont("Inter", 9, QFont.Bold))
        self._status_pill.setStyleSheet(f"""
            color: {ACCENT_GREEN};
            background: rgba(0, 255, 136, 0.12);
            border: 1px solid rgba(0, 255, 136, 0.35);
            border-radius: 12px;
            padding: 4px 12px;
            letter-spacing: 1.5px;
        """)
        layout.addSpacing(20)
        layout.addWidget(self._status_pill)

        return header

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        # Row: label + live badge
        top_row = QHBoxLayout()
        cam_label = QLabel("CAM-01  ·  FACTORY FLOOR")
        cam_label.setFont(QFont("Inter", 11, 57))
        cam_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 1px;")
        top_row.addWidget(cam_label)
        top_row.addStretch()
        self._live_badge = LiveBadge()
        top_row.addWidget(self._live_badge)
        layout.addLayout(top_row)

        # Video
        self._video_widget = VideoWidget()
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self._video_widget, 1)

        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        section_label = QLabel("KPI OVERVIEW")
        section_label.setFont(QFont("Inter", 9, 57))
        section_label.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent; letter-spacing: 2px;")
        layout.addWidget(section_label)

        # KPI 1: Total Count
        self._card_total = KPICard("Total Count", ACCENT_BLUE)
        self._card_total.set_value(0)
        layout.addWidget(self._card_total, 1)

        # KPI 2: Overlap Count
        self._card_overlap = KPICard("Overlapped", ACCENT_AMBER)
        self._card_overlap.set_value(0)
        layout.addWidget(self._card_overlap, 1)

        # Session Info
        self._session_card = SessionCard()
        layout.addWidget(self._session_card, 1)

        return panel

    def _build_action_bar(self) -> QWidget:
        bar = QWidget()
        bar.setFixedHeight(88)
        bar.setStyleSheet(f"""
            background: {BG_CARD};
            border-top: 1px solid {BORDER_DIM};
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(28, 16, 28, 16)
        layout.setSpacing(16)

        # Left hint
        hint = QLabel("Select an action to begin object counting session")
        hint.setFont(QFont("Inter", 10))
        hint.setStyleSheet(f"color: {TEXT_SECONDARY}; background: transparent;")
        layout.addWidget(hint)
        layout.addStretch()

        # Start
        self._btn_start = GlowButton("▶  Start Counting", ACCENT_GREEN)
        self._btn_start.setMinimumWidth(200)
        self._btn_start.clicked.connect(self._start_counting)
        layout.addWidget(self._btn_start)

        # Stop
        self._btn_stop = GlowButton("■  Stop Counting", ACCENT_RED)
        self._btn_stop.setMinimumWidth(200)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_counting)
        layout.addWidget(self._btn_stop)

        return bar

    # ── Timers ─────────────────────────────────────────────────────────────────
    def _setup_timers(self):
        # Clock
        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start(500)
        self._update_clock()

        # Session elapsed
        self._session_timer = QTimer(self)
        self._session_timer.timeout.connect(self._update_session)

        # Fake counter increments
        self._counter_timer = QTimer(self)
        self._counter_timer.timeout.connect(self._increment_count)

    # ── Slots ──────────────────────────────────────────────────────────────────
    def _update_clock(self):
        now = datetime.now().strftime("%H:%M:%S")
        self._clock_label.setText(now)

    def _update_session(self):
        if self._start_time:
            elapsed = datetime.now() - self._start_time
            h, rem = divmod(int(elapsed.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            self._session_card.set_value("Elapsed", f"{h:02d}:{m:02d}:{s:02d}")

    def _increment_count(self):
        if not self._counting:
            return
        if random.random() < 0.6:
            self._total_count += random.randint(1, 3)
            self._card_total.set_value(self._total_count)
        if random.random() < 0.25:
            self._overlap_count += 1
            self._card_overlap.set_value(self._overlap_count)

    def _start_counting(self):
        self._counting = True
        self._start_time = datetime.now()
        self._total_count = 0
        self._overlap_count = 0
        self._card_total.set_value(0)
        self._card_overlap.set_value(0)

        self._session_card.set_value("Start Time", self._start_time.strftime("%H:%M:%S"))
        self._session_card.set_value("Elapsed", "00:00:00")
        self._session_card.set_status(True)

        self._btn_start.setEnabled(False)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(True)
        self._btn_stop.set_active(True)

        self._status_pill.setText("  COUNTING ACTIVE  ")
        self._status_pill.setStyleSheet(f"""
            color: {ACCENT_AMBER};
            background: rgba(255, 184, 0, 0.12);
            border: 1px solid rgba(255, 184, 0, 0.35);
            border-radius: 12px;
            padding: 4px 12px;
            letter-spacing: 1.5px;
        """)

        self._session_timer.start(1000)
        self._counter_timer.start(400)
        self._video_thread.start()

    def _stop_counting(self):
        self._counting = False
        self._session_timer.stop()
        self._counter_timer.stop()
        self._video_thread.stop()

        self._btn_start.setEnabled(True)
        self._btn_start.set_active(False)
        self._btn_stop.setEnabled(False)
        self._btn_stop.set_active(False)

        self._session_card.set_status(False)

        self._status_pill.setText("  SYSTEM READY  ")
        self._status_pill.setStyleSheet(f"""
            color: {ACCENT_GREEN};
            background: rgba(0, 255, 136, 0.12);
            border: 1px solid rgba(0, 255, 136, 0.35);
            border-radius: 12px;
            padding: 4px 12px;
            letter-spacing: 1.5px;
        """)

    def _on_frame(self, img: QImage):
        self._video_widget.set_frame(img)

    def closeEvent(self, event):
        self._video_thread.stop()
        super().closeEvent(event)


# ─── Entry Point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Load Inter font if available, else fall back gracefully
    QFontDatabase.addApplicationFont("Inter.ttf")

    # Global palette
    from PyQt5.QtGui import QPalette
    palette = QPalette()
    palette.setColor(QPalette.Window,        QColor(BG_DEEP))
    palette.setColor(QPalette.WindowText,    QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Base,          QColor(BG_CARD))
    palette.setColor(QPalette.AlternateBase, QColor(BG_CARD))
    palette.setColor(QPalette.Text,          QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Button,        QColor(BG_CARD))
    palette.setColor(QPalette.ButtonText,    QColor(TEXT_PRIMARY))
    palette.setColor(QPalette.Highlight,     QColor(ACCENT_BLUE))
    app.setPalette(palette)

    win = FactoryDashboard()
    win.show()
    sys.exit(app.exec_())