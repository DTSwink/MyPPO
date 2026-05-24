"""Top-down Pygame visualizer for the 2D kinematic mini-test."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pygame

from kinematic2d.agent import MLPAgent
from kinematic2d.checkpoint import load_checkpoint, load_loss_history
from kinematic2d.loss_terms import LOSS_TERMS, MAX_LOSS_COEFF, MIN_LOSS_COEFF, normalize_loss_coeffs
from kinematic2d.settings import (
    MAX_AUTOREGRESSIVE_WINDOW,
    MAX_CHECKPOINT_REFRESH_SEC,
    MIN_AUTOREGRESSIVE_WINDOW,
    MIN_CHECKPOINT_REFRESH_SEC,
    SettingsStore,
)
from kinematic2d import viz_shutdown
from kinematic2d.state import (
    Simulation,
    agent_input_finite,
    agent_output_finite,
    limb_state_finite,
)
from kinematic2d.trajectory import TRAJECTORY_SPEED, forward_constant_trajectory
from kinematic2d.transforms import Transform2D, local_to_global, rotation_matrix

if TYPE_CHECKING:
    from kinematic2d.experiment_runner import ExperimentRunner

# Colors
BG_COLOR = (10, 20, 55)
GRID_COLOR = (70, 80, 110)
ROOT_COLOR = (220, 220, 230)
PELVIS_COLOR = (255, 140, 40)
FOOT_LOWER = (220, 60, 60)
FOOT_HIGHER = (60, 200, 90)
UI_TEXT = (210, 215, 230)
BUTTON_BG = (35, 50, 90)
BUTTON_HOVER = (50, 70, 120)
BUTTON_BORDER = (120, 140, 180)
TRAJECTORY_COLOR = (90, 110, 150)
LOSS_PANEL_BG = (18, 28, 62)
LOSS_PANEL_BORDER = (100, 120, 170)
LOSS_LINE = (100, 220, 255)
LOSS_FILL = (40, 90, 140)
SLIDE_LINE = (255, 180, 80)
SLIDE_FILL = (120, 70, 30)
RADIUS_LINE = (190, 140, 255)
RADIUS_FILL = (70, 45, 110)
STRIDE_LINE = (80, 255, 200)
STRIDE_FILL = (30, 100, 80)
LOSS_TERM_CHART_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "pelvis_tilt": (LOSS_LINE, LOSS_FILL),
    "foot_pin": (SLIDE_LINE, SLIDE_FILL),
    "limb_radius": (RADIUS_LINE, RADIUS_FILL),
    "foot_stride": (STRIDE_LINE, STRIDE_FILL),
}
TARGET_HZ = 30
LOSS_WIDGET_W = 404
LOSS_WIDGET_H = 198
LOSS_GRID_COLS = 2
SLIDE_WIDGET_W = 240
SLIDE_WIDGET_H = 130
LOSS_WIDGET_MARGIN = 12
BOTTOM_BAR_H = 48
BUTTON_H = 34
SLIDE_HISTORY_MAX = 600
SIM_STEP_SEC = 1.0 / TARGET_HZ
SETTINGS_PANEL_W = 480
SETTINGS_TITLE_DY = 14
SETTINGS_LABEL_LINE_H = 22
SETTINGS_CTRL_H = 34
SETTINGS_SECTION_GAP = 14
SETTINGS_TERMS_HEADER_H = 20
SETTINGS_REFRESH_LABEL_DY = 36
SETTINGS_REFRESH_ROW_DY = SETTINGS_REFRESH_LABEL_DY + SETTINGS_LABEL_LINE_H + 4
SETTINGS_AR_LABEL_DY = SETTINGS_REFRESH_ROW_DY + SETTINGS_CTRL_H + SETTINGS_SECTION_GAP
SETTINGS_AR_ROW_DY = SETTINGS_AR_LABEL_DY + SETTINGS_LABEL_LINE_H + 4
SETTINGS_TERMS_HEADER_DY = SETTINGS_AR_ROW_DY + SETTINGS_CTRL_H + SETTINGS_SECTION_GAP
SETTINGS_TERMS_ROW_DY = SETTINGS_TERMS_HEADER_DY + SETTINGS_TERMS_HEADER_H + 4
SETTINGS_LOSS_TERM_ROW_H = 52
SETTINGS_FOOTER_H = 56
REFRESH_STEP_SEC = 1.0
COEFF_STEP_FACTOR = 1.25
DEFAULT_PPM = 280.0
MIN_PPM = 60.0
MAX_PPM = 1400.0
ZOOM_STEP = 1.12
# Keep pygame draw coords in a safe int range when limbs diverge far from root.
SCREEN_COORD_CLAMP = 500_000


class Visualizer:
    def __init__(
        self,
        width: int = 960,
        height: int = 720,
        pixels_per_meter: float = DEFAULT_PPM,
        seed: int | None = None,
        training_mode: bool = False,
        checkpoint_dir: Path | str | None = None,
        experiment_runner: ExperimentRunner | None = None,
    ) -> None:
        pygame.display.init()
        pygame.font.init()
        self.width = width
        self.height = height
        self.ppm = pixels_per_meter
        self._default_ppm = pixels_per_meter
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0
        self.seed = seed

        self.screen = pygame.display.set_mode((width, height))
        title = "MyPPO — Training" if training_mode else "MyPPO — 2D Kinematic Walk Visualizer (top-down)"
        pygame.display.set_caption(title)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_tiny = pygame.font.Font(None, 16)

        self.settings_store = SettingsStore()
        self.show_settings = False
        self._draft_refresh_sec = self.settings_store.checkpoint_refresh_sec()
        self._draft_ar_window = self.settings_store.autoregressive_window()
        self._draft_loss_terms = self.settings_store.loss_terms_enabled()
        self._draft_loss_coeffs = self.settings_store.loss_coeffs()
        self.loss_term_toggle_btns: dict[str, pygame.Rect] = {}
        self.loss_term_coeff_minus_btns: dict[str, pygame.Rect] = {}
        self.loss_term_coeff_plus_btns: dict[str, pygame.Rect] = {}

        self.training_mode = training_mode
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.experiment_runner = experiment_runner
        self._checkpoint_timer_ms = 0
        self._last_metrics_mtime = 0.0
        self._loss_term_histories: dict[str, list[float]] = {
            spec.key: [] for spec in LOSS_TERMS
        }
        self._train_step = 0
        self._train_loss = float("nan")
        self._slide_history: list[float] = []
        self._slide_cm_s = 0.0
        self._slide_pinned_side = "—"

        self.roots = forward_constant_trajectory()
        self.sim = Simulation(roots=self.roots)
        self.agent = MLPAgent(seed=seed)

        viz_shutdown.clear()
        self.running = True
        self._user_requested_exit = False
        self.paused = False
        self.show_trajectory = False
        self._sim_steps_this_second = 0
        self._sim_rate = 0.0
        self._rate_timer_ms = 0

        self.reset_button = pygame.Rect(width // 2 - 60, height - BOTTOM_BAR_H, 120, BUTTON_H)
        self.settings_button = pygame.Rect(width // 2 - 190, height - BOTTOM_BAR_H, 120, BUTTON_H)
        self.new_run_button = pygame.Rect(width // 2 + 70, height - BOTTOM_BAR_H, 120, BUTTON_H)
        self._layout_settings_panel()

        if self.experiment_runner is not None:
            self.experiment_runner.start()

    def _settings_panel_height(self) -> int:
        return SETTINGS_TERMS_ROW_DY + len(LOSS_TERMS) * SETTINGS_LOSS_TERM_ROW_H + SETTINGS_FOOTER_H

    def _layout_settings_panel(self) -> None:
        panel_h = self._settings_panel_height()
        self.settings_panel = pygame.Rect(
            (self.width - SETTINGS_PANEL_W) // 2,
            (self.height - panel_h) // 2,
            SETTINGS_PANEL_W,
            panel_h,
        )
        row_y = self.settings_panel.y + SETTINGS_REFRESH_ROW_DY
        self.refresh_minus_btn = pygame.Rect(self.settings_panel.x + 36, row_y, 44, SETTINGS_CTRL_H)
        self.refresh_plus_btn = pygame.Rect(self.settings_panel.right - 80, row_y, 44, SETTINGS_CTRL_H)

        ar_row_y = self.settings_panel.y + SETTINGS_AR_ROW_DY
        self.ar_window_minus_btn = pygame.Rect(self.settings_panel.x + 36, ar_row_y, 44, SETTINGS_CTRL_H)
        self.ar_window_plus_btn = pygame.Rect(self.settings_panel.right - 80, ar_row_y, 44, SETTINGS_CTRL_H)

        self.loss_term_toggle_btns = {}
        self.loss_term_coeff_minus_btns = {}
        self.loss_term_coeff_plus_btns = {}
        terms_top = self.settings_panel.y + SETTINGS_TERMS_ROW_DY
        for i, spec in enumerate(LOSS_TERMS):
            row_y = terms_top + i * SETTINGS_LOSS_TERM_ROW_H
            self.loss_term_coeff_minus_btns[spec.key] = pygame.Rect(
                self.settings_panel.x + 188,
                row_y + 8,
                28,
                28,
            )
            self.loss_term_coeff_plus_btns[spec.key] = pygame.Rect(
                self.settings_panel.x + 268,
                row_y + 8,
                28,
                28,
            )
            self.loss_term_toggle_btns[spec.key] = pygame.Rect(
                self.settings_panel.right - 72,
                row_y + 8,
                56,
                28,
            )

        btn_y = self.settings_panel.bottom - 44
        self.settings_save_btn = pygame.Rect(self.settings_panel.x + 48, btn_y, 120, 34)
        self.settings_close_btn = pygame.Rect(self.settings_panel.right - 168, btn_y, 120, 34)

    def _checkpoint_poll_ms(self) -> int:
        return self.settings_store.checkpoint_refresh_ms()

    def _draw_button(self, rect: pygame.Rect, label: str, hovered: bool) -> None:
        color = BUTTON_HOVER if hovered else BUTTON_BG
        pygame.draw.rect(self.screen, color, rect, border_radius=6)
        pygame.draw.rect(self.screen, BUTTON_BORDER, rect, 1, border_radius=6)
        surf = self.font.render(label, True, UI_TEXT)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def world_to_screen(self, world_x: float, world_y: float, camera_root: Transform2D) -> tuple[int, int]:
        """Top-down view: translate only, no camera rotation."""
        if not (math.isfinite(world_x) and math.isfinite(world_y)):
            return self.width // 2, self.height // 2
        cam_x = camera_root.x + self._view_offset_x
        cam_y = camera_root.y + self._view_offset_y
        sx = self.width * 0.5 + (world_x - cam_x) * self.ppm
        sy = self.height * 0.5 - (world_y - cam_y) * self.ppm
        if not (math.isfinite(sx) and math.isfinite(sy)):
            return self.width // 2, self.height // 2
        sx = max(-SCREEN_COORD_CLAMP, min(SCREEN_COORD_CLAMP, sx))
        sy = max(-SCREEN_COORD_CLAMP, min(SCREEN_COORD_CLAMP, sy))
        return int(sx), int(sy)

    def _screen_point_visible(self, px: int, py: int, margin: int = 256) -> bool:
        return (
            -margin <= px < self.width + margin
            and -margin <= py < self.height + margin
        )

    def _scroll_zoom_allowed(self, pos: tuple[int, int]) -> bool:
        if self.show_settings:
            return False
        x, y = pos
        if y >= self.height - BOTTOM_BAR_H:
            return False
        bottom = self.height - BOTTOM_BAR_H - LOSS_WIDGET_MARGIN
        if y >= bottom - SLIDE_WIDGET_H:
            if x <= SLIDE_WIDGET_W + LOSS_WIDGET_MARGIN * 2:
                return False
            if self._experiment_active() and x >= self.width - LOSS_WIDGET_W - LOSS_WIDGET_MARGIN * 2:
                return False
        return True

    def _adjust_zoom(self, direction: int, focus: tuple[int, int]) -> None:
        if direction == 0:
            return
        mx, my = focus
        camera_root = self.sim.current_root
        cam_x = camera_root.x + self._view_offset_x
        cam_y = camera_root.y + self._view_offset_y
        world_x = cam_x + (mx - self.width / 2) / self.ppm
        world_y = cam_y - (my - self.height / 2) / self.ppm

        factor = ZOOM_STEP if direction > 0 else 1.0 / ZOOM_STEP
        new_ppm = max(MIN_PPM, min(MAX_PPM, self.ppm * factor))
        if math.isclose(new_ppm, self.ppm):
            return
        self.ppm = new_ppm

        self._view_offset_x = world_x - camera_root.x - (mx - self.width / 2) / self.ppm
        self._view_offset_y = world_y - camera_root.y + (my - self.height / 2) / self.ppm

    def _reset_view(self) -> None:
        self.ppm = self._default_ppm
        self._view_offset_x = 0.0
        self._view_offset_y = 0.0

    def draw_grid(self, camera_root: Transform2D, spacing: float = 0.25) -> None:
        half_w = self.width / self.ppm / 2.0 + spacing
        half_h = self.height / self.ppm / 2.0 + spacing

        origin_x = camera_root.x
        origin_y = camera_root.y

        x_min = origin_x - half_w
        x_max = origin_x + half_w
        y_min = origin_y - half_h
        y_max = origin_y + half_h

        start_x = math.floor(x_min / spacing) * spacing
        start_y = math.floor(y_min / spacing) * spacing

        x = start_x
        while x <= x_max:
            p0 = self.world_to_screen(x, y_min, camera_root)
            p1 = self.world_to_screen(x, y_max, camera_root)
            pygame.draw.line(self.screen, GRID_COLOR, p0, p1, 1)
            x += spacing

        y = start_y
        while y <= y_max:
            p0 = self.world_to_screen(x_min, y, camera_root)
            p1 = self.world_to_screen(x_max, y, camera_root)
            pygame.draw.line(self.screen, GRID_COLOR, p0, p1, 1)
            y += spacing

    def draw_trajectory(self, camera_root: Transform2D) -> None:
        if not self.show_trajectory:
            return
        points = [self.world_to_screen(r.x, r.y, camera_root) for r in self.roots]
        if len(points) >= 2:
            pygame.draw.lines(self.screen, TRAJECTORY_COLOR, False, points, 1)

    def draw_root_cross(self, root: Transform2D, camera_root: Transform2D, size: int = 10) -> None:
        center = self.world_to_screen(root.x, root.y, camera_root)
        rot = rotation_matrix(root.angle)
        for direction in ((1.0, 0.0), (0.0, 1.0)):
            offset = rot @ direction
            tip = self.world_to_screen(
                root.x + offset[0] * size / self.ppm,
                root.y + offset[1] * size / self.ppm,
                camera_root,
            )
            pygame.draw.line(self.screen, ROOT_COLOR, center, tip, 2)

    def draw_oriented_box(
        self,
        tf: Transform2D,
        camera_root: Transform2D,
        half_w: float,
        half_h: float,
        color,
    ) -> None:
        corners_local = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h),
        ]
        rot = rotation_matrix(tf.angle)
        points = []
        for lx, ly in corners_local:
            wx, wy = rot @ [lx, ly]
            points.append(self.world_to_screen(tf.x + wx, tf.y + wy, camera_root))
        if not any(self._screen_point_visible(px, py) for px, py in points):
            return
        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.polygon(self.screen, (30, 30, 40), points, 1)

    def draw_foot_disk(
        self,
        tf: Transform2D,
        camera_root: Transform2D,
        radius_m: float,
        color,
    ) -> None:
        center = self.world_to_screen(tf.x, tf.y, camera_root)
        if not self._screen_point_visible(*center, margin=512):
            return
        radius_px = max(4, int(radius_m * self.ppm))
        pygame.draw.circle(self.screen, color, center, radius_px)
        pygame.draw.circle(self.screen, (30, 30, 40), center, radius_px, 1)

        forward = rotation_matrix(tf.angle) @ [radius_m * 0.8, 0.0]
        tip = self.world_to_screen(tf.x + forward[0], tf.y + forward[1], camera_root)
        pygame.draw.line(self.screen, (40, 40, 50), center, tip, 2)

    def draw_pelvis(self, root: Transform2D, pelvis_local: Transform2D, camera_root: Transform2D) -> None:
        """Draw pelvis with +X as forward (narrow) and +Y as lateral (wide)."""
        pelvis_pos = local_to_global(root, Transform2D(pelvis_local.x, pelvis_local.y, 0.0))
        pelvis_tf = Transform2D(pelvis_pos.x, pelvis_pos.y, root.angle + pelvis_local.angle)
        self.draw_oriented_box(pelvis_tf, camera_root, 0.05, 0.11, PELVIS_COLOR)

        center = self.world_to_screen(pelvis_tf.x, pelvis_tf.y, camera_root)
        if not self._screen_point_visible(*center, margin=512):
            return
        forward = rotation_matrix(pelvis_tf.angle) @ [0.09, 0.0]
        tip = self.world_to_screen(
            pelvis_tf.x + forward[0],
            pelvis_tf.y + forward[1],
            camera_root,
        )
        pygame.draw.line(self.screen, (30, 30, 40), center, tip, 3)

    def draw_ui(self) -> None:
        mouse_pos = pygame.mouse.get_pos()
        self._draw_button(self.settings_button, "Settings", self.settings_button.collidepoint(mouse_pos))
        self._draw_button(self.reset_button, "Reset", self.reset_button.collidepoint(mouse_pos))
        self._draw_button(self.new_run_button, "New Run", self.new_run_button.collidepoint(mouse_pos))

        fps = self.clock.get_fps()
        fps_surf = self.font_small.render(
            f"Render {fps:.0f} FPS  |  Sim {self._sim_rate:.0f} Hz",
            True,
            UI_TEXT,
        )
        self.screen.blit(fps_surf, fps_surf.get_rect(topright=(self.width - 12, 12)))

        status = "PAUSED" if self.paused else ("DONE" if self.sim.finished else "RUNNING")
        help_line = (
            "Space: pause  |  R: reset  |  N: new run  |  T: path  |  S: settings  |  Scroll: zoom"
            if self._experiment_active()
            else "Space: pause  |  R: reset  |  N: start exp  |  T: path  |  S: settings  |  Scroll: zoom"
        )
        lines = [
            f"Frame {self.sim.frame_index} / {len(self.roots) - 1}  [{status}]",
            f"Root speed  {TRAJECTORY_SPEED / SIM_STEP_SEC * 100.0:.1f} cm/s",
            f"Foot h  L={self.sim.current_limbs.foot_left_height:.3f}  R={self.sim.current_limbs.foot_right_height:.3f}",
            help_line,
        ]
        y = 12
        for line in lines:
            surf = self.font_small.render(line, True, UI_TEXT)
            self.screen.blit(surf, (12, y))
            y += 18

    def _draw_chart_panel(
        self,
        rect: pygame.Rect,
        title: str,
        subtitle: str,
        history: list[float],
        line_color: tuple[int, int, int],
        fill_color: tuple[int, int, int],
        y_min: float | None = None,
    ) -> None:
        pygame.draw.rect(self.screen, LOSS_PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, LOSS_PANEL_BORDER, rect, 1, border_radius=8)

        title_surf = self.font_small.render(title, True, UI_TEXT)
        self.screen.blit(title_surf, (rect.x + 10, rect.y + 8))
        sub_surf = self.font_small.render(subtitle, True, UI_TEXT)
        self.screen.blit(sub_surf, (rect.x + 10, rect.y + 26))

        plot = pygame.Rect(rect.x + 10, rect.y + 46, rect.width - 20, rect.height - 56)
        pygame.draw.rect(self.screen, (12, 18, 40), plot, border_radius=4)

        if len(history) < 2:
            return

        values = history
        vmin = min(values) if y_min is None else y_min
        vmax = max(values)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-3

        points: list[tuple[int, int]] = []
        for i, value in enumerate(values):
            t = i / (len(values) - 1)
            px = plot.x + int(t * (plot.width - 1))
            norm = (value - vmin) / (vmax - vmin)
            py = plot.bottom - 1 - int(norm * (plot.height - 1))
            points.append((px, py))

        if len(points) >= 2:
            fill_points = points + [(points[-1][0], plot.bottom), (points[0][0], plot.bottom)]
            pygame.draw.polygon(self.screen, fill_color, fill_points)
            pygame.draw.lines(self.screen, line_color, False, points, 2)

    def _draw_sparkline(
        self,
        plot: pygame.Rect,
        history: list[float],
        line_color: tuple[int, int, int],
        fill_color: tuple[int, int, int],
        y_min: float | None = None,
    ) -> None:
        pygame.draw.rect(self.screen, (12, 18, 40), plot, border_radius=3)
        if len(history) < 2:
            return

        vmin = min(history) if y_min is None else y_min
        vmax = max(history)
        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1e-3

        points: list[tuple[int, int]] = []
        for i, value in enumerate(history):
            t = i / (len(history) - 1)
            px = plot.x + int(t * max(plot.width - 1, 0))
            norm = (value - vmin) / (vmax - vmin)
            py = plot.bottom - 1 - int(norm * max(plot.height - 1, 0))
            points.append((px, py))

        if len(points) >= 2:
            fill_points = points + [(points[-1][0], plot.bottom), (points[0][0], plot.bottom)]
            pygame.draw.polygon(self.screen, fill_color, fill_points)
            pygame.draw.lines(self.screen, line_color, False, points, 1)

    def _bottom_widget_y(self, widget_h: int) -> int:
        """Place chart widgets above the bottom button row."""
        return self.height - widget_h - BOTTOM_BAR_H - LOSS_WIDGET_MARGIN

    def draw_loss_widget(self) -> None:
        rect = pygame.Rect(
            self.width - LOSS_WIDGET_W - LOSS_WIDGET_MARGIN,
            self._bottom_widget_y(LOSS_WIDGET_H),
            LOSS_WIDGET_W,
            LOSS_WIDGET_H,
        )
        pygame.draw.rect(self.screen, LOSS_PANEL_BG, rect, border_radius=8)
        pygame.draw.rect(self.screen, LOSS_PANEL_BORDER, rect, 1, border_radius=8)

        if not math.isnan(self._train_loss):
            header = f"step {self._train_step}  total {self._train_loss:.4f}"
        elif self._experiment_active():
            header = "Starting trainer..."
        else:
            header = ""
        if header:
            header_surf = self.font_tiny.render(header, True, UI_TEXT)
            self.screen.blit(header_surf, (rect.x + 10, rect.y + 8))

        grid = pygame.Rect(rect.x + 8, rect.y + 26, rect.width - 16, rect.height - 34)
        active_terms = self._enabled_loss_terms()
        if not active_terms:
            empty = self.font_tiny.render("No active loss terms", True, UI_TEXT)
            self.screen.blit(empty, (grid.x + 8, grid.y + 8))
            return

        grid_cols = 1 if len(active_terms) == 1 else LOSS_GRID_COLS
        grid_rows = max(1, (len(active_terms) + grid_cols - 1) // grid_cols)
        cell_w = grid.width // grid_cols
        cell_h = grid.height // grid_rows
        pad = 4

        for idx, spec in enumerate(active_terms):
            col = idx % grid_cols
            row = idx // grid_cols
            cell = pygame.Rect(
                grid.x + col * cell_w + pad,
                grid.y + row * cell_h + pad,
                cell_w - pad * 2,
                cell_h - pad * 2,
            )
            history = self._loss_term_histories.get(spec.key, [])
            current = history[-1] if history else float("nan")
            current_text = f"{current:.4f}" if math.isfinite(current) else "—"
            label_surf = self.font_tiny.render(f"{spec.label}  {current_text}", True, UI_TEXT)
            self.screen.blit(label_surf, (cell.x + 2, cell.y + 1))
            plot = pygame.Rect(cell.x + 2, cell.y + 16, cell.width - 4, cell.height - 18)
            line_color, fill_color = LOSS_TERM_CHART_COLORS.get(
                spec.key,
                (LOSS_LINE, LOSS_FILL),
            )
            self._draw_sparkline(plot, history, line_color, fill_color, y_min=0.0)

    def draw_slide_widget(self) -> None:
        rect = pygame.Rect(
            LOSS_WIDGET_MARGIN,
            self._bottom_widget_y(SLIDE_WIDGET_H),
            SLIDE_WIDGET_W,
            SLIDE_WIDGET_H,
        )
        subtitle = f"pinned {self._slide_pinned_side}  now {self._slide_cm_s:.1f} cm/s"
        self._draw_chart_panel(
            rect,
            "Pinned foot slide",
            subtitle,
            self._slide_history,
            SLIDE_LINE,
            SLIDE_FILL,
            y_min=0.0,
        )

    def _record_pinned_foot_slide(self, output: dict) -> None:
        next_root = self.sim.next_root
        if next_root is None:
            return

        left_h = float(output["foot_left_height"])
        right_h = float(output["foot_right_height"])
        left_pinned = left_h <= right_h
        self._slide_pinned_side = "L" if left_pinned else "R"

        root = self.sim.current_root
        if left_pinned:
            g_before = local_to_global(root, self.sim.current_limbs.foot_left)
            g_after = local_to_global(
                next_root,
                Transform2D.from_array(output["future_limbs"][1]),
            )
        else:
            g_before = local_to_global(root, self.sim.current_limbs.foot_right)
            g_after = local_to_global(
                next_root,
                Transform2D.from_array(output["future_limbs"][2]),
            )

        disp_m = math.hypot(g_after.x - g_before.x, g_after.y - g_before.y)
        if not math.isfinite(disp_m):
            return

        speed_cm_s = disp_m / SIM_STEP_SEC * 100.0
        self._slide_cm_s = speed_cm_s
        self._slide_history.append(speed_cm_s)
        if len(self._slide_history) > SLIDE_HISTORY_MAX:
            self._slide_history = self._slide_history[-SLIDE_HISTORY_MAX:]

    def _reset_slide_history(self) -> None:
        self._slide_history = []
        self._slide_cm_s = 0.0
        self._slide_pinned_side = "—"

    def render_frame(self) -> None:
        camera_root = self.sim.current_root
        self.screen.fill(BG_COLOR)
        self.draw_grid(camera_root)
        self.draw_trajectory(camera_root)

        limbs_global = self.sim.current_limbs.limbs_global(camera_root)

        left_h = self.sim.current_limbs.foot_left_height
        right_h = self.sim.current_limbs.foot_right_height
        if left_h <= right_h:
            left_color, right_color = FOOT_LOWER, FOOT_HIGHER
        else:
            left_color, right_color = FOOT_HIGHER, FOOT_LOWER

        self.draw_foot_disk(limbs_global["foot_left"], camera_root, 0.05, left_color)
        self.draw_foot_disk(limbs_global["foot_right"], camera_root, 0.05, right_color)
        self.draw_pelvis(camera_root, self.sim.current_limbs.pelvis, camera_root)
        self.draw_root_cross(camera_root, camera_root)

        self.draw_slide_widget()
        if self._experiment_active():
            self.draw_loss_widget()
        self.draw_ui()
        if self.show_settings:
            self.draw_settings_panel()
        pygame.display.flip()

    def draw_settings_panel(self) -> None:
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        self.screen.blit(overlay, (0, 0))

        panel = self.settings_panel
        pygame.draw.rect(self.screen, LOSS_PANEL_BG, panel, border_radius=10)
        pygame.draw.rect(self.screen, LOSS_PANEL_BORDER, panel, 2, border_radius=10)

        title = self.font.render("Settings", True, UI_TEXT)
        self.screen.blit(title, (panel.x + 16, panel.y + 14))

        mouse_pos = pygame.mouse.get_pos()
        label = self.font_small.render("Loss / NN refresh interval (seconds)", True, UI_TEXT)
        self.screen.blit(label, (panel.x + 16, panel.y + SETTINGS_REFRESH_LABEL_DY))

        self._draw_button(self.refresh_minus_btn, "-", self.refresh_minus_btn.collidepoint(mouse_pos))
        self._draw_button(self.refresh_plus_btn, "+", self.refresh_plus_btn.collidepoint(mouse_pos))

        value = self.font.render(f"{self._draft_refresh_sec:.0f}", True, UI_TEXT)
        value_rect = value.get_rect(center=(panel.centerx, self.refresh_minus_btn.centery))
        self.screen.blit(value, value_rect)

        ar_label = self.font_small.render("Autoregressive window (K)", True, UI_TEXT)
        self.screen.blit(ar_label, (panel.x + 16, panel.y + SETTINGS_AR_LABEL_DY))
        self._draw_button(
            self.ar_window_minus_btn,
            "-",
            self.ar_window_minus_btn.collidepoint(mouse_pos),
        )
        self._draw_button(
            self.ar_window_plus_btn,
            "+",
            self.ar_window_plus_btn.collidepoint(mouse_pos),
        )
        ar_value = self.font.render(str(self._draft_ar_window), True, UI_TEXT)
        ar_value_rect = ar_value.get_rect(center=(panel.centerx, self.ar_window_minus_btn.centery))
        self.screen.blit(ar_value, ar_value_rect)

        header = self.font_tiny.render("Loss term", True, UI_TEXT)
        coeff_header = self.font_tiny.render("Coeff", True, UI_TEXT)
        self.screen.blit(header, (panel.x + 16, panel.y + SETTINGS_TERMS_HEADER_DY))
        self.screen.blit(coeff_header, (panel.x + 210, panel.y + SETTINGS_TERMS_HEADER_DY))

        for i, spec in enumerate(LOSS_TERMS):
            row_y = panel.y + SETTINGS_TERMS_ROW_DY + i * SETTINGS_LOSS_TERM_ROW_H
            enabled = self._draft_loss_terms.get(spec.key, True)
            coeff = self._draft_loss_coeffs.get(spec.key, spec.default_coeff)
            name_surf = self.font_tiny.render(spec.label, True, UI_TEXT)
            self.screen.blit(name_surf, (panel.x + 16, row_y + 12))
            minus_rect = self.loss_term_coeff_minus_btns[spec.key]
            plus_rect = self.loss_term_coeff_plus_btns[spec.key]
            self._draw_button(minus_rect, "-", minus_rect.collidepoint(mouse_pos))
            self._draw_button(plus_rect, "+", plus_rect.collidepoint(mouse_pos))
            coeff_surf = self.font_tiny.render(self._format_loss_coeff(coeff), True, UI_TEXT)
            coeff_rect = coeff_surf.get_rect(center=(panel.x + 232, row_y + 22))
            self.screen.blit(coeff_surf, coeff_rect)
            toggle_rect = self.loss_term_toggle_btns[spec.key]
            self._draw_button(
                toggle_rect,
                "ON" if enabled else "OFF",
                toggle_rect.collidepoint(mouse_pos),
            )

        self._draw_button(self.settings_save_btn, "Save", self.settings_save_btn.collidepoint(mouse_pos))
        self._draw_button(self.settings_close_btn, "Close", self.settings_close_btn.collidepoint(mouse_pos))

    def _format_loss_coeff(self, value: float) -> str:
        if value >= 1000 or (0 < value < 0.01):
            return f"{value:.3g}"
        if value >= 100:
            return f"{value:.0f}"
        if value >= 10:
            return f"{value:.1f}"
        return f"{value:.3g}"

    def _adjust_coeff_draft(self, key: str, direction: int) -> None:
        current = self._draft_loss_coeffs.get(key, 1.0)
        factor = COEFF_STEP_FACTOR if direction > 0 else 1.0 / COEFF_STEP_FACTOR
        self._draft_loss_coeffs[key] = max(MIN_LOSS_COEFF, min(MAX_LOSS_COEFF, current * factor))

    def _adjust_ar_window_draft(self, delta: int) -> None:
        self._draft_ar_window = max(
            MIN_AUTOREGRESSIVE_WINDOW,
            min(MAX_AUTOREGRESSIVE_WINDOW, self._draft_ar_window + delta),
        )

    def _open_settings(self) -> None:
        self.show_settings = True
        self._draft_refresh_sec = self.settings_store.checkpoint_refresh_sec()
        self._draft_ar_window = self.settings_store.autoregressive_window()
        self._draft_loss_terms = self.settings_store.loss_terms_enabled()
        self._draft_loss_coeffs = normalize_loss_coeffs(self.settings_store.loss_coeffs())
        self._layout_settings_panel()

    def _close_settings(self) -> None:
        self.show_settings = False
        self._draft_refresh_sec = self.settings_store.checkpoint_refresh_sec()
        self._draft_ar_window = self.settings_store.autoregressive_window()
        self._draft_loss_terms = self.settings_store.loss_terms_enabled()
        self._draft_loss_coeffs = normalize_loss_coeffs(self.settings_store.loss_coeffs())

    def _adjust_refresh_draft(self, delta: float) -> None:
        self._draft_refresh_sec = max(
            MIN_CHECKPOINT_REFRESH_SEC,
            min(MAX_CHECKPOINT_REFRESH_SEC, self._draft_refresh_sec + delta),
        )

    def _save_settings(self) -> None:
        self.settings_store.save(
            {
                "checkpoint_refresh_sec": self._draft_refresh_sec,
                "autoregressive_window": float(self._draft_ar_window),
            },
            loss_terms=self._draft_loss_terms,
            loss_coeffs=normalize_loss_coeffs(self._draft_loss_coeffs),
        )
        self._checkpoint_timer_ms = 0
        self.show_settings = False

    def _sync_checkpoint(self, force: bool = False) -> None:
        if not self.checkpoint_dir:
            return
        try:
            ckpt = load_checkpoint(self.checkpoint_dir)
            if ckpt is None:
                return
            self.agent.load_weights(ckpt["w1"], ckpt["b1"], ckpt["w2"], ckpt["b2"])
            self._train_step = int(ckpt["step"])
            self._train_loss = float(ckpt["loss"])
            points = load_loss_history(self.checkpoint_dir)
            for spec in LOSS_TERMS:
                self._loss_term_histories[spec.key] = [
                    float(p[spec.key]) for p in points if spec.key in p
                ]
        except OSError:
            return
        if force:
            self._checkpoint_timer_ms = 0

    def _metrics_mtime(self) -> float:
        if not self.checkpoint_dir:
            return 0.0
        path = self.checkpoint_dir / "metrics.json"
        if not path.is_file():
            return 0.0
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _recover_sim_pose(self) -> None:
        self.sim.reset_pose()
        self._reset_slide_history()

    def step_simulation(self) -> None:
        if self.sim.finished:
            return
        if not limb_state_finite(self.sim.current_limbs) or not limb_state_finite(self.sim.previous_limbs):
            self._recover_sim_pose()
            return
        agent_input = self.sim.build_agent_input()
        if not agent_input_finite(agent_input):
            self._recover_sim_pose()
            return
        output = self.agent.predict(agent_input)
        if not agent_output_finite(output):
            self._recover_sim_pose()
            return
        self._record_pinned_foot_slide(output)
        self.sim.apply_agent_output(output)
        if not limb_state_finite(self.sim.current_limbs):
            self._recover_sim_pose()

    def handle_loop(self) -> None:
        self.sim.reset()
        self._reset_slide_history()

    def _experiment_active(self) -> bool:
        return self.experiment_runner is not None

    def _enabled_loss_terms(self) -> tuple:
        enabled = self.settings_store.loss_terms_enabled()
        return tuple(spec for spec in LOSS_TERMS if enabled.get(spec.key, True))

    def _launch_experiment(self, fresh: bool = True) -> None:
        import shutil

        from kinematic2d.experiment_runner import ExperimentRunner

        config = self.settings_store.get_experiment_config()
        checkpoint_dir = config.checkpoint_dir

        if self.experiment_runner is not None:
            self.experiment_runner.stop()

        if fresh and checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)

        self.checkpoint_dir = checkpoint_dir
        self.experiment_runner = ExperimentRunner(
            checkpoint_dir=checkpoint_dir,
            batch_size=config.batch_size,
            lr=config.lr,
            seed=config.seed,
            loss_terms_enabled=self.settings_store.loss_terms_enabled(),
        )
        self.training_mode = True
        self.seed = config.seed
        pygame.display.set_caption("MyPPO — Training")
        self.experiment_runner.start()
        self._reset_experiment_view_state(reinit_agent=True)

    def _reset_experiment_view_state(self, reinit_agent: bool) -> None:
        self.sim.reset()
        self._loss_term_histories = {spec.key: [] for spec in LOSS_TERMS}
        self._train_step = 0
        self._train_loss = float("nan")
        self._checkpoint_timer_ms = 0
        self._last_metrics_mtime = 0.0
        self._sim_steps_this_second = 0
        self._sim_rate = 0.0
        self._rate_timer_ms = 0
        self._reset_slide_history()
        if reinit_agent:
            self.agent.reinit_weights(seed=self.seed)
        if self._experiment_active() and self.checkpoint_dir:
            self._sync_checkpoint(force=True)

    def handle_experiment_reset(self) -> None:
        if self.experiment_runner is None:
            self._launch_experiment(fresh=True)
            return
        self.experiment_runner.reset_and_relaunch()
        self._reset_experiment_view_state(reinit_agent=True)

    def handle_reset(self) -> None:
        self.sim.reset()
        self._reset_slide_history()
        self._reset_view()
        if not self._experiment_active():
            self.agent.reinit_weights(seed=self.seed)
        elif self.checkpoint_dir:
            self._sync_checkpoint(force=True)
        self._sim_steps_this_second = 0
        self._sim_rate = 0.0
        self._rate_timer_ms = 0

    def _handle_settings_click(self, pos: tuple[int, int]) -> bool:
        if self.refresh_minus_btn.collidepoint(pos):
            self._adjust_refresh_draft(-REFRESH_STEP_SEC)
            return True
        if self.refresh_plus_btn.collidepoint(pos):
            self._adjust_refresh_draft(REFRESH_STEP_SEC)
            return True
        if self.ar_window_minus_btn.collidepoint(pos):
            self._adjust_ar_window_draft(-1)
            return True
        if self.ar_window_plus_btn.collidepoint(pos):
            self._adjust_ar_window_draft(1)
            return True
        if self.settings_save_btn.collidepoint(pos):
            self._save_settings()
            return True
        for key, rect in self.loss_term_toggle_btns.items():
            if rect.collidepoint(pos):
                self._draft_loss_terms[key] = not self._draft_loss_terms.get(key, True)
                return True
        for key, rect in self.loss_term_coeff_minus_btns.items():
            if rect.collidepoint(pos):
                self._adjust_coeff_draft(key, -1)
                return True
        for key, rect in self.loss_term_coeff_plus_btns.items():
            if rect.collidepoint(pos):
                self._adjust_coeff_draft(key, 1)
                return True
        if self.settings_close_btn.collidepoint(pos):
            self._close_settings()
            return True
        return False

    def _request_exit(self) -> None:
        self._user_requested_exit = True
        viz_shutdown.mark()
        self.running = False

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._request_exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.show_settings:
                        self._close_settings()
                    else:
                        self._request_exit()
                elif event.key == pygame.K_SPACE and not self.show_settings:
                    self.paused = not self.paused
                elif event.key == pygame.K_r and not self.show_settings:
                    self.handle_reset()
                elif event.key == pygame.K_n and not self.show_settings:
                    self.handle_experiment_reset()
                elif event.key == pygame.K_t and not self.show_settings:
                    self.show_trajectory = not self.show_trajectory
                elif event.key == pygame.K_s:
                    if self.show_settings:
                        self._close_settings()
                    else:
                        self._open_settings()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if self.show_settings:
                    if not self.settings_panel.collidepoint(event.pos):
                        self._close_settings()
                    else:
                        self._handle_settings_click(event.pos)
                    continue
                if self.settings_button.collidepoint(event.pos):
                    self._open_settings()
                elif self.new_run_button.collidepoint(event.pos):
                    self.handle_experiment_reset()
                elif self.reset_button.collidepoint(event.pos):
                    self.handle_reset()
            elif event.type == pygame.MOUSEWHEEL and not self.show_settings:
                focus = pygame.mouse.get_pos()
                if self._scroll_zoom_allowed(focus):
                    self._adjust_zoom(event.y, focus)

    def run(self) -> None:
        try:
            while self.running:
                dt = self.clock.tick(TARGET_HZ)
                self.handle_events()

                if self._experiment_active() and self.checkpoint_dir:
                    metrics_mtime = self._metrics_mtime()
                    if metrics_mtime > self._last_metrics_mtime:
                        self._last_metrics_mtime = metrics_mtime
                        self._sync_checkpoint()

                    self._checkpoint_timer_ms += dt
                    poll_ms = self._checkpoint_poll_ms()
                    if self._checkpoint_timer_ms >= poll_ms:
                        self._sync_checkpoint()
                        self._checkpoint_timer_ms = 0

                if not self.paused:
                    if self.sim.finished:
                        self.handle_loop()
                    else:
                        self.step_simulation()
                        self._sim_steps_this_second += 1

                self._rate_timer_ms += dt
                if self._rate_timer_ms >= 1000:
                    self._sim_rate = self._sim_steps_this_second
                    self._sim_steps_this_second = 0
                    self._rate_timer_ms = 0

                self.render_frame()
        finally:
            if self.experiment_runner is not None:
                self.experiment_runner.stop(join=not self._user_requested_exit)

        pygame.quit()
        if self._user_requested_exit:
            sys.exit(0)


def main(
    training_mode: bool = False,
    checkpoint_dir: Path | str | None = None,
    experiment_runner: ExperimentRunner | None = None,
) -> None:
    viz = Visualizer(
        training_mode=training_mode,
        checkpoint_dir=checkpoint_dir,
        experiment_runner=experiment_runner,
    )
    viz.run()


if __name__ == "__main__":
    main()
    sys.exit(0)
