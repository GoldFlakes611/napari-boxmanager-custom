import math
import numpy as np

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSpinBox, QComboBox, QCheckBox
)


class GalleryWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()

        self.viewer = napari_viewer
        self.gallery_layer = None

        self.tile_to_particle = {}
        self.tile_shape = None
        self.base_gallery = None

        self.selected_particle_indices = set()
        self.selected_tile_ids = set()

        layout = QVBoxLayout()

        self.image_combo = QComboBox()
        self.points_combo = QComboBox()
        self.refresh_button = QPushButton("Refresh layers")

        self.box_size = QSpinBox()
        self.box_size.setRange(8, 512)
        self.box_size.setValue(64)

        self.ncols = QSpinBox()
        self.ncols.setRange(1, 50)
        self.ncols.setValue(10)

        self.projection = QComboBox()
        self.projection.addItems(["middle", "mean", "max"])

        self.hide_rejected = QCheckBox("Hide hidden/rejected points")
        self.hide_rejected.setChecked(True)

        self.build_button = QPushButton("Build gallery")
        self.jump_button = QPushButton("Jump to selected")
        self.reject_button = QPushButton("Reject selected")
        self.keep_button = QPushButton("Keep selected")

        layout.addWidget(QLabel("Tomogram image layer"))
        layout.addWidget(self.image_combo)
        layout.addWidget(QLabel("Particle points layer"))
        layout.addWidget(self.points_combo)

        row = QHBoxLayout()
        row.addWidget(QLabel("Crop size"))
        row.addWidget(self.box_size)
        row.addWidget(QLabel("Gallery columns"))
        row.addWidget(self.ncols)
        layout.addLayout(row)

        layout.addWidget(QLabel("Thumbnail projection"))
        layout.addWidget(self.projection)
        layout.addWidget(self.hide_rejected)

        layout.addWidget(self.refresh_button)
        layout.addWidget(self.build_button)
        layout.addWidget(self.jump_button)
        layout.addWidget(self.reject_button)
        layout.addWidget(self.keep_button)

        self.status = QLabel("Selected: none")
        layout.addWidget(self.status)

        self.setLayout(layout)

        self.refresh_button.clicked.connect(self.refresh_layers)
        self.build_button.clicked.connect(self.build_gallery)
        self.jump_button.clicked.connect(self.jump_to_selected)
        self.reject_button.clicked.connect(lambda: self.set_keep_for_selected(False))
        self.keep_button.clicked.connect(lambda: self.set_keep_for_selected(True))

        self.refresh_layers()

    def refresh_layers(self):
        self.image_combo.clear()
        self.points_combo.clear()

        for layer in self.viewer.layers:
            if layer.__class__.__name__ == "Image":
                if layer.name != "particle_gallery":
                    self.image_combo.addItem(layer.name)
            elif layer.__class__.__name__ == "Points":
                self.points_combo.addItem(layer.name)

    def _get_layer(self, combo):
        name = combo.currentText()
        if not name:
            return None
        return self.viewer.layers[name]

    def _ensure_shown_mask(self, points_layer):
        """Ensure the Points layer has a boolean shown mask.

        BoxManager's writer already respects ``meta["shown"]`` when saving.
        Using ``shown`` avoids adding a custom feature column such as ``keep``,
        which can make saved coordinate files incompatible with BoxManager's
        expected metadata schema.
        """
        n = len(points_layer.data)

        shown = getattr(points_layer, "shown", None)
        if shown is None or len(shown) != n:
            points_layer.shown = np.ones(n, dtype=bool)
            return

        points_layer.shown = np.asarray(shown, dtype=bool)

    def _crop_thumbnail(self, volume, point, box):
        z, y, x = np.round(point[-3:]).astype(int)
        half = box // 2

        z0, z1 = z - half, z + half
        y0, y1 = y - half, y + half
        x0, x1 = x - half, x + half

        crop = np.zeros((box, box, box), dtype=np.float32)

        vz0, vz1 = max(z0, 0), min(z1, volume.shape[-3])
        vy0, vy1 = max(y0, 0), min(y1, volume.shape[-2])
        vx0, vx1 = max(x0, 0), min(x1, volume.shape[-1])

        cz0, cy0, cx0 = vz0 - z0, vy0 - y0, vx0 - x0
        cz1 = cz0 + (vz1 - vz0)
        cy1 = cy0 + (vy1 - vy0)
        cx1 = cx0 + (vx1 - vx0)

        crop[cz0:cz1, cy0:cy1, cx0:cx1] = volume[
            vz0:vz1,
            vy0:vy1,
            vx0:vx1,
        ]

        mode = self.projection.currentText()

        if mode == "mean":
            thumb = crop.mean(axis=0)
        elif mode == "max":
            thumb = crop.max(axis=0)
        else:
            thumb = crop[crop.shape[0] // 2]

        return self._normalize_thumbnail(thumb)

    def _normalize_thumbnail(self, img):
        img = img.astype(np.float32, copy=False)
        lo, hi = np.percentile(img, [1, 99])

        if hi <= lo:
            return np.zeros_like(img, dtype=np.float32)

        return np.clip((img - lo) / (hi - lo), 0, 1)

    def _gray_to_rgb(self, gray):
        rgb = np.repeat(gray[..., None], 3, axis=-1)
        return rgb.astype(np.float32)

    def _draw_selection_box(self, rgb, row, col):
        box_y, box_x, _ = self.tile_shape

        y0, y1 = row * box_y, (row + 1) * box_y
        x0, x1 = col * box_x, (col + 1) * box_x

        thickness = 3

        rgb[y0:y0 + thickness, x0:x1, :] = [1, 0, 0]
        rgb[y1 - thickness:y1, x0:x1, :] = [1, 0, 0]
        rgb[y0:y1, x0:x0 + thickness, :] = [1, 0, 0]
        rgb[y0:y1, x1 - thickness:x1, :] = [1, 0, 0]

        return rgb

    def build_gallery(self):
        image_layer = self._get_layer(self.image_combo)
        points_layer = self._get_layer(self.points_combo)

        if image_layer is None or points_layer is None:
            self.status.setText("Select both an image layer and a points layer.")
            return

        volume = np.asarray(image_layer.data)
        points = np.asarray(points_layer.data)

        if volume.ndim < 3:
            raise ValueError("GalleryWidget expects a 3D tomogram image layer.")

        self._ensure_shown_mask(points_layer)

        box = int(self.box_size.value())
        ncols = int(self.ncols.value())

        shown = np.asarray(points_layer.shown, dtype=bool)

        visible_indices = [
            i for i in range(len(points))
            if shown[i] or not self.hide_rejected.isChecked()
        ]

        if len(visible_indices) == 0:
            self.status.setText("No particles to show.")
            return

        nrows = math.ceil(len(visible_indices) / ncols)
        gallery = np.zeros((nrows * box, ncols * box), dtype=np.float32)

        self.tile_to_particle = {}

        for tile_id, particle_index in enumerate(visible_indices):
            r = tile_id // ncols
            c = tile_id % ncols

            thumb = self._crop_thumbnail(volume, points[particle_index], box)

            gallery[
                r * box:(r + 1) * box,
                c * box:(c + 1) * box,
            ] = thumb

            self.tile_to_particle[tile_id] = particle_index

        self.tile_shape = (box, box, ncols)
        self.base_gallery = gallery
        self.selected_particle_indices = set()
        self.selected_tile_ids = set()

        if self.gallery_layer is None or self.gallery_layer not in self.viewer.layers:
            self.gallery_layer = self.viewer.add_image(
                self._gray_to_rgb(gallery),
                name="particle_gallery",
                rgb=True,
                metadata={"is_particle_gallery": True},
            )
        else:
            self.gallery_layer.data = self._gray_to_rgb(gallery)
            self.gallery_layer.refresh()

        if self._on_gallery_click not in self.gallery_layer.mouse_drag_callbacks:
            self.gallery_layer.mouse_drag_callbacks.append(self._on_gallery_click)

        self.viewer.layers.selection.active = self.gallery_layer
        self.status.setText(
            f"Gallery built: {len(visible_indices)} particles. Click a tile to select."
        )

    def _redraw_selection_overlay(self):
        if self.base_gallery is None or self.tile_shape is None:
            return

        _, _, ncols = self.tile_shape
        rgb = self._gray_to_rgb(self.base_gallery)

        for tile_id in self.selected_tile_ids:
            row = tile_id // ncols
            col = tile_id % ncols
            rgb = self._draw_selection_box(rgb, row, col)

        self.gallery_layer.data = rgb
        self.gallery_layer.refresh()

    def _on_gallery_click(self, layer, event):
        if self.tile_shape is None or self.base_gallery is None:
            return

        y, x = event.position[-2:]

        box_y, box_x, ncols = self.tile_shape

        row = int(y // box_y)
        col = int(x // box_x)

        tile_id = row * ncols + col
        particle_index = self.tile_to_particle.get(tile_id)

        if particle_index is None:
            self.status.setText("Clicked outside particle tiles.")
            return

        # Click once to select. Click the same tile again to deselect.
        if particle_index in self.selected_particle_indices:
            self.selected_particle_indices.remove(particle_index)
            self.selected_tile_ids.discard(tile_id)
        else:
            self.selected_particle_indices.add(particle_index)
            self.selected_tile_ids.add(tile_id)

        points_layer = self._get_layer(self.points_combo)
        points_layer.selected_data = set(self.selected_particle_indices)
        points_layer.refresh()

        self._redraw_selection_overlay()

        n_selected = len(self.selected_particle_indices)
        if n_selected == 0:
            self.status.setText("Selected: none")
        else:
            self.status.setText(
                f"Selected {n_selected} particle(s). "
                "Use Reject, Keep, or Jump to selected."
            )

    def jump_to_selected(self):
        points_layer = self._get_layer(self.points_combo)

        if self.selected_particle_indices:
            particle_index = next(iter(self.selected_particle_indices))
            self.jump_to_particle(particle_index)
            return

        if points_layer is None or not points_layer.selected_data:
            self.status.setText("No particle selected.")
            return

        particle_index = next(iter(points_layer.selected_data))
        self.jump_to_particle(particle_index)

    def jump_to_particle(self, particle_index):
        points_layer = self._get_layer(self.points_combo)

        if points_layer is None:
            self.status.setText("No points layer selected.")
            return

        point = np.asarray(points_layer.data[particle_index])
        z, y, x = point[-3:]

        try:
            self.viewer.dims.set_point(0, z)
        except Exception:
            pass

        try:
            self.viewer.camera.center = tuple(point[-3:])
        except Exception:
            pass

        self.viewer.layers.selection.active = points_layer
        self.status.setText(f"Jumped to particle {particle_index}.")

    def set_keep_for_selected(self, keep_value):
        points_layer = self._get_layer(self.points_combo)

        if points_layer is None:
            self.status.setText("No points layer selected.")
            return

        indices = set(self.selected_particle_indices)

        if not indices and points_layer.selected_data:
            indices = set(points_layer.selected_data)

        if not indices:
            self.status.setText("No particles selected.")
            return

        self._ensure_shown_mask(points_layer)

        shown = np.asarray(points_layer.shown, dtype=bool).copy()
        for idx in indices:
            if 0 <= idx < len(shown):
                shown[idx] = bool(keep_value)
        points_layer.shown = shown

        points_layer.refresh()

        state = "kept" if keep_value else "rejected"
        n_changed = len(indices)

        # Rebuild so hidden rejected particles disappear from the gallery.
        self.build_gallery()
        self.status.setText(f"{n_changed} particle(s) marked as {state}.")
