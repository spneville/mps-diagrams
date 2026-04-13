# Tensor Network Diagram Editor

An interactive GUI application for building matrix product state (MPS) and matrix product operator (MPO) tensor network diagrams.

## Requirements

- Python 3.10+
- PySide6

```bash
pip install PySide6
```

On Linux you may also need:

```bash
sudo apt-get install -y libxcb-cursor0
```

## Usage

```bash
python mps_diagrams.py
```

## Building Blocks

| Block | Shape | Legs | Shortcut |
|-------|-------|------|----------|
| MPS site tensor | Circle | 3 (left, right, down) | M |
| MPO site tensor | Square | 4 (left, right, up, down) | O |
| Left boundary | Tall rectangle | 3 (all right-facing) | L |
| Right boundary | Tall rectangle | 3 (all left-facing) | R |

## Controls

### Tool Selection

| Key | Action |
|-----|--------|
| S | Select mode |
| M | MPS placement mode |
| O | MPO placement mode |
| L | Left boundary placement mode |
| R | Right boundary placement mode |

### Editing

| Shortcut | Action |
|----------|--------|
| Click + drag | Move selected tensors (connected tensors move as a group) |
| Click on empty space | Deselect all |
| Middle-click drag | Pan the canvas |
| Scroll wheel | Zoom |
| Ctrl + +/- | Zoom in/out |
| [ | Rotate selected 90° counter-clockwise |
| ] | Rotate selected 90° clockwise |
| Delete / Backspace | Delete selected tensors |
| Ctrl+C | Copy selected tensors |
| Ctrl+V | Paste at cursor position |
| Ctrl+Z | Undo |
| Ctrl+Y | Redo |
| Ctrl+Shift+S | Export diagram as image |

### Snapping

Tensors automatically snap together when their opposing legs are placed close enough. Connected legs maintain a constant bond length. Boundary tensor legs are spaced to align with the three rows of an MPS-MPO-MPS layout.

### Saving and Loading

Diagrams can be saved and loaded using the `.mpsd` file format (or alternatively `.json`). Use Ctrl+S to save and the load button to open a previously saved diagram.

### Export

Diagrams can be exported as PNG, JPEG, SVG, or PDF via the toolbar save button or Ctrl+Shift+S.
