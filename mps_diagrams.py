#!/usr/bin/env python3
"""
Tensor Network Diagram Editor
Interactive GUI for building MPS/MPO tensor network diagrams.
"""

import sys
import json
import math
from abc import ABC, abstractmethod
from enum import Enum, auto
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsScene, QGraphicsView,
    QGraphicsItem, QGraphicsEllipseItem, QGraphicsRectItem,
    QGraphicsLineItem, QToolBar, QToolButton, QButtonGroup,
    QWidget, QVBoxLayout, QStatusBar, QGraphicsPathItem,
    QColorDialog, QFileDialog, QMenu,
)
from PySide6.QtCore import Qt, QPointF, QRectF, QLineF
from PySide6.QtGui import (
    QPen, QBrush, QColor, QPainter, QAction, QFont,
    QPainterPath, QTransform, QImage, QKeySequence,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_SIZE = 60
LEG_LENGTH = 30
BOND_LENGTH = 40          # fixed length of connection line between snapped tensors
LEG_SNAP_RADIUS = 25
TENSOR_RADIUS = 22       # MPS circle radius
TENSOR_SQUARE = 44       # MPO square side length (matches MPS diameter for alignment)
BOUNDARY_W = 30          # boundary tensor width
BOUNDARY_H = 44          # boundary tensor height
# Vertical distance between MPS-MPO-MPS row centers
# Must match the snapped center-to-center distance: body_offset + BOND_LENGTH + body_offset
ROW_SPACING = TENSOR_RADIUS + BOND_LENGTH + TENSOR_SQUARE // 2

COLOR_MPS = QColor("#5b9bd5")
COLOR_MPO = QColor("#ed7d31")
COLOR_BOUNDARY = QColor("#70ad47")
COLOR_LEG = QColor("#333333")
COLOR_LEG_CONNECTED = QColor("#1a8c1a")
COLOR_SNAP_HINT = QColor(100, 200, 100, 120)
COLOR_BACKGROUND = QColor(255, 255, 255)
COLOR_GRID = QColor(220, 220, 220)
COLOR_SELECTION = QColor("#2270d4")


# ---------------------------------------------------------------------------
# Undo / Redo infrastructure
# ---------------------------------------------------------------------------

class UndoAction(ABC):
    @abstractmethod
    def undo(self): ...
    @abstractmethod
    def redo(self): ...


class UndoStack:
    def __init__(self):
        self._undo: list[UndoAction] = []
        self._redo: list[UndoAction] = []

    def push(self, action: UndoAction):
        self._undo.append(action)
        self._redo.clear()

    def undo(self):
        if self._undo:
            action = self._undo.pop()
            action.undo()
            self._redo.append(action)

    def redo(self):
        if self._redo:
            action = self._redo.pop()
            action.redo()
            self._undo.append(action)

    @property
    def can_undo(self) -> bool:
        return len(self._undo) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo) > 0


class PlaceAction(UndoAction):
    """Undoable action: placing a tensor on the scene."""
    def __init__(self, scene: "TensorScene", tensor: "TensorItem",
                 connections: list[tuple["Leg", "TensorItem", "Leg", "TensorItem"]]):
        self.scene = scene
        self.tensor = tensor
        # snapshot of connections formed during placement
        self.connection_info = connections

    def undo(self):
        self.scene.remove_connections_for(self.tensor)
        self.scene.removeItem(self.tensor)

    def redo(self):
        self.scene.addItem(self.tensor)
        for leg_a, tensor_a, leg_b, tensor_b in self.connection_info:
            self.scene.add_connection(leg_a, tensor_a, leg_b, tensor_b)


class DeleteAction(UndoAction):
    """Undoable action: deleting one or more tensors from the scene."""
    def __init__(self, scene: "TensorScene",
                 tensors: list["TensorItem"],
                 connections: list[tuple["Leg", "TensorItem", "Leg", "TensorItem"]],
                 positions: list[QPointF]):
        self.scene = scene
        self.tensors = tensors
        self.connection_info = connections
        self.positions = positions

    def undo(self):
        for tensor, pos in zip(self.tensors, self.positions):
            tensor._moving_group = True
            self.scene.addItem(tensor)
            tensor.setPos(pos)
            tensor._moving_group = False
        for leg_a, tensor_a, leg_b, tensor_b in self.connection_info:
            self.scene.add_connection(leg_a, tensor_a, leg_b, tensor_b)

    def redo(self):
        for tensor in self.tensors:
            self.scene.remove_connections_for(tensor)
            self.scene.removeItem(tensor)


class RotateAction(UndoAction):
    """Undoable action: rotating a tensor by 90 degrees."""
    def __init__(self, scene: "TensorScene", tensor: "TensorItem", clockwise: bool,
                 old_connections: list[tuple["Leg", "TensorItem", "Leg", "TensorItem"]]):
        self.scene = scene
        self.tensor = tensor
        self.clockwise = clockwise
        self.old_connections = old_connections

    def undo(self):
        # rotate the opposite way
        self.tensor.rotate_90(not self.clockwise)
        # restore old connections
        for leg_a, tensor_a, leg_b, tensor_b in self.old_connections:
            self.scene.add_connection(leg_a, tensor_a, leg_b, tensor_b)

    def redo(self):
        self.scene.remove_connections_for(self.tensor)
        self.tensor.rotate_90(self.clockwise)


class ColorAction(UndoAction):
    """Undoable action: changing the color of tensors."""
    def __init__(self, tensors: list["TensorItem"],
                 old_colors: list[QColor], new_color: QColor):
        self.tensors = tensors
        self.old_colors = old_colors
        self.new_color = new_color

    def undo(self):
        for tensor, old_color in zip(self.tensors, self.old_colors):
            tensor.color = QColor(old_color)
            tensor.update()

    def redo(self):
        for tensor in self.tensors:
            tensor.color = QColor(self.new_color)
            tensor.update()


class MoveAction(UndoAction):
    """Undoable action: moving a connected group of tensors.
    Also tracks connections formed (snapped) or broken during the move.
    """
    def __init__(self, scene: "TensorScene",
                 items_and_old_pos: list[tuple["TensorItem", QPointF]],
                 items_and_new_pos: list[tuple["TensorItem", QPointF]],
                 conns_formed: list[tuple["Leg", "TensorItem", "Leg", "TensorItem"]],
                 conns_broken: list[tuple["Leg", "TensorItem", "Leg", "TensorItem"]]):
        self.scene = scene
        self.items_old = items_and_old_pos
        self.items_new = items_and_new_pos
        self.conns_formed = conns_formed
        self.conns_broken = conns_broken

    def undo(self):
        # remove connections that were formed during the snap
        for leg_a, tensor_a, leg_b, tensor_b in self.conns_formed:
            self.scene._remove_connection(leg_a, leg_b)
        # move tensors back to pre-drag positions
        for tensor, pos in self.items_old:
            tensor._moving_group = True
            tensor.setPos(pos)
            tensor._moving_group = False
        # restore connections that were broken during the move
        for leg_a, tensor_a, leg_b, tensor_b in self.conns_broken:
            self.scene.add_connection(leg_a, tensor_a, leg_b, tensor_b)
        self.scene.update_connections()

    def redo(self):
        # break connections that will be broken by the move
        for leg_a, tensor_a, leg_b, tensor_b in self.conns_broken:
            self.scene._remove_connection(leg_a, leg_b)
        # move tensors to post-drag positions
        for tensor, pos in self.items_new:
            tensor._moving_group = True
            tensor.setPos(pos)
            tensor._moving_group = False
        # recreate connections formed by the snap
        for leg_a, tensor_a, leg_b, tensor_b in self.conns_formed:
            self.scene.add_connection(leg_a, tensor_a, leg_b, tensor_b)
        self.scene.update_connections()


# ---------------------------------------------------------------------------
# Enum for placement tool
# ---------------------------------------------------------------------------

class ToolMode(Enum):
    SELECT = auto()
    MPS = auto()
    MPO = auto()
    MPS_LEFT = auto()
    MPS_RIGHT = auto()
    BOUNDARY_LEFT = auto()
    BOUNDARY_RIGHT = auto()


# ---------------------------------------------------------------------------
# Leg (connection point) on a tensor
# ---------------------------------------------------------------------------

class Direction(Enum):
    UP = auto()
    DOWN = auto()
    LEFT = auto()
    RIGHT = auto()


class Leg:
    """A connection point on a tensor item."""
    def __init__(self, direction: Direction, offset: QPointF):
        self.direction = direction
        self.offset = offset          # relative to parent center
        self.connected_to: "Leg | None" = None
        self.line_item: QGraphicsLineItem | None = None

    def scene_pos(self, parent_center: QPointF) -> QPointF:
        return parent_center + self.offset

    def tip_pos(self, parent_center: QPointF) -> QPointF:
        """End point of the drawn leg line (the free tip)."""
        base = self.scene_pos(parent_center)
        dx, dy = 0.0, 0.0
        if self.direction == Direction.UP:
            dy = -LEG_LENGTH
        elif self.direction == Direction.DOWN:
            dy = LEG_LENGTH
        elif self.direction == Direction.LEFT:
            dx = -LEG_LENGTH
        elif self.direction == Direction.RIGHT:
            dx = LEG_LENGTH
        return QPointF(base.x() + dx, base.y() + dy)


# ---------------------------------------------------------------------------
# Base class for tensor graphics items
# ---------------------------------------------------------------------------

class TensorItem(QGraphicsItem):
    """Abstract base for all tensor diagram building blocks."""

    def __init__(self, legs: list[Leg], parent=None):
        super().__init__(parent)
        self.legs = legs
        self.color: QColor = QColor("#cccccc")  # overridden by subclasses
        self._moving_group = False  # guards against recursive group moves
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self._build_leg_lines()

    # -- subclass must implement ------------------------------------------------
    def _shape_rect(self) -> QRectF:
        raise NotImplementedError

    def _draw_shape(self, painter: QPainter):
        raise NotImplementedError

    def surface_point(self, leg: "Leg") -> QPointF:
        """Return the point on the tensor body surface where the leg visually connects.
        Defaults to leg.offset. Subclasses can override for non-rectangular shapes."""
        return leg.offset

    # -- leg line management ---------------------------------------------------
    def _build_leg_lines(self):
        for leg in self.legs:
            line_item = QGraphicsLineItem(self)
            pen = QPen(COLOR_LEG, 2.5)
            line_item.setPen(pen)
            leg.line_item = line_item
        self._update_leg_lines()

    def _update_leg_lines(self):
        for leg in self.legs:
            if leg.line_item is None:
                continue
            if leg.connected_to is not None:
                # connected legs are drawn by the ConnectionLine instead
                leg.line_item.setVisible(False)
                continue
            leg.line_item.setVisible(True)
            base = leg.offset
            dx, dy = 0.0, 0.0
            if leg.direction == Direction.UP:
                dy = -LEG_LENGTH
            elif leg.direction == Direction.DOWN:
                dy = LEG_LENGTH
            elif leg.direction == Direction.LEFT:
                dx = -LEG_LENGTH
            elif leg.direction == Direction.RIGHT:
                dx = LEG_LENGTH
            tip = QPointF(base.x() + dx, base.y() + dy)
            leg.line_item.setLine(QLineF(base, tip))
            leg.line_item.setPen(QPen(COLOR_LEG, 2.5))

    # -- QGraphicsItem overrides -----------------------------------------------
    def boundingRect(self) -> QRectF:
        margin = LEG_LENGTH + 5
        r = self._shape_rect()
        return r.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._draw_shape(painter)
        # selection highlight
        if self.isSelected():
            pen = QPen(COLOR_SELECTION, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = self._shape_rect().adjusted(-4, -4, 4, 4)
            painter.drawRect(r)

    # -- rotation -------------------------------------------------------------
    _CW = {
        Direction.UP: Direction.RIGHT,
        Direction.RIGHT: Direction.DOWN,
        Direction.DOWN: Direction.LEFT,
        Direction.LEFT: Direction.UP,
    }
    _CCW = {v: k for k, v in _CW.items()}

    def rotate_90(self, clockwise: bool = True):
        """Rotate all legs by 90 degrees. Updates offsets and directions."""
        self.prepareGeometryChange()
        for leg in self.legs:
            ox, oy = leg.offset.x(), leg.offset.y()
            if clockwise:
                leg.offset = QPointF(-oy, ox)
                leg.direction = self._CW[leg.direction]
            else:
                leg.offset = QPointF(oy, -ox)
                leg.direction = self._CCW[leg.direction]
        self._update_leg_lines()
        scene = self.scene()
        if scene and isinstance(scene, TensorScene):
            scene.update_connections()

    def connected_group(self) -> set["TensorItem"]:
        """Return all tensors transitively connected to this one."""
        visited: set[TensorItem] = set()
        stack = [self]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for leg in current.legs:
                if leg.connected_to is not None:
                    # find the tensor that owns the other leg
                    scene = self.scene()
                    if scene:
                        for item in scene.items():
                            if isinstance(item, TensorItem) and item is not current:
                                if leg.connected_to in item.legs:
                                    stack.append(item)
        return visited

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # move the entire connected group together
            if not self._moving_group:
                old_pos = self.pos()
                new_pos = value
                delta = new_pos - old_pos
                if delta.x() != 0 or delta.y() != 0:
                    group = self.connected_group()
                    group.discard(self)  # self is already moving
                    for other in group:
                        other._moving_group = True
                        other.setPos(other.pos() + delta)
                        other._moving_group = False
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self._update_leg_lines()
            scene = self.scene()
            if scene and isinstance(scene, TensorScene):
                scene.update_connections()
        return super().itemChange(change, value)


# ---------------------------------------------------------------------------
# MPS Site Tensor — circle with 3 legs (left, right, down)
# ---------------------------------------------------------------------------

class MPSSiteTensor(TensorItem):
    def __init__(self, parent=None):
        r = TENSOR_RADIUS
        legs = [
            Leg(Direction.LEFT,  QPointF(-r, 0)),
            Leg(Direction.RIGHT, QPointF(r, 0)),
            Leg(Direction.DOWN,  QPointF(0, r)),
        ]
        super().__init__(legs, parent)
        self.color = QColor(COLOR_MPS)

    def _shape_rect(self) -> QRectF:
        r = TENSOR_RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def _draw_shape(self, painter: QPainter):
        r = TENSOR_RADIUS
        painter.setPen(QPen(self.color.darker(130), 2))
        painter.setBrush(QBrush(self.color))
        painter.drawEllipse(QPointF(0, 0), r, r)


# ---------------------------------------------------------------------------
# Left-Orthogonal MPS Site Tensor — right-pointing triangle with 3 legs
# ---------------------------------------------------------------------------

class LeftOrthoMPSTensor(TensorItem):
    def __init__(self, parent=None):
        r = TENSOR_RADIUS
        legs = [
            Leg(Direction.LEFT,  QPointF(-r, 0)),
            Leg(Direction.RIGHT, QPointF(r, 0)),
            Leg(Direction.DOWN,  QPointF(0, r)),
        ]
        super().__init__(legs, parent)
        self.color = QColor(COLOR_MPS)

    def _shape_rect(self) -> QRectF:
        r = TENSOR_RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def surface_point(self, leg):
        """Return the point on the triangle surface where the leg connects."""
        r = TENSOR_RADIUS
        # Bottom edge runs from (-r, r) to (r, 0)
        # At x=0: y = r/2
        if leg.direction == Direction.DOWN:
            return QPointF(leg.offset.x(), r / 2)
        # Top edge runs from (-r, -r) to (r, 0)
        # At x=0: y = -r/2
        if leg.direction == Direction.UP:
            return QPointF(leg.offset.x(), -r / 2)
        return leg.offset

    def _update_leg_lines(self):
        for leg in self.legs:
            if leg.line_item is None:
                continue
            if leg.connected_to is not None:
                leg.line_item.setVisible(False)
                continue
            leg.line_item.setVisible(True)
            base = self.surface_point(leg)
            # tip stays at the same position as standard MPS: offset + LEG_LENGTH
            dx, dy = 0.0, 0.0
            if leg.direction == Direction.UP:
                dy = -LEG_LENGTH
            elif leg.direction == Direction.DOWN:
                dy = LEG_LENGTH
            elif leg.direction == Direction.LEFT:
                dx = -LEG_LENGTH
            elif leg.direction == Direction.RIGHT:
                dx = LEG_LENGTH
            tip = QPointF(leg.offset.x() + dx, leg.offset.y() + dy)
            leg.line_item.setLine(QLineF(base, tip))
            leg.line_item.setPen(QPen(COLOR_LEG, 2.5))

    def _draw_shape(self, painter: QPainter):
        r = TENSOR_RADIUS
        painter.setPen(QPen(self.color.darker(130), 2))
        painter.setBrush(QBrush(self.color))
        triangle = QPainterPath()
        triangle.moveTo(-r, -r)       # top-left
        triangle.lineTo(r, 0)         # right (point)
        triangle.lineTo(-r, r)        # bottom-left
        triangle.closeSubpath()
        painter.drawPath(triangle)


# ---------------------------------------------------------------------------
# Right-Orthogonal MPS Site Tensor — left-pointing triangle with 3 legs
# ---------------------------------------------------------------------------

class RightOrthoMPSTensor(TensorItem):
    def __init__(self, parent=None):
        r = TENSOR_RADIUS
        legs = [
            Leg(Direction.LEFT,  QPointF(-r, 0)),
            Leg(Direction.RIGHT, QPointF(r, 0)),
            Leg(Direction.DOWN,  QPointF(0, r)),
        ]
        super().__init__(legs, parent)
        self.color = QColor(COLOR_MPS)

    def _shape_rect(self) -> QRectF:
        r = TENSOR_RADIUS
        return QRectF(-r, -r, 2 * r, 2 * r)

    def surface_point(self, leg):
        """Return the point on the triangle surface where the leg connects."""
        r = TENSOR_RADIUS
        # Bottom edge runs from (r, r) to (-r, 0)
        # At x=0: y = r/2
        if leg.direction == Direction.DOWN:
            return QPointF(leg.offset.x(), r / 2)
        # Top edge runs from (r, -r) to (-r, 0)
        # At x=0: y = -r/2
        if leg.direction == Direction.UP:
            return QPointF(leg.offset.x(), -r / 2)
        return leg.offset

    def _update_leg_lines(self):
        for leg in self.legs:
            if leg.line_item is None:
                continue
            if leg.connected_to is not None:
                leg.line_item.setVisible(False)
                continue
            leg.line_item.setVisible(True)
            base = self.surface_point(leg)
            dx, dy = 0.0, 0.0
            if leg.direction == Direction.UP:
                dy = -LEG_LENGTH
            elif leg.direction == Direction.DOWN:
                dy = LEG_LENGTH
            elif leg.direction == Direction.LEFT:
                dx = -LEG_LENGTH
            elif leg.direction == Direction.RIGHT:
                dx = LEG_LENGTH
            tip = QPointF(leg.offset.x() + dx, leg.offset.y() + dy)
            leg.line_item.setLine(QLineF(base, tip))
            leg.line_item.setPen(QPen(COLOR_LEG, 2.5))

    def _draw_shape(self, painter: QPainter):
        r = TENSOR_RADIUS
        painter.setPen(QPen(self.color.darker(130), 2))
        painter.setBrush(QBrush(self.color))
        triangle = QPainterPath()
        triangle.moveTo(r, -r)        # top-right
        triangle.lineTo(-r, 0)        # left (point)
        triangle.lineTo(r, r)         # bottom-right
        triangle.closeSubpath()
        painter.drawPath(triangle)


# ---------------------------------------------------------------------------
# MPO Site Tensor — square with 4 legs (left, right, up, down)
# ---------------------------------------------------------------------------

class MPOSiteTensor(TensorItem):
    def __init__(self, parent=None):
        h = TENSOR_SQUARE / 2
        legs = [
            Leg(Direction.LEFT,  QPointF(-h, 0)),
            Leg(Direction.RIGHT, QPointF(h, 0)),
            Leg(Direction.UP,    QPointF(0, -h)),
            Leg(Direction.DOWN,  QPointF(0, h)),
        ]
        super().__init__(legs, parent)
        self.color = QColor(COLOR_MPO)

    def _shape_rect(self) -> QRectF:
        h = TENSOR_SQUARE / 2
        return QRectF(-h, -h, TENSOR_SQUARE, TENSOR_SQUARE)

    def _draw_shape(self, painter: QPainter):
        painter.setPen(QPen(self.color.darker(130), 2))
        painter.setBrush(QBrush(self.color))
        painter.drawRect(self._shape_rect())


# ---------------------------------------------------------------------------
# Boundary Tensors — tall rectangles with 3 legs, spaced to align with
# the three rows of an MPS–MPO–MPS expectation-value diagram.
#   Left boundary:  3 legs all going right
#   Right boundary: 3 legs all going left
# ---------------------------------------------------------------------------

class BoundaryTensor(TensorItem):
    def __init__(self, side: str, parent=None):
        """side: 'left' or 'right'"""
        self.side = side
        hw = BOUNDARY_W / 2
        if side == "left":
            legs = [
                Leg(Direction.RIGHT, QPointF(hw, -ROW_SPACING)),  # top MPS
                Leg(Direction.RIGHT, QPointF(hw, 0)),             # middle MPO
                Leg(Direction.RIGHT, QPointF(hw, ROW_SPACING)),   # bottom MPS
            ]
        else:
            legs = [
                Leg(Direction.LEFT, QPointF(-hw, -ROW_SPACING)),  # top MPS
                Leg(Direction.LEFT, QPointF(-hw, 0)),             # middle MPO
                Leg(Direction.LEFT, QPointF(-hw, ROW_SPACING)),   # bottom MPS
            ]
        super().__init__(legs, parent)
        self.color = QColor(COLOR_BOUNDARY)

    def _shape_rect(self) -> QRectF:
        hw = BOUNDARY_W / 2
        hh = ROW_SPACING + 18  # extend past the top/bottom legs
        return QRectF(-hw, -hh, BOUNDARY_W, 2 * hh)

    def _draw_shape(self, painter: QPainter):
        painter.setPen(QPen(self.color.darker(130), 2))
        painter.setBrush(QBrush(self.color))
        r = self._shape_rect()
        painter.drawRoundedRect(r, 4, 4)


# ---------------------------------------------------------------------------
# Connection line between two legs (drawn separately in the scene)
# ---------------------------------------------------------------------------

class ConnectionLine(QGraphicsLineItem):
    def __init__(self, leg_a: Leg, tensor_a: TensorItem,
                 leg_b: Leg, tensor_b: TensorItem):
        super().__init__()
        self.leg_a = leg_a
        self.tensor_a = tensor_a
        self.leg_b = leg_b
        self.tensor_b = tensor_b
        pen = QPen(COLOR_LEG, 2.5)
        self.setPen(pen)
        self.setZValue(-1)
        self.update_position()

    def update_position(self):
        # draw from body surface to body surface
        p1 = self.tensor_a.scenePos() + self.tensor_a.surface_point(self.leg_a)
        p2 = self.tensor_b.scenePos() + self.tensor_b.surface_point(self.leg_b)
        self.setLine(QLineF(p1, p2))


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class TensorScene(QGraphicsScene):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSceneRect(-2000, -2000, 4000, 4000)
        self.connections: list[ConnectionLine] = []
        self._snap_indicator: QGraphicsEllipseItem | None = None
        self.undo_stack = UndoStack()
        self._exporting = False
        self.selection_order: list[TensorItem] = []  # tracks order of selection

    # -- grid ----------------------------------------------------------------
    def drawBackground(self, painter: QPainter, rect: QRectF):
        painter.fillRect(rect, COLOR_BACKGROUND)
        if self._exporting:
            return
        pen = QPen(COLOR_GRID, 0.5)
        painter.setPen(pen)
        left = int(rect.left()) - (int(rect.left()) % GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % GRID_SIZE)
        x = left
        while x < rect.right():
            painter.drawLine(x, int(rect.top()), x, int(rect.bottom()))
            x += GRID_SIZE
        y = top
        while y < rect.bottom():
            painter.drawLine(int(rect.left()), y, int(rect.right()), y)
            y += GRID_SIZE

    # -- connections ---------------------------------------------------------
    def update_connections(self):
        for conn in self.connections:
            conn.update_position()

    def add_connection(self, leg_a: Leg, tensor_a: TensorItem,
                       leg_b: Leg, tensor_b: TensorItem):
        leg_a.connected_to = leg_b
        leg_b.connected_to = leg_a
        conn = ConnectionLine(leg_a, tensor_a, leg_b, tensor_b)
        self.addItem(conn)
        self.connections.append(conn)
        tensor_a._update_leg_lines()
        tensor_b._update_leg_lines()

    def _remove_connection(self, leg_a: Leg, leg_b: Leg):
        """Remove a specific connection identified by its two legs."""
        for conn in list(self.connections):
            if (conn.leg_a is leg_a and conn.leg_b is leg_b) or \
               (conn.leg_a is leg_b and conn.leg_b is leg_a):
                conn.leg_a.connected_to = None
                conn.leg_b.connected_to = None
                self.removeItem(conn)
                self.connections.remove(conn)
                conn.tensor_a._update_leg_lines()
                conn.tensor_b._update_leg_lines()
                return

    def remove_connections_for(self, tensor: TensorItem):
        to_remove = []
        for conn in self.connections:
            if conn.tensor_a is tensor or conn.tensor_b is tensor:
                conn.leg_a.connected_to = None
                conn.leg_b.connected_to = None
                to_remove.append(conn)
        for conn in to_remove:
            self.removeItem(conn)
            self.connections.remove(conn)
            # restore individual leg lines on the remaining tensor
            if conn.tensor_a is not tensor:
                conn.tensor_a._update_leg_lines()
            if conn.tensor_b is not tensor:
                conn.tensor_b._update_leg_lines()

    # -- snap helpers --------------------------------------------------------
    # Only legs that face each other can connect
    _OPPOSING = {
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
    }

    def find_snap_target(self, tensor: TensorItem, leg: Leg) -> tuple["TensorItem", "Leg"] | None:
        """Find the closest unconnected, opposing leg tip from another tensor.
        Requires that the tips are aligned on the perpendicular axis
        (horizontal legs must share y, vertical legs must share x) so
        that connections are always straight.
        """
        tip = leg.tip_pos(tensor.scenePos())
        required = self._OPPOSING[leg.direction]
        horizontal = leg.direction in (Direction.LEFT, Direction.RIGHT)
        best_dist = LEG_SNAP_RADIUS
        best = None
        for item in self.items():
            if not isinstance(item, TensorItem) or item is tensor:
                continue
            for other_leg in item.legs:
                if other_leg.connected_to is not None:
                    continue
                if other_leg.direction != required:
                    continue
                other_tip = other_leg.tip_pos(item.scenePos())
                # Perpendicular axis must be aligned. Use a tight
                # tolerance when a boundary tensor is involved (its
                # legs must connect exactly horizontally), and a
                # normal tolerance otherwise.
                boundary_involved = (isinstance(tensor, BoundaryTensor)
                                     or isinstance(item, BoundaryTensor))
                perp_tol = LEG_SNAP_RADIUS // 2 if boundary_involved else LEG_SNAP_RADIUS
                if horizontal:
                    if abs(tip.y() - other_tip.y()) > perp_tol:
                        continue
                else:
                    if abs(tip.x() - other_tip.x()) > perp_tol:
                        continue
                dist = math.hypot(tip.x() - other_tip.x(), tip.y() - other_tip.y())
                if dist < best_dist:
                    best_dist = dist
                    best = (item, other_leg)
        return best

    def try_snap_connections(self, tensor: TensorItem):
        """After placing/moving a tensor, auto-connect nearby legs.
        Repositions the tensor so that each connection has the correct
        bond length — horizontal candidates determine x, vertical
        candidates determine y.
        """
        # Gather all candidate snaps at current position
        candidates: list[tuple[Leg, TensorItem, Leg]] = []
        for leg in tensor.legs:
            if leg.connected_to is not None:
                continue
            target = self.find_snap_target(tensor, leg)
            if target:
                other_tensor, other_leg = target
                candidates.append((leg, other_tensor, other_leg))

        if not candidates:
            return

        # Compute the ideal position from candidates.
        # Each candidate constrains the position on two axes:
        #   - along the connection axis: BOND_LENGTH from the other surface
        #   - perpendicular axis: aligned so the connection is straight
        # Collect all constraints and average them.
        x_values: list[float] = []
        y_values: list[float] = []
        for leg, other_tensor, other_leg in candidates:
            other_surface = other_tensor.scenePos() + other_leg.offset
            if leg.direction in (Direction.LEFT, Direction.RIGHT):
                sign = 1 if leg.direction == Direction.LEFT else -1
                # along axis: x from bond length
                x_values.append(other_surface.x() + sign * BOND_LENGTH - leg.offset.x())
                # perpendicular: y must match so connection is horizontal
                y_values.append(other_surface.y() - leg.offset.y())
            else:
                # along axis: y from bond length
                y_values.append(other_surface.y() + (1 if leg.direction == Direction.UP else -1) * BOND_LENGTH - leg.offset.y())
                # perpendicular: x must match so connection is vertical
                x_values.append(other_surface.x() - leg.offset.x())

        new_x = sum(x_values) / len(x_values) if x_values else tensor.pos().x()
        new_y = sum(y_values) / len(y_values) if y_values else tensor.pos().y()

        tensor._moving_group = True
        tensor.setPos(QPointF(new_x, new_y))
        tensor._moving_group = False

        # Connect all candidates
        for leg, other_tensor, other_leg in candidates:
            if leg.connected_to is None:
                self.add_connection(leg, tensor, other_leg, other_tensor)

    def show_snap_hint(self, pos: QPointF):
        if self._snap_indicator is None:
            self._snap_indicator = QGraphicsEllipseItem(
                -LEG_SNAP_RADIUS, -LEG_SNAP_RADIUS,
                2 * LEG_SNAP_RADIUS, 2 * LEG_SNAP_RADIUS,
            )
            self._snap_indicator.setBrush(QBrush(COLOR_SNAP_HINT))
            self._snap_indicator.setPen(QPen(Qt.PenStyle.NoPen))
            self._snap_indicator.setZValue(-2)
            self.addItem(self._snap_indicator)
        self._snap_indicator.setPos(pos)
        self._snap_indicator.setVisible(True)

    def hide_snap_hint(self):
        if self._snap_indicator:
            self._snap_indicator.setVisible(False)


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------

class TensorView(QGraphicsView):
    def __init__(self, scene: TensorScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self._tool_mode = ToolMode.SELECT
        self._panning = False
        self._pan_start = QPointF()
        self._dragging_tensor = False
        self._drag_group: set[TensorItem] = set()
        self._drag_start_positions: list[tuple[TensorItem, QPointF]] = []
        self._drag_start_connections: set[tuple[int, int]] = set()
        self._clipboard: list[tuple[type, str | None, QPointF, list[tuple[Direction, QPointF]], QColor]] = []

    def set_tool_mode(self, mode: ToolMode):
        self._tool_mode = mode
        if mode == ToolMode.SELECT:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)

    def _tensor_at(self, view_pos) -> TensorItem | None:
        """Return the TensorItem under the given view position, if any."""
        scene_pos = self.mapToScene(view_pos.toPoint())
        for item in self.scene().items(scene_pos):
            if isinstance(item, TensorItem):
                return item
            # also detect clicks on child items (leg lines) of a tensor
            parent = item.parentItem()
            if isinstance(parent, TensorItem):
                return parent
        return None

    # -- mouse events --------------------------------------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._tool_mode != ToolMode.SELECT:
                scene_pos = self.mapToScene(event.position().toPoint())
                # snap to grid
                gx = round(scene_pos.x() / GRID_SIZE) * GRID_SIZE
                gy = round(scene_pos.y() / GRID_SIZE) * GRID_SIZE
                self._place_tensor(QPointF(gx, gy))
                return

            # Select mode: click on tensor = drag tensor, click on empty = pan
            hit = self._tensor_at(event.position())
            if hit is not None:
                shift_held = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
                # record positions and connections before dragging for undo
                self._dragging_tensor = True
                group = hit.connected_group()
                scene_obj = self.scene()
                # Track selection order
                if isinstance(scene_obj, TensorScene):
                    if shift_held:
                        # Add to existing selection and append new tensors to order
                        for item in scene_obj.selectedItems():
                            if isinstance(item, TensorItem):
                                group.add(item)
                        for t in group:
                            if t not in scene_obj.selection_order:
                                scene_obj.selection_order.append(t)
                    else:
                        # Replace selection order with this group
                        scene_obj.selection_order = list(group)
                self._drag_group = group
                self._drag_start_positions = []
                for t in group:
                    self._drag_start_positions.append((t, QPointF(t.pos())))
                # snapshot current connections involving dragged tensors
                scene = self.scene()
                self._drag_start_connections = set()
                if isinstance(scene, TensorScene):
                    for conn in scene.connections:
                        if conn.tensor_a in group or conn.tensor_b in group:
                            self._drag_start_connections.add(
                                (id(conn.leg_a), id(conn.leg_b)))
                # Call super for drag setup, then select entire group
                super().mousePressEvent(event)
                for t in group:
                    t.setSelected(True)
            else:
                # deselect and pan the canvas
                scene = self.scene()
                if isinstance(scene, TensorScene):
                    scene.clearSelection()
                    scene.selection_order.clear()
                self._panning = True
                self._pan_start = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            return
        super().mouseMoveEvent(event)
        # Re-apply group selection after super (Qt resets selection during drag)
        if self._dragging_tensor and self._drag_group:
            for t in self._drag_group:
                if not t.isSelected():
                    t.setSelected(True)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self._panning:
            self._panning = False
            if self._tool_mode == ToolMode.SELECT:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
            return

        if event.button() == Qt.MouseButton.LeftButton:
            if self._panning:
                self._panning = False
                self.setCursor(Qt.CursorShape.OpenHandCursor)
                return

            # after drag-move of a tensor in select mode, try snapping and record undo
            if self._tool_mode == ToolMode.SELECT and self._drag_start_positions:
                scene = self.scene()
                if isinstance(scene, TensorScene):
                    dragged = {t for t, _ in self._drag_start_positions}

                    # detect connections broken by the drag (were connected
                    # before, but the group move pulled them apart)
                    conns_broken: list[tuple[Leg, TensorItem, Leg, TensorItem]] = []
                    for conn in list(scene.connections):
                        key = (id(conn.leg_a), id(conn.leg_b))
                        if key in self._drag_start_connections:
                            # connection existed before — check if one end
                            # is in the dragged group and the other is not
                            a_in = conn.tensor_a in dragged
                            b_in = conn.tensor_b in dragged
                            if a_in != b_in:
                                conns_broken.append(
                                    (conn.leg_a, conn.tensor_a,
                                     conn.leg_b, conn.tensor_b))
                                scene._remove_connection(conn.leg_a, conn.leg_b)

                    # snapshot connections before snap
                    conns_before = {(id(c.leg_a), id(c.leg_b))
                                    for c in scene.connections}

                    for item in scene.selectedItems():
                        if isinstance(item, TensorItem):
                            scene.try_snap_connections(item)

                    # detect connections formed by the snap
                    conns_formed: list[tuple[Leg, TensorItem, Leg, TensorItem]] = []
                    for conn in scene.connections:
                        key = (id(conn.leg_a), id(conn.leg_b))
                        if key not in conns_before:
                            conns_formed.append(
                                (conn.leg_a, conn.tensor_a,
                                 conn.leg_b, conn.tensor_b))

                    new_positions = [(t, QPointF(t.pos())) for t, _ in self._drag_start_positions]
                    moved = any(
                        (old.x() != new.x() or old.y() != new.y())
                        for (_, old), (_, new) in zip(self._drag_start_positions, new_positions)
                    )
                    if moved or conns_formed or conns_broken:
                        scene.undo_stack.push(MoveAction(
                            scene,
                            self._drag_start_positions,
                            new_positions,
                            conns_formed,
                            conns_broken,
                        ))
                self._drag_start_positions = []
                self._drag_start_connections = set()
                self._dragging_tensor = False

        release_group = self._drag_group
        self._drag_group = set()
        super().mouseReleaseEvent(event)

        # After super processes the release, re-apply group selection
        if event.button() == Qt.MouseButton.LeftButton and release_group:
            for t in release_group:
                t.setSelected(True)

    def wheelEvent(self, event):
        factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(factor, factor)
        else:
            self.scale(1.0 / factor, 1.0 / factor)

    def keyPressEvent(self, event):
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15
            if event.key() in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
                self.scale(factor, factor)
                return
            if event.key() == Qt.Key.Key_Minus:
                self.scale(1.0 / factor, 1.0 / factor)
                return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            scene = self.scene()
            if isinstance(scene, TensorScene):
                tensors = [item for item in scene.selectedItems() if isinstance(item, TensorItem)]
                if tensors:
                    # snapshot connections involving any of the deleted tensors
                    conn_info = []
                    for conn in list(scene.connections):
                        if conn.tensor_a in tensors or conn.tensor_b in tensors:
                            conn_info.append((conn.leg_a, conn.tensor_a, conn.leg_b, conn.tensor_b))
                    positions = [QPointF(t.pos()) for t in tensors]
                    for t in tensors:
                        scene.remove_connections_for(t)
                        scene.removeItem(t)
                    scene.undo_stack.push(DeleteAction(scene, tensors, conn_info, positions))
            return
        if event.key() in (Qt.Key.Key_BracketLeft, Qt.Key.Key_BracketRight):
            scene = self.scene()
            if isinstance(scene, TensorScene):
                tensors = [item for item in scene.selectedItems() if isinstance(item, TensorItem)]
                if tensors:
                    clockwise = event.key() == Qt.Key.Key_BracketRight
                    for t in tensors:
                        # snapshot connections before rotating
                        conn_info = [
                            (c.leg_a, c.tensor_a, c.leg_b, c.tensor_b)
                            for c in scene.connections
                            if c.tensor_a is t or c.tensor_b is t
                        ]
                        scene.remove_connections_for(t)
                        t.rotate_90(clockwise)
                        scene.undo_stack.push(RotateAction(scene, t, clockwise, conn_info))
            return
        if event.key() == Qt.Key.Key_C and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            scene = self.scene()
            if isinstance(scene, TensorScene):
                tensors = [item for item in scene.selectedItems() if isinstance(item, TensorItem)]
                self._clipboard.clear()
                for t in tensors:
                    side = t.side if isinstance(t, BoundaryTensor) else None
                    leg_state = [(leg.direction, QPointF(leg.offset)) for leg in t.legs]
                    self._clipboard.append((type(t), side, QPointF(t.pos()), leg_state, QColor(t.color)))
            return
        if event.key() == Qt.Key.Key_V and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            scene = self.scene()
            if isinstance(scene, TensorScene) and self._clipboard:
                scene.clearSelection()
                # Compute centroid of copied positions
                cx = sum(pos.x() for _, _, pos, _, _ in self._clipboard) / len(self._clipboard)
                cy = sum(pos.y() for _, _, pos, _, _ in self._clipboard) / len(self._clipboard)
                centroid = QPointF(cx, cy)
                # Place at cursor position if cursor is over the viewport
                cursor_view = self.mapFromGlobal(self.cursor().pos())
                if self.rect().contains(cursor_view):
                    target = self.mapToScene(cursor_view)
                else:
                    target = centroid + QPointF(GRID_SIZE, GRID_SIZE)
                offset = target - centroid
                new_tensors = []
                for cls, side, pos, leg_state, color in self._clipboard:
                    if cls is BoundaryTensor:
                        tensor = BoundaryTensor(side)
                    else:
                        tensor = cls()
                    # Apply stored leg directions/offsets (preserves rotation) and color
                    for leg, (direction, leg_offset) in zip(tensor.legs, leg_state):
                        leg.direction = direction
                        leg.offset = QPointF(leg_offset)
                    tensor.color = QColor(color)
                    tensor.setPos(pos + offset)
                    tensor._update_leg_lines()
                    scene.addItem(tensor)
                    tensor.setSelected(True)
                    new_tensors.append(tensor)
                # Try snapping each new tensor
                all_conn_info = []
                for tensor in new_tensors:
                    conn_before = len(scene.connections)
                    scene.try_snap_connections(tensor)
                    new_conns = scene.connections[conn_before:]
                    all_conn_info.extend([(c.leg_a, c.tensor_a, c.leg_b, c.tensor_b) for c in new_conns])
                for tensor in new_tensors:
                    scene.undo_stack.push(PlaceAction(scene, tensor, all_conn_info))
                    all_conn_info = []  # only first push gets the connections
            return
        if event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            scene = self.scene()
            if isinstance(scene, TensorScene):
                scene.undo_stack.undo()
            return
        if event.key() == Qt.Key.Key_Y and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            scene = self.scene()
            if isinstance(scene, TensorScene):
                scene.undo_stack.redo()
            return
        super().keyPressEvent(event)

    # -- placement -----------------------------------------------------------
    def _place_tensor(self, pos: QPointF):
        scene = self.scene()
        if not isinstance(scene, TensorScene):
            return
        tensor: TensorItem | None = None
        if self._tool_mode == ToolMode.MPS:
            tensor = MPSSiteTensor()
        elif self._tool_mode == ToolMode.MPS_LEFT:
            tensor = LeftOrthoMPSTensor()
        elif self._tool_mode == ToolMode.MPS_RIGHT:
            tensor = RightOrthoMPSTensor()
        elif self._tool_mode == ToolMode.MPO:
            tensor = MPOSiteTensor()
        elif self._tool_mode == ToolMode.BOUNDARY_LEFT:
            tensor = BoundaryTensor("left")
        elif self._tool_mode == ToolMode.BOUNDARY_RIGHT:
            tensor = BoundaryTensor("right")

        if tensor:
            tensor.setPos(pos)
            scene.addItem(tensor)
            conn_before = len(scene.connections)
            scene.try_snap_connections(tensor)
            # snapshot connections formed during this placement
            new_conns = scene.connections[conn_before:]
            conn_info = [(c.leg_a, c.tensor_a, c.leg_b, c.tensor_b) for c in new_conns]
            scene.undo_stack.push(PlaceAction(scene, tensor, conn_info))


# ---------------------------------------------------------------------------
# Main Window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tensor Network Diagram Editor")
        self.resize(1400, 700)

        # Scene & View
        self.scene = TensorScene(self)
        self.view = TensorView(self.scene, self)
        self.setCentralWidget(self.view)

        # Toolbar
        self._build_toolbar()

        # Status bar
        self.statusBar().showMessage("Select a tool, then click on the canvas to place tensors.  "
                                     "Middle-click to pan.  Scroll to zoom.  [ / ] to rotate.  Delete removes selected.")

    def _build_toolbar(self):
        toolbar = QToolBar("Tools", self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._tool_group = QButtonGroup(self)
        self._tool_group.setExclusive(True)

        def make_btn(text, mode: ToolMode, tooltip: str):
            btn = QToolButton()
            btn.setText(text)
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.setMinimumWidth(90)
            font = btn.font()
            font.setPointSize(10)
            btn.setFont(font)
            btn.clicked.connect(lambda checked, m=mode: self._set_mode(m))
            toolbar.addWidget(btn)
            self._tool_group.addButton(btn)
            return btn

        select_btn = make_btn("\u2190 Select", ToolMode.SELECT, "Select & move tensors (S)")
        select_btn.setChecked(True)

        make_btn("\u25CB MPS", ToolMode.MPS, "MPS site tensor: circle with 3 legs (M)")
        make_btn("\u25B7 MPS-L", ToolMode.MPS_LEFT, "Left-orthogonal MPS: right-pointing triangle (A)")
        make_btn("\u25C1 MPS-R", ToolMode.MPS_RIGHT, "Right-orthogonal MPS: left-pointing triangle (B)")
        make_btn("\u25A1 MPO", ToolMode.MPO, "MPO site tensor: square with 4 legs (O)")
        make_btn("\u25B6| Left", ToolMode.BOUNDARY_LEFT, "Left boundary tensor (L)")
        make_btn("|\u25C0 Right", ToolMode.BOUNDARY_RIGHT, "Right boundary tensor (R)")

        toolbar.addSeparator()

        rotate_ccw_btn = QToolButton()
        rotate_ccw_btn.setText("\u21B6")
        rotate_ccw_btn.setToolTip("Rotate selected 90° counter-clockwise ([)")
        rotate_ccw_btn.setMinimumWidth(70)
        font = rotate_ccw_btn.font()
        font.setPointSize(10)
        rotate_ccw_btn.setFont(font)
        rotate_ccw_btn.clicked.connect(lambda: self._rotate_selected(clockwise=False))
        toolbar.addWidget(rotate_ccw_btn)

        rotate_cw_btn = QToolButton()
        rotate_cw_btn.setText("\u21B7")
        rotate_cw_btn.setToolTip("Rotate selected 90° clockwise (])")
        rotate_cw_btn.setMinimumWidth(70)
        font = rotate_cw_btn.font()
        font.setPointSize(10)
        rotate_cw_btn.setFont(font)
        rotate_cw_btn.clicked.connect(lambda: self._rotate_selected(clockwise=True))
        toolbar.addWidget(rotate_cw_btn)

        toolbar.addSeparator()

        color_btn = QToolButton()
        color_btn.setText("\u25A0 Color")
        color_btn.setToolTip("Change color of selected tensors")
        color_btn.setMinimumWidth(80)
        font = color_btn.font()
        font.setPointSize(10)
        color_btn.setFont(font)
        color_btn.clicked.connect(self._change_color)
        toolbar.addWidget(color_btn)

        toolbar.addSeparator()

        align_btn = QToolButton()
        align_btn.setText("\u2261 Align")
        align_btn.setToolTip("Align selected groups")
        align_btn.setMinimumWidth(80)
        font = align_btn.font()
        font.setPointSize(10)
        align_btn.setFont(font)
        align_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        align_menu = QMenu(self)
        align_menu.addAction("Align Left Edges", lambda: self._align_selected("left"))
        align_menu.addAction("Align Right Edges", lambda: self._align_selected("right"))
        align_menu.addAction("Align Center Vertically", lambda: self._align_selected("center_v"))
        align_menu.addSeparator()
        align_menu.addAction("Align Top Edges", lambda: self._align_selected("top"))
        align_menu.addAction("Align Bottom Edges", lambda: self._align_selected("bottom"))
        align_menu.addAction("Align Center Horizontally", lambda: self._align_selected("center_h"))
        align_btn.setMenu(align_menu)
        toolbar.addWidget(align_btn)

        toolbar.addSeparator()

        file_save_btn = QToolButton()
        file_save_btn.setText("Save")
        file_save_btn.setToolTip("Save diagram to file (Ctrl+S)")
        file_save_btn.setMinimumWidth(60)
        font = file_save_btn.font()
        font.setPointSize(10)
        file_save_btn.setFont(font)
        file_save_btn.clicked.connect(self._save_diagram)
        toolbar.addWidget(file_save_btn)

        file_load_btn = QToolButton()
        file_load_btn.setText("Load")
        file_load_btn.setToolTip("Load diagram from file (Ctrl+O)")
        file_load_btn.setMinimumWidth(60)
        font = file_load_btn.font()
        font.setPointSize(10)
        file_load_btn.setFont(font)
        file_load_btn.clicked.connect(self._load_diagram)
        toolbar.addWidget(file_load_btn)

        export_btn = QToolButton()
        export_btn.setText("Export")
        export_btn.setToolTip("Export diagram as image (Ctrl+Shift+S)")
        export_btn.setMinimumWidth(70)
        font = export_btn.font()
        font.setPointSize(10)
        export_btn.setFont(font)
        export_btn.clicked.connect(self._save_image)
        toolbar.addWidget(export_btn)

        save_file_action = QAction(self)
        save_file_action.setShortcut(QKeySequence("Ctrl+S"))
        save_file_action.triggered.connect(self._save_diagram)
        self.addAction(save_file_action)

        load_file_action = QAction(self)
        load_file_action.setShortcut(QKeySequence("Ctrl+O"))
        load_file_action.triggered.connect(self._load_diagram)
        self.addAction(load_file_action)

        export_action = QAction(self)
        export_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        export_action.triggered.connect(self._save_image)
        self.addAction(export_action)

        # keyboard shortcuts
        shortcuts = {
            Qt.Key.Key_S: ToolMode.SELECT,
            Qt.Key.Key_M: ToolMode.MPS,
            Qt.Key.Key_A: ToolMode.MPS_LEFT,
            Qt.Key.Key_B: ToolMode.MPS_RIGHT,
            Qt.Key.Key_O: ToolMode.MPO,
            Qt.Key.Key_L: ToolMode.BOUNDARY_LEFT,
            Qt.Key.Key_R: ToolMode.BOUNDARY_RIGHT,
        }
        for key, mode in shortcuts.items():
            action = QAction(self)
            action.setShortcut(key)
            action.triggered.connect(lambda checked, m=mode: self._set_mode(m))
            self.addAction(action)

    # -- diagram file save/load ------------------------------------------------

    _TYPE_MAP = {
        "MPS": MPSSiteTensor,
        "MPS_LEFT": LeftOrthoMPSTensor,
        "MPS_RIGHT": RightOrthoMPSTensor,
        "MPO": MPOSiteTensor,
        "BOUNDARY": BoundaryTensor,
    }
    _TYPE_MAP_REV = {v: k for k, v in _TYPE_MAP.items()}

    def _save_diagram(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Diagram", "", "MPS Diagram (*.mpsd);;JSON (*.json)")
        if not path:
            return

        tensors = [item for item in self.scene.items() if isinstance(item, TensorItem)]
        # Assign stable IDs for serialization
        tensor_ids = {id(t): i for i, t in enumerate(tensors)}

        tensor_data = []
        for t in tensors:
            type_name = self._TYPE_MAP_REV.get(type(t), "MPS")
            entry = {
                "id": tensor_ids[id(t)],
                "type": type_name,
                "x": t.pos().x(),
                "y": t.pos().y(),
                "color": t.color.name(),
                "legs": [
                    {"direction": leg.direction.name, "ox": leg.offset.x(), "oy": leg.offset.y()}
                    for leg in t.legs
                ],
            }
            if isinstance(t, BoundaryTensor):
                entry["side"] = t.side
            tensor_data.append(entry)

        connection_data = []
        for conn in self.scene.connections:
            ta_id = tensor_ids.get(id(conn.tensor_a))
            tb_id = tensor_ids.get(id(conn.tensor_b))
            if ta_id is not None and tb_id is not None:
                la_idx = conn.tensor_a.legs.index(conn.leg_a)
                lb_idx = conn.tensor_b.legs.index(conn.leg_b)
                connection_data.append({
                    "tensor_a": ta_id, "leg_a": la_idx,
                    "tensor_b": tb_id, "leg_b": lb_idx,
                })

        data = {"tensors": tensor_data, "connections": connection_data}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        self.statusBar().showMessage(f"Diagram saved to {path}", 5000)

    def _load_diagram(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Diagram", "", "MPS Diagram (*.mpsd);;JSON (*.json);;All Files (*)")
        if not path:
            return

        with open(path, "r") as f:
            data = json.load(f)

        # Clear existing diagram
        self.scene.clearSelection()
        for conn in list(self.scene.connections):
            self.scene._remove_connection(conn.leg_a, conn.leg_b)
        for item in list(self.scene.items()):
            if isinstance(item, TensorItem):
                self.scene.removeItem(item)
        self.scene.undo_stack = UndoStack()
        self.scene.selection_order.clear()

        # Recreate tensors
        id_to_tensor: dict[int, TensorItem] = {}
        for entry in data["tensors"]:
            type_name = entry["type"]
            cls = self._TYPE_MAP.get(type_name)
            if cls is None:
                continue
            if cls is BoundaryTensor:
                tensor = BoundaryTensor(entry.get("side", "left"))
            else:
                tensor = cls()
            tensor.setPos(QPointF(entry["x"], entry["y"]))
            tensor.color = QColor(entry["color"])
            # Restore leg directions and offsets (preserves rotation)
            for leg, leg_data in zip(tensor.legs, entry["legs"]):
                leg.direction = Direction[leg_data["direction"]]
                leg.offset = QPointF(leg_data["ox"], leg_data["oy"])
            tensor._update_leg_lines()
            self.scene.addItem(tensor)
            id_to_tensor[entry["id"]] = tensor

        # Recreate connections
        for conn_data in data["connections"]:
            ta = id_to_tensor.get(conn_data["tensor_a"])
            tb = id_to_tensor.get(conn_data["tensor_b"])
            if ta is not None and tb is not None:
                la = ta.legs[conn_data["leg_a"]]
                lb = tb.legs[conn_data["leg_b"]]
                self.scene.add_connection(la, ta, lb, tb)

        self.statusBar().showMessage(f"Diagram loaded from {path}", 5000)

    # -- image export ----------------------------------------------------------

    def _save_image(self):
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Diagram As Image", "",
            "PNG Image (*.png);;JPEG Image (*.jpg);;SVG Image (*.svg);;PDF Document (*.pdf)"
        )
        if not path:
            return

        # Clear selection so dashed outlines don't appear in export
        self.scene.clearSelection()

        # Suppress grid lines during export
        self.scene._exporting = True
        try:
            # Tight bounding rect around all items with a small margin
            rect = self.scene.itemsBoundingRect()
            margin = 8
            rect.adjust(-margin, -margin, margin, margin)
            target = QRectF(0, 0, rect.width(), rect.height())

            if path.lower().endswith(".svg"):
                from PySide6.QtSvg import QSvgGenerator
                generator = QSvgGenerator()
                generator.setFileName(path)
                generator.setSize(rect.size().toSize())
                generator.setViewBox(target)
                painter = QPainter(generator)
                painter.fillRect(target, Qt.GlobalColor.white)
                self.scene.render(painter, target, rect)
                painter.end()
            elif path.lower().endswith(".pdf"):
                from PySide6.QtCore import QMarginsF, QSizeF
                from PySide6.QtGui import QPageSize
                from PySide6.QtPrintSupport import QPrinter
                printer = QPrinter()
                printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
                printer.setOutputFileName(path)
                # Use exact content size as page size (in points)
                page_size = QPageSize(QSizeF(rect.width(), rect.height()), QPageSize.Unit.Point)
                printer.setPageSize(page_size)
                printer.setPageMargins(QMarginsF(0, 0, 0, 0))
                painter = QPainter(printer)
                page_rect = printer.pageRect(QPrinter.Unit.DevicePixel)
                painter.fillRect(page_rect, Qt.GlobalColor.white)
                self.scene.render(painter, page_rect, rect)
                painter.end()
            else:
                # Raster (PNG/JPG) at 2x resolution for crisp output
                scale = 2
                image = QImage(
                    int(rect.width() * scale), int(rect.height() * scale),
                    QImage.Format.Format_ARGB32_Premultiplied
                )
                image.fill(Qt.GlobalColor.white)
                painter = QPainter(image)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                self.scene.render(painter, QRectF(0, 0, image.width(), image.height()), rect)
                painter.end()
                image.save(path)
        finally:
            self.scene._exporting = False

        self.statusBar().showMessage(f"Saved to {path}", 5000)

    def _align_selected(self, mode: str):
        """Align selected connected groups.
        mode: 'left', 'right', 'top', 'bottom', 'center_h', 'center_v'."""
        # Use selection order to determine the reference (first) group
        ordered = self.scene.selection_order if self.scene.selection_order else [
            item for item in self.scene.selectedItems() if isinstance(item, TensorItem)
        ]
        if not ordered:
            return

        # Partition into distinct connected groups, preserving selection order
        visited: set[TensorItem] = set()
        groups: list[set[TensorItem]] = []
        for t in ordered:
            if t not in visited:
                group = t.connected_group()
                groups.append(group)
                visited.update(group)

        if len(groups) < 2:
            return

        # Compute bounding box for each group
        def group_bbox(group):
            xs, ys = [], []
            for t in group:
                r = t._shape_rect()
                xs.append(t.pos().x() + r.left())
                xs.append(t.pos().x() + r.right())
                ys.append(t.pos().y() + r.top())
                ys.append(t.pos().y() + r.bottom())
            return min(xs), max(xs), min(ys), max(ys)

        bboxes = [group_bbox(g) for g in groups]

        # Compute the target value based on the first group (reference)
        ref = bboxes[0]
        ref_left, ref_right, ref_top, ref_bottom = ref

        # Compute delta for each group (skip the first — it's the reference)
        all_old = []
        all_new = []
        for group, (g_left, g_right, g_top, g_bottom) in zip(groups[1:], bboxes[1:]):
            dx, dy = 0.0, 0.0
            if mode == "left":
                dx = ref_left - g_left
            elif mode == "right":
                dx = ref_right - g_right
            elif mode == "center_v":
                ref_cx = (ref_left + ref_right) / 2
                g_cx = (g_left + g_right) / 2
                dx = ref_cx - g_cx
            elif mode == "top":
                dy = ref_top - g_top
            elif mode == "bottom":
                dy = ref_bottom - g_bottom
            elif mode == "center_h":
                ref_cy = (ref_top + ref_bottom) / 2
                g_cy = (g_top + g_bottom) / 2
                dy = ref_cy - g_cy

            if dx == 0 and dy == 0:
                continue
            delta = QPointF(dx, dy)
            # Guard against itemChange triggering recursive group movement
            for t in group:
                all_old.append((t, QPointF(t.pos())))
                t._moving_group = True
            for t in group:
                t.setPos(t.pos() + delta)
            for t in group:
                t._moving_group = False
                all_new.append((t, QPointF(t.pos())))

        if all_old:
            self.scene.undo_stack.push(MoveAction(
                self.scene, all_old, all_new, [], []))

    def _change_color(self):
        tensors = [item for item in self.scene.selectedItems() if isinstance(item, TensorItem)]
        if not tensors:
            return
        # Use the first selected tensor's color as the initial color
        initial = tensors[0].color
        color = QColorDialog.getColor(initial, self, "Choose Tensor Color")
        if color.isValid():
            old_colors = [QColor(t.color) for t in tensors]
            for t in tensors:
                t.color = QColor(color)
                t.update()
            self.scene.undo_stack.push(ColorAction(tensors, old_colors, color))

    def _rotate_selected(self, clockwise: bool):
        tensors = [item for item in self.scene.selectedItems() if isinstance(item, TensorItem)]
        if tensors:
            for t in tensors:
                conn_info = [
                    (c.leg_a, c.tensor_a, c.leg_b, c.tensor_b)
                    for c in self.scene.connections
                    if c.tensor_a is t or c.tensor_b is t
                ]
                self.scene.remove_connections_for(t)
                t.rotate_90(clockwise)
                self.scene.undo_stack.push(RotateAction(self.scene, t, clockwise, conn_info))

    def _set_mode(self, mode: ToolMode):
        self.view.set_tool_mode(mode)
        labels = {
            ToolMode.SELECT: "Select mode — click to select, drag to move",
            ToolMode.MPS: "MPS mode — click to place MPS site tensor (circle, 3 legs)",
            ToolMode.MPS_LEFT: "Left-orthogonal MPS mode — click to place right-pointing triangle (3 legs)",
            ToolMode.MPS_RIGHT: "Right-orthogonal MPS mode — click to place left-pointing triangle (3 legs)",
            ToolMode.MPO: "MPO mode — click to place MPO site tensor (square, 4 legs)",
            ToolMode.BOUNDARY_LEFT: "Left boundary mode — click to place left boundary tensor",
            ToolMode.BOUNDARY_RIGHT: "Right boundary mode — click to place right boundary tensor",
        }
        self.statusBar().showMessage(labels.get(mode, ""))
        # update button checked state
        buttons = self._tool_group.buttons()
        modes = [ToolMode.SELECT, ToolMode.MPS, ToolMode.MPS_LEFT, ToolMode.MPS_RIGHT,
                 ToolMode.MPO, ToolMode.BOUNDARY_LEFT, ToolMode.BOUNDARY_RIGHT]
        for btn, m in zip(buttons, modes):
            btn.setChecked(m == mode)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
