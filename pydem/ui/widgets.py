# pydem/ui/widgets.py
"""
UI Widgets
----------
Custom widgets for the PyDEM user interface.
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, 
    QSpinBox, QDoubleSpinBox, QPushButton, QGroupBox,
    QFormLayout, QCheckBox, QComboBox, QLineEdit,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QColor, QFont, QPen, QBrush

class OpenGLWidget(QWidget):
    """
    Placeholder for OpenGL rendering widget.
    This will eventually be replaced with a real OpenGL widget.
    """
    
    def __init__(self, parent=None):
        """Initialize widget."""
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        
    def paintEvent(self, event):
        """Handle paint event."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # Draw text
        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 12))
        painter.drawText(
            self.rect(),
            Qt.AlignCenter,
            "OpenGL Visualization Placeholder\n(Will be replaced with real OpenGL rendering)"
        )


class ParticleTable(QTableWidget):
    """Table widget for displaying particle properties."""
    
    def __init__(self, parent=None):
        """Initialize widget."""
        super().__init__(parent)
        
        # Set up table
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["ID", "Position", "Velocity", "Mass", "Radius"])
        
        # Adjust column widths
        header = self.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        
        # Set selection behavior
        self.setSelectionBehavior(QTableWidget.SelectRows)
        self.setSelectionMode(QTableWidget.SingleSelection)
        
    def update_data(self, particles):
        """
        Update table with particle data.
        
        Args:
            particles: List of particle objects
        """
        self.setRowCount(len(particles))
        
        for i, p in enumerate(particles):
            # ID
            self.setItem(i, 0, QTableWidgetItem(str(p.getId())))
            
            # Position
            pos = p.getPos()
            pos_str = f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            self.setItem(i, 1, QTableWidgetItem(pos_str))
            
            # Velocity
            vel = p.getVel()
            vel_str = f"({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f})"
            self.setItem(i, 2, QTableWidgetItem(vel_str))
            
            # Mass
            self.setItem(i, 3, QTableWidgetItem(f"{p.getMass():.2f}"))
            
            # Radius (if available)
            radius = p.getShape().radius if hasattr(p.getShape(), "radius") else "N/A"
            self.setItem(i, 4, QTableWidgetItem(str(radius)))


class EnergyPlot(QWidget):
    """Widget for plotting energy evolution."""
    
    def __init__(self, parent=None):
        """Initialize widget."""
        super().__init__(parent)
        
        self.setMinimumSize(300, 200)
        
        # Plot data
        self.times = []
        self.kinetic_energy = []
        self.potential_energy = []
        self.total_energy = []
        self.max_energy = 1.0
        
    def update_data(self, time, kinetic, potential, total):
        """
        Update plot data.
        
        Args:
            time: Current simulation time
            kinetic: Kinetic energy value
            potential: Potential energy value
            total: Total energy value
        """
        self.times.append(time)
        self.kinetic_energy.append(kinetic)
        self.potential_energy.append(potential)
        self.total_energy.append(total)
        
        # Keep only the last 100 points
        if len(self.times) > 100:
            self.times = self.times[-100:]
            self.kinetic_energy = self.kinetic_energy[-100:]
            self.potential_energy = self.potential_energy[-100:]
            self.total_energy = self.total_energy[-100:]
        
        # Update maximum energy value
        self.max_energy = max(max(self.total_energy), 1.0)
        
        # Trigger repaint
        self.update()
        
    def paintEvent(self, event):
        """Handle paint event."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background
        painter.fillRect(self.rect(), QColor(255, 255, 255))
        
        # Draw border
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        
        if not self.times:
            # Draw "No data" message
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(
                self.rect(),
                Qt.AlignCenter,
                "No energy data available"
            )
            return
        
        # Calculate plot dimensions
        margin = 30
        plot_width = self.width() - 2 * margin
        plot_height = self.height() - 2 * margin
        
        # Draw axes
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.drawLine(margin, self.height() - margin, margin, margin)  # Y-axis
        painter.drawLine(margin, self.height() - margin, self.width() - margin, self.height() - margin)  # X-axis
        
        # Draw axis labels
        painter.setPen(QColor(0, 0, 0))
        painter.setFont(QFont("Arial", 8))
        painter.drawText(5, margin, "Energy")
        painter.drawText(self.width() - margin, self.height() - 5, "Time")
        
        # Draw scale marks
        painter.setPen(QPen(QColor(100, 100, 100), 0.5, Qt.DashLine))
        
        # X-axis marks
        for i in range(5):
            x = margin + i * plot_width / 4
            painter.drawLine(x, self.height() - margin, x, margin)
            
        # Y-axis marks
        for i in range(5):
            y = self.height() - margin - i * plot_height / 4
            painter.drawLine(margin, y, self.width() - margin, y)
        
        # Draw energy plots
        if len(self.times) > 1:
            # Time scale
            min_time = self.times[0]
            max_time = self.times[-1]
            time_range = max(max_time - min_time, 0.1)
            
            # Draw total energy
            painter.setPen(QPen(QColor(0, 0, 0), 2))
            path = []
            
            for i, t in enumerate(self.times):
                x = margin + (t - min_time) / time_range * plot_width
                y = self.height() - margin - self.total_energy[i] / self.max_energy * plot_height
                path.append((x, y))
                
            for i in range(len(path) - 1):
                painter.drawLine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            
            # Draw kinetic energy
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            path = []
            
            for i, t in enumerate(self.times):
                x = margin + (t - min_time) / time_range * plot_width
                y = self.height() - margin - self.kinetic_energy[i] / self.max_energy * plot_height
                path.append((x, y))
                
            for i in range(len(path) - 1):
                painter.drawLine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
            
            # Draw potential energy
            painter.setPen(QPen(QColor(0, 0, 255), 2))
            path = []
            
            for i, t in enumerate(self.times):
                x = margin + (t - min_time) / time_range * plot_width
                y = self.height() - margin - self.potential_energy[i] / self.max_energy * plot_height
                path.append((x, y))
                
            for i in range(len(path) - 1):
                painter.drawLine(path[i][0], path[i][1], path[i+1][0], path[i+1][1])
        
        # Draw legend
        legend_width = 120
        legend_height = 70
        legend_x = self.width() - legend_width - 10
        legend_y = 10
        
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        painter.setBrush(QBrush(QColor(255, 255, 255, 200)))
        painter.drawRect(legend_x, legend_y, legend_width, legend_height)
        
        # Legend items
        painter.setFont(QFont("Arial", 8))
        
        # Total energy
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.drawLine(legend_x + 10, legend_y + 15, legend_x + 30, legend_y + 15)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(legend_x + 40, legend_y + 20, "Total Energy")
        
        # Kinetic energy
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.drawLine(legend_x + 10, legend_y + 35, legend_x + 30, legend_y + 35)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(legend_x + 40, legend_y + 40, "Kinetic Energy")
        
        # Potential energy
        painter.setPen(QPen(QColor(0, 0, 255), 2))
        painter.drawLine(legend_x + 10, legend_y + 55, legend_x + 30, legend_y + 55)
        painter.setPen(QColor(0, 0, 0))
        painter.drawText(legend_x + 40, legend_y + 60, "Potential Energy")


class ParticleInspector(QWidget):
    """Widget for inspecting particle properties."""
    
    def __init__(self, parent=None):
        """Initialize widget."""
        super().__init__(parent)
        
        # Create layout
        layout = QVBoxLayout(self)
        
        # Create form layout for properties
        form_layout = QFormLayout()
        
        # ID
        self.id_label = QLabel("None")
        form_layout.addRow("ID:", self.id_label)
        
        # Position
        self.pos_label = QLabel("None")
        form_layout.addRow("Position:", self.pos_label)
        
        # Velocity
        self.vel_label = QLabel("None")
        form_layout.addRow("Velocity:",