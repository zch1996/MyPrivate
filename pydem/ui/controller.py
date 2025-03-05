# pydem/ui/controller.py
"""
UI Controller
------------
Main controller for the PyDEM user interface.
"""

import logging
import threading
import queue
import time
import sys
import os
from datetime import datetime

try:
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QPushButton,
        QLabel,
        QSlider,
        QSpinBox,
        QDoubleSpinBox,
        QFrame,
        QSplitter,
        QGroupBox,
        QStatusBar,
        QAction,
        QFileDialog,
        QDockWidget,
        QTabWidget,
        QTextEdit,
        QProgressBar,
    )
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
    from PyQt5.QtGui import QIcon, QPixmap, QFont

    HAS_QT = True
except ImportError:
    logging.warning("PyQt5 not found. UI will be disabled.")
    HAS_QT = False

logger = logging.getLogger("PyDEM.UI")


class UIController:
    """Controller for the PyDEM user interface."""

    def __init__(
        self,
        omega=None,
        sim_cmd_queue=None,
        sim_status_queue=None,
        vis_cmd_queue=None,
        vis_data_queue=None,
    ):
        """
        Initialize UI controller.

        Args:
            omega: Omega instance, if None will use global instance
            sim_cmd_queue: Queue for sending commands to simulation process
            sim_status_queue: Queue for receiving status from simulation process
            vis_cmd_queue: Queue for sending commands to visualization process
            vis_data_queue: Queue for sending data to visualization process
        """
        # Get Omega instance if not provided
        if omega is None:
            from pydem import Omega

            self.omega = Omega.instance()
        else:
            self.omega = omega

        # Get scene
        self.scene = self.omega.getScene()

        # Store queues for inter-process communication
        self.sim_cmd_queue = sim_cmd_queue
        self.sim_status_queue = sim_status_queue
        self.vis_cmd_queue = vis_cmd_queue
        self.vis_data_queue = vis_data_queue

        # For single-process operation
        self.use_queues = sim_cmd_queue is not None

        # Initialize UI if PyQt5 is available
        self.app = None
        self.main_window = None

        if HAS_QT:
            self._initialize_ui()
        else:
            logger.warning("PyQt5 not available, UI disabled")

    def _initialize_ui(self):
        """Initialize the PyQt5 UI."""
        # Create QApplication if not already running
        if QApplication.instance() is None:
            self.app = QApplication(sys.argv)
        else:
            self.app = QApplication.instance()

        # Create main window
        self.main_window = MainWindow(self)

        # Set up status update timer
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_status)
        self.status_timer.start(100)  # Update every 100ms

        # Show the main window
        self.main_window.show()

    def _update_status(self):
        """Update status display with latest simulation data."""
        if self.use_queues and self.sim_status_queue:
            # Process all pending status messages
            while True:
                try:
                    msg_type, data = self.sim_status_queue.get(block=False)

                    if msg_type == "STATUS" or msg_type == "STATS":
                        self.main_window.update_stats(data)
                    elif msg_type == "ERROR" or msg_type == "FATAL":
                        self.main_window.show_error(data)

                    self.sim_status_queue.task_done()

                except queue.Empty:
                    break
        else:
            # Direct access to scene
            stats = {
                "step": self.scene.step,
                "time": self.scene.time,
                "particles": len(self.scene.field.getParticles()),
                "contacts": self.scene.field.getContacts().size(),
                "running": self.scene.isRunning(),
            }
            self.main_window.update_stats(stats)

    def run(self):
        """Run the UI main loop."""
        if HAS_QT and self.app:
            # Start the PyQt event loop
            return self.app.exec_()
        return 0

    def start_simulation(self):
        """Start the simulation."""
        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("START", None))
        else:
            self.scene.run(wait=False)

    def stop_simulation(self):
        """Stop the simulation."""
        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("STOP", None))
        else:
            self.scene.stop()

    def step_simulation(self):
        """Execute a single simulation step."""
        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("STEP", None))
        else:
            self.scene.oneStep()

    def load_script(self, script_path):
        """Load a simulation script."""
        if not script_path:
            return False

        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("LOAD_SCRIPT", script_path))
        else:
            try:
                # First reset the scene
                self.omega.reset()
                self.scene = self.omega.getScene()

                # Load the script
                script_dir = os.path.dirname(script_path)
                script_name = os.path.basename(script_path)

                # Add script directory to path if not already there
                if script_dir not in sys.path:
                    sys.path.insert(0, script_dir)

                # Load module
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    os.path.splitext(script_name)[0], script_path
                )
                if spec is None:
                    logger.error(f"Could not find script: {script_path}")
                    return False

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                logger.info(f"Script loaded successfully: {script_path}")
                return True

            except Exception as e:
                logger.error(f"Error loading script: {str(e)}")
                return False

    def reset_simulation(self):
        """Reset the simulation to initial state."""
        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("RESET", None))
        else:
            self.omega.reset()
            self.scene = self.omega.getScene()

    def set_camera(self, position=None, target=None):
        """Set camera position and target."""
        if self.use_queues and self.vis_cmd_queue:
            if position:
                self.vis_cmd_queue.put(("CAMERA_POSITION", position))
            if target:
                self.vis_cmd_queue.put(("CAMERA_TARGET", target))

    def set_parameter(self, param_name, param_value):
        """Set a simulation parameter."""
        if self.use_queues and self.sim_cmd_queue:
            self.sim_cmd_queue.put(("SET_PARAM", (param_name, param_value)))
        else:
            # Direct parameter setting, would need to implement specific logic
            pass

    def exit(self):
        """Exit the application."""
        if self.use_queues:
            if self.sim_cmd_queue:
                self.sim_cmd_queue.put(("EXIT", None))
            if self.vis_cmd_queue:
                self.vis_cmd_queue.put(("EXIT", None))

        # Exit PyQt application
        if HAS_QT and self.app:
            self.app.quit()


class MainWindow(QMainWindow):
    """Main window for the PyDEM UI."""

    def __init__(self, controller):
        """Initialize main window."""
        super().__init__()

        self.controller = controller

        # Window properties
        self.setWindowTitle("PyDEM - Python Discrete Element Method")
        self.resize(1200, 800)

        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Create main layout
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create menu bar
        self._create_menu_bar()

        # Create toolbar
        self._create_tool_bar()

        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)

        # Create main content area
        self._create_main_content()

        # Create dock widgets
        self._create_dock_widgets()

    def _create_menu_bar(self):
        """Create the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        # Open script action
        open_action = QAction("&Open Script", self)
        # pydem/ui/controller.py (continued)

        # Open script action
        open_action = QAction("&Open Script", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_script)
        file_menu.addAction(open_action)

        # Save state action
        save_action = QAction("&Save State", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._on_save_state)
        file_menu.addAction(save_action)

        # Export data action
        export_action = QAction("&Export Data", self)
        export_action.triggered.connect(self._on_export_data)
        file_menu.addAction(export_action)

        file_menu.addSeparator()

        # Exit action
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self._on_exit)
        file_menu.addAction(exit_action)

        # Simulation menu
        sim_menu = menubar.addMenu("&Simulation")

        # Start action
        start_action = QAction("&Start", self)
        start_action.setShortcut("F5")
        start_action.triggered.connect(self._on_start)
        sim_menu.addAction(start_action)

        # Pause action
        pause_action = QAction("&Pause", self)
        pause_action.setShortcut("F6")
        pause_action.triggered.connect(self._on_pause)
        sim_menu.addAction(pause_action)

        # Step action
        step_action = QAction("Step &Forward", self)
        step_action.setShortcut("F7")
        step_action.triggered.connect(self._on_step)
        sim_menu.addAction(step_action)

        sim_menu.addSeparator()

        # Reset action
        reset_action = QAction("&Reset", self)
        reset_action.setShortcut("F8")
        reset_action.triggered.connect(self._on_reset)
        sim_menu.addAction(reset_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Toggle contacts action
        contacts_action = QAction("Show &Contacts", self)
        contacts_action.setCheckable(True)
        contacts_action.setChecked(True)
        contacts_action.triggered.connect(self._on_toggle_contacts)
        view_menu.addAction(contacts_action)

        # Toggle forces action
        forces_action = QAction("Show &Forces", self)
        forces_action.setCheckable(True)
        forces_action.setChecked(True)
        forces_action.triggered.connect(self._on_toggle_forces)
        view_menu.addAction(forces_action)

        # Toggle velocities action
        velocities_action = QAction("Show &Velocities", self)
        velocities_action.setCheckable(True)
        velocities_action.setChecked(False)
        velocities_action.triggered.connect(self._on_toggle_velocities)
        view_menu.addAction(velocities_action)

        view_menu.addSeparator()

        # Render mode submenu
        render_menu = view_menu.addMenu("Render &Mode")

        # Solid render mode
        solid_action = QAction("&Solid", self)
        solid_action.setCheckable(True)
        solid_action.setChecked(True)
        solid_action.triggered.connect(lambda: self._on_set_render_mode("solid"))
        render_menu.addAction(solid_action)

        # Wireframe render mode
        wireframe_action = QAction("&Wireframe", self)
        wireframe_action.setCheckable(True)
        wireframe_action.triggered.connect(
            lambda: self._on_set_render_mode("wireframe")
        )
        render_menu.addAction(wireframe_action)

        # Points render mode
        points_action = QAction("&Points", self)
        points_action.setCheckable(True)
        points_action.triggered.connect(lambda: self._on_set_render_mode("points"))
        render_menu.addAction(points_action)

        # Group render mode actions
        self.render_mode_group = QActionGroup(self)
        self.render_mode_group.addAction(solid_action)
        self.render_mode_group.addAction(wireframe_action)
        self.render_mode_group.addAction(points_action)
        self.render_mode_group.setExclusive(True)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        # About action
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)

    def _create_tool_bar(self):
        """Create the toolbar."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setMovable(False)

        # Open script button
        open_btn = QPushButton("Open Script")
        open_btn.clicked.connect(self._on_open_script)
        toolbar.addWidget(open_btn)

        toolbar.addSeparator()

        # Start button
        start_btn = QPushButton("Start")
        start_btn.clicked.connect(self._on_start)
        toolbar.addWidget(start_btn)

        # Pause button
        pause_btn = QPushButton("Pause")
        pause_btn.clicked.connect(self._on_pause)
        toolbar.addWidget(pause_btn)

        # Step button
        step_btn = QPushButton("Step")
        step_btn.clicked.connect(self._on_step)
        toolbar.addWidget(step_btn)

        # Reset button
        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(self._on_reset)
        toolbar.addWidget(reset_btn)

        toolbar.addSeparator()

        # Speed control
        speed_label = QLabel("Simulation Speed:")
        toolbar.addWidget(speed_label)

        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(100)
        self.speed_slider.setValue(50)
        self.speed_slider.setFixedWidth(100)
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        toolbar.addWidget(self.speed_slider)

        # Speed display
        self.speed_display = QLabel("50%")
        toolbar.addWidget(self.speed_display)

    def _create_main_content(self):
        """Create the main content area."""
        # Create a horizontal splitter for main content
        main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(main_splitter)

        # Visualization panel (placeholder for now)
        self.viz_panel = QWidget()
        viz_layout = QVBoxLayout(self.viz_panel)
        viz_layout.addWidget(
            QLabel("Visualization Panel\n(Placeholder for OpenGL view)")
        )

        # Right panel with controls and stats
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Statistics group
        stats_group = QGroupBox("Simulation Statistics")
        stats_layout = QVBoxLayout(stats_group)

        # Stats labels
        self.step_label = QLabel("Step: 0")
        stats_layout.addWidget(self.step_label)

        self.time_label = QLabel("Time: 0.0 s")
        stats_layout.addWidget(self.time_label)

        self.particles_label = QLabel("Particles: 0")
        stats_layout.addWidget(self.particles_label)

        self.contacts_label = QLabel("Contacts: 0")
        stats_layout.addWidget(self.contacts_label)

        self.status_label = QLabel("Status: Ready")
        stats_layout.addWidget(self.status_label)

        right_layout.addWidget(stats_group)

        # Parameters group
        params_group = QGroupBox("Simulation Parameters")
        params_layout = QVBoxLayout(params_group)

        # Gravity parameter
        gravity_layout = QHBoxLayout()
        gravity_layout.addWidget(QLabel("Gravity:"))

        self.gravity_x = QDoubleSpinBox()
        self.gravity_x.setRange(-100, 100)
        self.gravity_x.setValue(0)
        self.gravity_x.valueChanged.connect(lambda: self._on_gravity_changed())
        gravity_layout.addWidget(self.gravity_x)

        self.gravity_y = QDoubleSpinBox()
        self.gravity_y.setRange(-100, 100)
        self.gravity_y.setValue(0)
        self.gravity_y.valueChanged.connect(lambda: self._on_gravity_changed())
        gravity_layout.addWidget(self.gravity_y)

        self.gravity_z = QDoubleSpinBox()
        self.gravity_z.setRange(-100, 100)
        self.gravity_z.setValue(-9.81)
        self.gravity_z.valueChanged.connect(lambda: self._on_gravity_changed())
        gravity_layout.addWidget(self.gravity_z)

        params_layout.addLayout(gravity_layout)

        # Time step parameter
        dt_layout = QHBoxLayout()
        dt_layout.addWidget(QLabel("Time Step:"))

        self.dt_spinbox = QDoubleSpinBox()
        self.dt_spinbox.setDecimals(6)
        self.dt_spinbox.setRange(0.000001, 1.0)
        self.dt_spinbox.setValue(0.001)
        self.dt_spinbox.setSingleStep(0.0001)
        self.dt_spinbox.valueChanged.connect(self._on_dt_changed)
        dt_layout.addWidget(self.dt_spinbox)

        params_layout.addLayout(dt_layout)

        # Damping parameter
        damping_layout = QHBoxLayout()
        damping_layout.addWidget(QLabel("Damping:"))

        self.damping_spinbox = QDoubleSpinBox()
        self.damping_spinbox.setRange(0.0, 1.0)
        self.damping_spinbox.setValue(0.1)
        self.damping_spinbox.setSingleStep(0.01)
        self.damping_spinbox.valueChanged.connect(self._on_damping_changed)
        damping_layout.addWidget(self.damping_spinbox)

        params_layout.addLayout(damping_layout)

        right_layout.addWidget(params_group)

        # Add stretch to push controls to the top
        right_layout.addStretch()

        # Add panels to the splitter
        main_splitter.addWidget(self.viz_panel)
        main_splitter.addWidget(right_panel)

        # Set initial sizes
        main_splitter.setSizes([800, 400])

    def _create_dock_widgets(self):
        """Create dock widgets."""
        # Log widget
        log_dock = QDockWidget("Log", self)
        log_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_dock.setWidget(self.log_text)

        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)

        # Console widget (for IPython-like interface)
        console_dock = QDockWidget("Console", self)
        console_dock.setAllowedAreas(Qt.BottomDockWidgetArea)

        self.console_text = QTextEdit()
        console_dock.setWidget(self.console_text)

        self.addDockWidget(Qt.BottomDockWidgetArea, console_dock)

        # Tab the dock widgets
        self.tabifyDockWidget(log_dock, console_dock)

        # Inspector widget (for particle details)
        inspector_dock = QDockWidget("Inspector", self)
        inspector_dock.setAllowedAreas(Qt.RightDockWidgetArea)

        inspector_widget = QWidget()
        inspector_layout = QVBoxLayout(inspector_widget)
        inspector_layout.addWidget(
            QLabel("Particle Inspector\n(Select a particle to view details)")
        )

        inspector_dock.setWidget(inspector_widget)

        self.addDockWidget(Qt.RightDockWidgetArea, inspector_dock)

    def _on_open_script(self):
        """Handle open script action."""
        script_path, _ = QFileDialog.getOpenFileName(
            self, "Open Script", "", "Python Files (*.py)"
        )

        if script_path:
            success = self.controller.load_script(script_path)
            if success:
                self.statusBar.showMessage(f"Script loaded: {script_path}", 3000)
                self.log_message(f"Loaded script: {script_path}")
            else:
                self.statusBar.showMessage(
                    f"Failed to load script: {script_path}", 3000
                )
                self.log_message(f"Failed to load script: {script_path}", error=True)

    def _on_save_state(self):
        """Handle save state action."""
        # Not implemented yet
        self.statusBar.showMessage("Save state not implemented yet", 3000)

    def _on_export_data(self):
        """Handle export data action."""
        # Not implemented yet
        self.statusBar.showMessage("Export data not implemented yet", 3000)

    def _on_exit(self):
        """Handle exit action."""
        self.controller.exit()

    def _on_start(self):
        """Handle start action."""
        self.controller.start_simulation()
        self.statusBar.showMessage("Simulation started", 3000)
        self.log_message("Simulation started")

    def _on_pause(self):
        """Handle pause action."""
        self.controller.stop_simulation()
        self.statusBar.showMessage("Simulation paused", 3000)
        self.log_message("Simulation paused")

    def _on_step(self):
        """Handle step action."""
        self.controller.step_simulation()
        self.statusBar.showMessage("Simulation stepped", 3000)

    def _on_reset(self):
        """Handle reset action."""
        self.controller.reset_simulation()
        self.statusBar.showMessage("Simulation reset", 3000)
        self.log_message("Simulation reset")

    def _on_speed_changed(self):
        """Handle simulation speed slider change."""
        value = self.speed_slider.value()
        self.speed_display.setText(f"{value}%")

        # Adjust time step based on speed
        if value < 50:
            # Slower than normal (50% to 1%)
            factor = value / 50.0  # 1.0 to 0.02
        else:
            # Faster than normal (50% to 100%)
            factor = 1.0 + (value - 50) / 25.0  # 1.0 to 3.0

        # Update simulation speed parameter
        self.controller.set_parameter("speed_factor", factor)

    def _on_gravity_changed(self):
        """Handle gravity change."""
        gravity = [
            self.gravity_x.value(),
            self.gravity_y.value(),
            self.gravity_z.value(),
        ]

        self.controller.set_parameter("gravity", gravity)
        self.log_message(f"Gravity changed to {gravity}")

    def _on_dt_changed(self):
        """Handle time step change."""
        dt = self.dt_spinbox.value()
        self.controller.set_parameter("dt", dt)
        self.log_message(f"Time step changed to {dt}")

    def _on_damping_changed(self):
        """Handle damping change."""
        damping = self.damping_spinbox.value()
        self.controller.set_parameter("damping", damping)
        self.log_message(f"Damping changed to {damping}")

    def _on_toggle_contacts(self, checked):
        """Handle toggle contacts action."""
        self.controller.set_parameter("show_contacts", checked)

    def _on_toggle_forces(self, checked):
        """Handle toggle forces action."""
        self.controller.set_parameter("show_forces", checked)

    def _on_toggle_velocities(self, checked):
        """Handle toggle velocities action."""
        self.controller.set_parameter("show_velocities", checked)

    def _on_set_render_mode(self, mode):
        """Handle render mode change."""
        self.controller.set_parameter("render_mode", mode)

    def _on_about(self):
        """Handle about action."""
        from PyQt5.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About PyDEM",
            """<b>PyDEM - Python Discrete Element Method</b>
            <p>Version 0.1.0</p>
            <p>A Python framework for discrete element method simulations.</p>
            <p>Created with ❤️ by the PyDEM Team.</p>""",
        )

    def update_stats(self, stats):
        """Update statistics display."""
        if "step" in stats:
            self.step_label.setText(f"Step: {stats['step']}")

        if "time" in stats:
            self.time_label.setText(f"Time: {stats['time']:.6f} s")

        if "particles" in stats:
            self.particles_label.setText(f"Particles: {stats['particles']}")

        if "contacts" in stats:
            self.contacts_label.setText(f"Contacts: {stats['contacts']}")

        if "running" in stats:
            status = "Running" if stats["running"] else "Paused"
            self.status_label.setText(f"Status: {status}")

    def show_error(self, error_msg):
        """Show error message."""
        self.log_message(error_msg, error=True)
        self.statusBar.showMessage(f"Error: {error_msg}", 5000)

    def log_message(self, message, error=False):
        """Add message to log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = "ERROR" if error else "INFO"
        color = "red" if error else "black"

        html = (
            f'<span style="color:{color}">[{timestamp}] [{prefix}] {message}</span><br>'
        )
        self.log_text.insertHtml(html)

        # Scroll to bottom
        cursor = self.log_text.textCursor()
        cursor.movePosition(cursor.End)
        self.log_text.setTextCursor(cursor)
