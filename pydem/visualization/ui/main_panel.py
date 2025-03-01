# pydem/visualization/ui/main_panel.py
import imgui


class MainPanel:
    """Main control panel for the visualization"""

    def __init__(self, renderer):
        self.renderer = renderer
        self.visible = True
        self.color_modes = ["uniform", "velocity", "type"]
        self.selected_particle = -1

    def render(self):
        """Render the main control panel"""
        if not self.visible:
            return

        imgui.begin("PyDEM Controls", True)

        # Display simulation info
        scene = self.renderer.omega.scene
        imgui.text(f"Time: {scene.time:.4f}")
        imgui.text(f"Step: {scene.step}")
        imgui.text(f"FPS: {self.renderer.fps:.1f}")

        imgui.separator()

        # Visualization options
        changed, value = imgui.combo(
            "Color Mode",
            self.color_modes.index(self.renderer.color_mode),
            self.color_modes,
        )
        if changed:
            self.renderer.color_mode = self.color_modes[value]

        changed, value = imgui.checkbox("Wireframe", self.renderer.show_wireframe)
        if changed:
            self.renderer.show_wireframe = value

        changed, value = imgui.checkbox("Show Tracers", self.renderer.show_tracers)
        if changed:
            self.renderer.show_tracers = value

        imgui.separator()

        # Particle selection
        imgui.text("Particle Selection")
        particles = (
            self.renderer.omega.scene.field.particles
            if self.renderer.omega.scene and self.renderer.omega.scene.field
            else []
        )

        # Particle ID input
        changed, value = imgui.input_int("Particle ID", self.selected_particle)
        if changed:
            self.selected_particle = max(
                0, min(value, len(particles) - 1 if particles else 0)
            )

        if imgui.button("Inspect Particle"):
            # Show particle inspector for selected particle
            pass

        imgui.same_line()

        if imgui.button("Toggle Tracing"):
            if (
                self.selected_particle >= 0
                and particles
                and self.selected_particle < len(particles)
            ):
                if self.selected_particle in self.renderer.tracer_particles:
                    self.renderer.disable_tracing(self.selected_particle)
                else:
                    self.renderer.enable_tracing(self.selected_particle)

        imgui.separator()

        # Simulation controls
        if imgui.button("Pause/Resume"):
            if self.renderer.omega.scene.isRunning():
                self.renderer.omega.scene.stop()
            else:
                self.renderer.omega.scene.run(wait=False)

        imgui.same_line()

        if imgui.button("Step"):
            # Step simulation
            if not self.renderer.omega.scene.isRunning():
                self.renderer.omega.scene.step += 1
                self.renderer.omega.scene.doOneStep()

        imgui.same_line()

        if imgui.button("Reset Camera"):
            self.renderer.camera.reset()

        imgui.end()
