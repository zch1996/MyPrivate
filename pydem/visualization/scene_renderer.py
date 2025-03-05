# pydem/visualization/scene_renderer.py
"""
Scene Renderer
-------------
Renders DEM scenes with particles, contacts, and other elements.
"""

import logging
import numpy as np
from OpenGL.GL import *
from OpenGL.arrays import vbo
import ctypes

logger = logging.getLogger("PyDEM.SceneRenderer")


class SceneRenderer:
    """Renders DEM scenes."""

    def __init__(self, renderer, scene):
        """
        Initialize scene renderer.

        Args:
            renderer: OpenGL renderer
            scene: DEM scene to render
        """
        self.renderer = renderer
        self.scene = scene

        # Render settings
        self.render_mode = "solid"  # "solid", "wireframe", "points"
        self.render_contacts = True
        self.render_forces = True
        self.render_velocities = False

        # Rendering data
        self.sphere_vao = None
        self.sphere_vbo = None
        self.sphere_ebo = None
        self.sphere_vertices = []
        self.sphere_indices = []

        # Initialize geometry
        self._initialize_geometry()

    def _initialize_geometry(self):
        """Initialize geometry for rendering."""
        # Create sphere geometry for particle rendering
        self._create_sphere_geometry()

        # Create VAOs, VBOs, and EBOs for OpenGL
        self._create_buffers()

    def _create_sphere_geometry(self, resolution=16):
        """
        Create sphere geometry for rendering particles.

        Args:
            resolution: Number of segments for the sphere
        """
        vertices = []
        indices = []

        # Generate sphere vertices
        for i in range(resolution + 1):
            phi = i * np.pi / resolution
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            for j in range(resolution * 2):
                theta = j * np.pi / resolution
                sin_theta = np.sin(theta)
                cos_theta = np.cos(theta)

                # Position
                x = cos_theta * sin_phi
                y = sin_theta * sin_phi
                z = cos_phi

                # Normal (same as position for unit sphere)
                nx, ny, nz = x, y, z

                # Add vertex
                vertices.extend([x, y, z, nx, ny, nz, 1.0, 1.0, 1.0, 1.0])

        # Generate indices for triangle strips
        for i in range(resolution):
            for j in range(resolution * 2):
                next_j = (j + 1) % (resolution * 2)

                # First triangle
                indices.extend(
                    [
                        i * (resolution * 2) + j,
                        (i + 1) * (resolution * 2) + j,
                        i * (resolution * 2) + next_j,
                    ]
                )

                # Second triangle
                indices.extend(
                    [
                        i * (resolution * 2) + next_j,
                        (i + 1) * (resolution * 2) + j,
                        (i + 1) * (resolution * 2) + next_j,
                    ]
                )

        self.sphere_vertices = np.array(vertices, dtype=np.float32)
        self.sphere_indices = np.array(indices, dtype=np.uint32)

    def _create_buffers(self):
        """Create OpenGL buffers for rendering."""
        # Create VAO for sphere
        self.sphere_vao = glGenVertexArrays(1)
        glBindVertexArray(self.sphere_vao)

        # Create VBO for sphere vertices
        self.sphere_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.sphere_vbo)
        glBufferData(
            GL_ARRAY_BUFFER,
            self.sphere_vertices.nbytes,
            self.sphere_vertices,
            GL_STATIC_DRAW,
        )

        # Create EBO for sphere indices
        self.sphere_ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.sphere_ebo)
        glBufferData(
            GL_ELEMENT_ARRAY_BUFFER,
            self.sphere_indices.nbytes,
            self.sphere_indices,
            GL_STATIC_DRAW,
        )

        # Position attribute
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        # Normal attribute
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        # Color attribute
        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Unbind VAO
        glBindVertexArray(0)

    def update_scene(self, scene):
        """
        Update scene data for rendering.

        Args:
            scene: New scene to render
        """
        self.scene = scene

    def set_render_mode(self, mode):
        """
        Set rendering mode.

        Args:
            mode: Rendering mode ("solid", "wireframe", "points")
        """
        if mode in ["solid", "wireframe", "points"]:
            self.render_mode = mode

    def render(self, camera):
        """
        Render the scene.

        Args:
            camera: Camera for rendering
        """
        # Use default shader program
        program = self.renderer.use_program()
        if not program:
            return

        # Set projection matrix
        aspect_ratio = self.renderer.width / self.renderer.height
        projection_matrix = camera.perspective_matrix(aspect_ratio)
        self.renderer.set_uniform_matrix4fv(program, "projection", projection_matrix)

        # Set view matrix
        self.renderer.set_uniform_matrix4fv(program, "view", camera.view_matrix)

        # Set light position (in world space)
        self.renderer.set_uniform_3fv(program, "lightPos", camera.position)

        # Set view position (camera position)
        self.renderer.set_uniform_3fv(program, "viewPos", camera.position)

        # Set polygon mode based on render_mode
        if self.render_mode == "wireframe":
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        elif self.render_mode == "points":
            glPolygonMode(GL_FRONT_AND_BACK, GL_POINT)
        else:  # "solid"
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Render particles
        self._render_particles(program)

        # Render contacts if enabled
        if self.render_contacts:
            self._render_contacts(program)

        # Render forces if enabled
        if self.render_forces:
            self._render_forces(program)

        # Render velocities if enabled
        if self.render_velocities:
            self._render_velocities(program)

        # Reset polygon mode
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _render_particles(self, program):
        """
        Render particles in the scene.

        Args:
            program: Shader program to use
        """
        # Bind sphere VAO
        glBindVertexArray(self.sphere_vao)

        # Render each particle
        for p in self.scene.field.getParticles():
            if not p or not p.getShape():
                continue

            # Check if it's a sphere
            if hasattr(p.getShape(), "radius"):
                # Get sphere properties
                pos = p.getPos()
                radius = p.getShape().radius

                # Create model matrix
                model_matrix = np.identity(4, dtype=np.float32)

                # Apply translation
                model_matrix[0, 3] = pos[0]
                model_matrix[1, 3] = pos[1]
                model_matrix[2, 3] = pos[2]

                # Apply scale
                model_matrix[0, 0] = radius
                model_matrix[1, 1] = radius
                model_matrix[2, 2] = radius

                # Set model matrix in shader
                self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

                # Draw the sphere
                glDrawElements(
                    GL_TRIANGLES, len(self.sphere_indices), GL_UNSIGNED_INT, None
                )

        # Unbind VAO
        glBindVertexArray(0)

    def _render_contacts(self, program):
        """
        Render contacts in the scene.

        Args:
            program: Shader program to use
        """
        # TODO: Implement contact rendering
        pass

    # pydem/visualization/scene_renderer.py (continued)

    def _render_forces(self, program):
        """
        Render forces in the scene.

        Args:
            program: Shader program to use
        """
        # Create a temporary VBO for lines
        lines_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, lines_vbo)

        # Collect force lines data
        force_lines = []
        force_colors = []

        # Add force line for each particle with non-zero force
        for p in self.scene.field.getParticles():
            if not p or not p.getShape():
                continue

            # Get force
            force = p.getForce()
            force_mag = np.linalg.norm(force)

            # Skip if force is too small
            if force_mag < 1e-6:
                continue

            # Get particle position
            pos = p.getPos()

            # Normalize force for display
            scale = min(0.5, force_mag / 100)  # Limit max length
            force_norm = force / force_mag * scale

            # Line start/end points
            force_lines.extend([pos[0], pos[1], pos[2]])
            force_lines.extend(
                [pos[0] + force_norm[0], pos[1] + force_norm[1], pos[2] + force_norm[2]]
            )

            # Colors (red for forces)
            force_colors.extend([1.0, 0.0, 0.0, 1.0])
            force_colors.extend([1.0, 0.0, 0.0, 1.0])

        # If no forces to render, return
        if not force_lines:
            glDeleteBuffers(1, [lines_vbo])
            return

        # Combine data
        force_data = []
        for i in range(0, len(force_lines), 3):
            force_data.extend(force_lines[i : i + 3])  # Position
            force_data.extend([0.0, 0.0, 1.0])  # Normal (dummy)
            j = (i // 3) * 4
            force_data.extend(force_colors[j : j + 4])  # Color

        # Upload data to VBO
        force_data_array = np.array(force_data, dtype=np.float32)
        glBufferData(
            GL_ARRAY_BUFFER, force_data_array.nbytes, force_data_array, GL_STATIC_DRAW
        )

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Set line width
        glLineWidth(3.0)

        # Create identity model matrix
        model_matrix = np.identity(4, dtype=np.float32)
        self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

        # Draw lines
        glDrawArrays(GL_LINES, 0, len(force_lines) // 3)

        # Reset line width
        glLineWidth(1.0)

        # Clean up
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDeleteBuffers(1, [lines_vbo])

    def _render_velocities(self, program):
        """
        Render velocity vectors in the scene.

        Args:
            program: Shader program to use
        """
        # Create a temporary VBO for lines
        lines_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, lines_vbo)

        # Collect velocity lines data
        vel_lines = []
        vel_colors = []

        # Add velocity line for each particle with non-zero velocity
        for p in self.scene.field.getParticles():
            if not p or not p.getShape():
                continue

            # Get velocity
            vel = p.getVel()
            vel_mag = np.linalg.norm(vel)

            # Skip if velocity is too small
            if vel_mag < 1e-6:
                continue

            # Get particle position
            pos = p.getPos()

            # Normalize velocity for display
            scale = min(0.5, vel_mag / 10)  # Limit max length
            vel_norm = vel / vel_mag * scale

            # Line start/end points
            vel_lines.extend([pos[0], pos[1], pos[2]])
            vel_lines.extend(
                [pos[0] + vel_norm[0], pos[1] + vel_norm[1], pos[2] + vel_norm[2]]
            )

            # Colors (green for velocities)
            vel_colors.extend([0.0, 1.0, 0.0, 1.0])
            vel_colors.extend([0.0, 1.0, 0.0, 1.0])

        # If no velocities to render, return
        if not vel_lines:
            glDeleteBuffers(1, [lines_vbo])
            return

        # Combine data
        vel_data = []
        for i in range(0, len(vel_lines), 3):
            vel_data.extend(vel_lines[i : i + 3])  # Position
            vel_data.extend([0.0, 0.0, 1.0])  # Normal (dummy)
            j = (i // 3) * 4
            vel_data.extend(vel_colors[j : j + 4])  # Color

        # Upload data to VBO
        vel_data_array = np.array(vel_data, dtype=np.float32)
        glBufferData(
            GL_ARRAY_BUFFER, vel_data_array.nbytes, vel_data_array, GL_STATIC_DRAW
        )

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Set line width
        glLineWidth(2.0)

        # Create identity model matrix
        model_matrix = np.identity(4, dtype=np.float32)
        self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

        # Draw lines
        glDrawArrays(GL_LINES, 0, len(vel_lines) // 3)

        # Reset line width
        glLineWidth(1.0)

        # Clean up
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDeleteBuffers(1, [lines_vbo])

    def render_contact_network(self, program):
        """
        Render contact network between particles.

        Args:
            program: Shader program to use
        """
        # Create a temporary VBO for lines
        lines_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, lines_vbo)

        # Collect contact lines data
        contact_lines = []
        contact_colors = []

        # Process each contact
        contacts = self.scene.field.getContacts()
        for c in contacts:
            if not c.isReal():
                continue

            pA = c.getParticleA()
            pB = c.getParticleB()

            if not pA or not pB:
                continue

            # Get particle positions
            posA = pA.getPos()
            posB = pB.getPos()

            # Account for periodic boundaries
            if self.scene.isPeriodic:
                cellDist = c.getCellDist()
                shift = self.scene.cell.intrShiftPos(cellDist)
                posB = posB + shift

            # Line start/end points
            contact_lines.extend([posA[0], posA[1], posA[2]])
            contact_lines.extend([posB[0], posB[1], posB[2]])

            # Get contact force magnitude for coloring
            force_mag = np.linalg.norm(c.getForce()) if c.getPhys() else 0

            # Normalize force for color intensity
            norm_force = min(1.0, force_mag / 100)

            # Colors (blue for contacts, intensity based on force)
            contact_colors.extend([0.0, 0.0, norm_force, 0.7])
            contact_colors.extend([0.0, 0.0, norm_force, 0.7])

        # If no contacts to render, return
        if not contact_lines:
            glDeleteBuffers(1, [lines_vbo])
            return

        # Combine data
        contact_data = []
        for i in range(0, len(contact_lines), 3):
            contact_data.extend(contact_lines[i : i + 3])  # Position
            contact_data.extend([0.0, 0.0, 1.0])  # Normal (dummy)
            j = (i // 3) * 4
            contact_data.extend(contact_colors[j : j + 4])  # Color

        # Upload data to VBO
        contact_data_array = np.array(contact_data, dtype=np.float32)
        glBufferData(
            GL_ARRAY_BUFFER,
            contact_data_array.nbytes,
            contact_data_array,
            GL_STATIC_DRAW,
        )

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Set line width
        glLineWidth(1.0)

        # Create identity model matrix
        model_matrix = np.identity(4, dtype=np.float32)
        self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

        # Enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Draw lines
        glDrawArrays(GL_LINES, 0, len(contact_lines) // 3)

        # Disable alpha blending
        glDisable(GL_BLEND)

        # Reset line width
        glLineWidth(1.0)

        # Clean up
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDeleteBuffers(1, [lines_vbo])

    def render_bounding_boxes(self, program):
        """
        Render bounding boxes for particles.

        Args:
            program: Shader program to use
        """
        # Create a temporary VBO for lines
        lines_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, lines_vbo)

        # Collect bounding box lines data
        bbox_lines = []
        bbox_colors = []

        # Add bounding box for each particle
        for p in self.scene.field.getParticles():
            if not p or not p.getShape() or not p.getShape().bound:
                continue

            # Get bounding box
            aabb = p.getShape().bound
            min_pt = aabb.min
            max_pt = aabb.max

            # Define 8 corners of the box
            corners = [
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ]

            # Define 12 edges of the box (connects the corners)
            edges = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),  # Bottom face
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),  # Top face
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),  # Connecting edges
            ]

            # Add lines for each edge
            for edge in edges:
                bbox_lines.extend(corners[edge[0]])
                bbox_lines.extend(corners[edge[1]])

                # Colors (yellow for bounding boxes)
                bbox_colors.extend([1.0, 1.0, 0.0, 0.3])
                bbox_colors.extend([1.0, 1.0, 0.0, 0.3])

        # If no bounding boxes to render, return
        if not bbox_lines:
            glDeleteBuffers(1, [lines_vbo])
            return

        # Combine data
        bbox_data = []
        for i in range(0, len(bbox_lines), 3):
            bbox_data.extend(bbox_lines[i : i + 3])  # Position
            bbox_data.extend([0.0, 0.0, 1.0])  # Normal (dummy)
            j = (i // 3) * 4
            bbox_data.extend(bbox_colors[j : j + 4])  # Color

        # Upload data to VBO
        bbox_data_array = np.array(bbox_data, dtype=np.float32)
        glBufferData(
            GL_ARRAY_BUFFER, bbox_data_array.nbytes, bbox_data_array, GL_STATIC_DRAW
        )

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Enable alpha blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Create identity model matrix
        model_matrix = np.identity(4, dtype=np.float32)
        self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

        # Draw lines
        glDrawArrays(GL_LINES, 0, len(bbox_lines) // 3)

        # Disable alpha blending
        glDisable(GL_BLEND)

        # Clean up
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDeleteBuffers(1, [lines_vbo])

    def render_coordinate_axes(self, program):
        """
        Render coordinate axes at the origin.

        Args:
            program: Shader program to use
        """
        # Create a temporary VBO for lines
        axes_vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, axes_vbo)

        # Define axes lines
        axes_lines = [
            # X-axis (red)
            0.0,
            0.0,
            0.0,  # Start at origin
            1.0,
            0.0,
            0.0,  # End at (1,0,0)
            # Y-axis (green)
            0.0,
            0.0,
            0.0,  # Start at origin
            0.0,
            1.0,
            0.0,  # End at (0,1,0)
            # Z-axis (blue)
            0.0,
            0.0,
            0.0,  # Start at origin
            0.0,
            0.0,
            1.0,  # End at (0,0,1)
        ]

        # Define colors for each axis
        axes_colors = [
            # X-axis (red)
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            # Y-axis (green)
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            0.0,
            1.0,
            # Z-axis (blue)
            0.0,
            0.0,
            1.0,
            1.0,
            0.0,
            0.0,
            1.0,
            1.0,
        ]

        # Combine data
        axes_data = []
        for i in range(0, len(axes_lines), 3):
            axes_data.extend(axes_lines[i : i + 3])  # Position
            axes_data.extend([0.0, 0.0, 1.0])  # Normal (dummy)
            j = (i // 3) * 4
            axes_data.extend(axes_colors[j : j + 4])  # Color

        # Upload data to VBO
        axes_data_array = np.array(axes_data, dtype=np.float32)
        glBufferData(
            GL_ARRAY_BUFFER, axes_data_array.nbytes, axes_data_array, GL_STATIC_DRAW
        )

        # Configure vertex attributes
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)

        glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 10 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)

        # Set line width
        glLineWidth(3.0)

        # Scale and position the axes
        model_matrix = np.identity(4, dtype=np.float32)
        model_matrix[0, 0] = 1.0  # Scale X
        model_matrix[1, 1] = 1.0  # Scale Y
        model_matrix[2, 2] = 1.0  # Scale Z
        self.renderer.set_uniform_matrix4fv(program, "model", model_matrix)

        # Draw lines
        glDrawArrays(GL_LINES, 0, len(axes_lines) // 3)

        # Reset line width
        glLineWidth(1.0)

        # Clean up
        glDisableVertexAttribArray(0)
        glDisableVertexAttribArray(1)
        glDisableVertexAttribArray(2)
        glDeleteBuffers(1, [axes_vbo])

    def cleanup(self):
        """Clean up resources."""
        # Delete sphere buffers
        if self.sphere_vao:
            glDeleteVertexArrays(1, [self.sphere_vao])
        if self.sphere_vbo:
            glDeleteBuffers(1, [self.sphere_vbo])
        if self.sphere_ebo:
            glDeleteBuffers(1, [self.sphere_ebo])
