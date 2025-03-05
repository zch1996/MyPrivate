# pydem/visualization/renderer.py
"""
OpenGL Renderer
--------------
Handles the OpenGL context and rendering operations.
"""

import logging
import numpy as np
import sys

logger = logging.getLogger("PyDEM.Renderer")

# Try to import OpenGL
try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GL import shaders
    import pygame
    from pygame.locals import *

    HAS_OPENGL = True
except ImportError:
    logger.warning("OpenGL libraries not found. Visualization will be disabled.")
    HAS_OPENGL = False


class Renderer:
    """OpenGL renderer for PyDEM visualization."""

    def __init__(self, width=800, height=600, title="PyDEM Visualization"):
        """
        Initialize OpenGL renderer.

        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        self.width = width
        self.height = height
        self.title = title

        self.window = None
        self.gl_context = None

        # Shaders and programs
        self.shaders = {}
        self.programs = {}

        # Initialize pygame and OpenGL
        if HAS_OPENGL:
            self._initialize_pygame()
            self._initialize_opengl()
            self._compile_shaders()
        else:
            logger.warning("OpenGL not available, renderer initialized in dummy mode")

    def _initialize_pygame(self):
        """Initialize pygame and create window."""
        try:
            pygame.init()
            pygame.display.set_caption(self.title)

            # Set OpenGL attributes
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )

            # Create window with OpenGL context
            self.window = pygame.display.set_mode(
                (self.width, self.height), DOUBLEBUF | OPENGL
            )

            # Get OpenGL context
            self.gl_context = pygame.display.get_context()

            logger.info("Pygame and OpenGL window initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize pygame: {str(e)}")
            raise

    def _initialize_opengl(self):
        """Initialize OpenGL settings."""
        try:
            # Basic OpenGL settings
            glClearColor(0.2, 0.2, 0.2, 1.0)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Get OpenGL info
            gl_info = {
                "vendor": glGetString(GL_VENDOR).decode("utf-8"),
                "renderer": glGetString(GL_RENDERER).decode("utf-8"),
                "version": glGetString(GL_VERSION).decode("utf-8"),
                "shader_version": glGetString(GL_SHADING_LANGUAGE_VERSION).decode(
                    "utf-8"
                ),
            }

            logger.info(
                f"OpenGL initialized: {gl_info['renderer']} - {gl_info['version']}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize OpenGL: {str(e)}")
            raise

    def _compile_shaders(self):
        """Compile and link OpenGL shaders."""
        # Define basic vertex and fragment shaders
        vertex_shader_source = """
        #version 330 core
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 normal;
        layout(location = 2) in vec4 color;
        
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;
        
        out vec3 fragNormal;
        out vec3 fragPosition;
        out vec4 fragColor;
        
        void main() {
            fragPosition = vec3(model * vec4(position, 1.0));
            fragNormal = mat3(transpose(inverse(model))) * normal;
            fragColor = color;
            gl_Position = projection * view * model * vec4(position, 1.0);
        }
        """

        fragment_shader_source = """
        #version 330 core
        in vec3 fragNormal;
        in vec3 fragPosition;
        in vec4 fragColor;
        
        uniform vec3 lightPos;
        uniform vec3 viewPos;
        
        out vec4 finalColor;
        
        void main() {
            // Ambient lighting
            float ambientStrength = 0.3;
            vec3 ambient = ambientStrength * vec3(1.0, 1.0, 1.0);
            
            // Diffuse lighting
            vec3 norm = normalize(fragNormal);
            vec3 lightDir = normalize(lightPos - fragPosition);
            float diff = max(dot(norm, lightDir), 0.0);
            vec3 diffuse = diff * vec3(1.0, 1.0, 1.0);
            
            // Specular lighting
            float specularStrength = 0.5;
            vec3 viewDir = normalize(viewPos - fragPosition);
            vec3 reflectDir = reflect(-lightDir, norm);
            float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
            vec3 specular = specularStrength * spec * vec3(1.0, 1.0, 1.0);
            
            // Combine lighting with fragment color
            vec3 result = (ambient + diffuse + specular) * fragColor.rgb;
            finalColor = vec4(result, fragColor.a);
        }
        """

        try:
            # Compile shaders
            vertex_shader = shaders.compileShader(
                vertex_shader_source, GL_VERTEX_SHADER
            )
            fragment_shader = shaders.compileShader(
                fragment_shader_source, GL_FRAGMENT_SHADER
            )

            # Link shaders into program
            shader_program = shaders.compileProgram(vertex_shader, fragment_shader)

            # Store shader program
            self.programs["default"] = shader_program

            logger.info("Shaders compiled and linked successfully")

        except Exception as e:
            logger.error(f"Failed to compile shaders: {str(e)}")
            raise

    def make_context_current(self):
        """Make the OpenGL context current in the calling thread."""
        if HAS_OPENGL and self.gl_context:
            pygame.display.get_context().make_current()

    def resize(self, width, height):
        """Resize the rendering viewport."""
        self.width = width
        self.height = height

        if HAS_OPENGL:
            glViewport(0, 0, width, height)

    def begin_frame(self):
        """Begin a new frame."""
        if HAS_OPENGL:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    def end_frame(self):
        """End the current frame and swap buffers."""
        if HAS_OPENGL:
            pygame.display.flip()

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Signal application to exit
                    pass

    def use_program(self, program_name="default"):
        """Activate a shader program."""
        if HAS_OPENGL and program_name in self.programs:
            glUseProgram(self.programs[program_name])
            return self.programs[program_name]
        return None

    def set_uniform_matrix4fv(self, program, name, matrix):
        """Set a uniform 4x4 matrix in a shader program."""
        if HAS_OPENGL:
            loc = glGetUniformLocation(program, name)
            if loc != -1:
                glUniformMatrix4fv(loc, 1, GL_FALSE, matrix)

    def set_uniform_3fv(self, program, name, vector):
        """Set a uniform 3D vector in a shader program."""
        if HAS_OPENGL:
            loc = glGetUniformLocation(program, name)
            if loc != -1:
                glUniform3fv(loc, 1, vector)

    def cleanup(self):
        """Clean up resources."""
        if HAS_OPENGL:
            # Delete shader programs
            for program in self.programs.values():
                glDeleteProgram(program)

            # Quit pygame
            pygame.quit()
