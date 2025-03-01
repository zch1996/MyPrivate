import moderngl
import numpy as np
import pyrr
from ..vis_utils import create_sphere_mesh, create_cylinder_mesh
from pydem.src.demmath import Vector3r


class ConeRenderer:
    """Renderer for cone shapes"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = self._create_shader_program()
        self.sphere_vao = self._create_sphere_vao()
        self.cylinder_vao = self._create_cylinder_vao()

    def _create_shader_program(self):
        """Create shader program for rendering cones"""
        vertex_shader = """
            #version 330
            
            uniform mat4 model;
            uniform mat4 view;
            uniform mat4 projection;
            
            in vec3 in_position;
            in vec3 in_normal;
            
            out vec3 normal;
            out vec3 frag_pos;
            
            void main() {
                frag_pos = vec3(model * vec4(in_position, 1.0));
                normal = mat3(transpose(inverse(model))) * in_normal;
                gl_Position = projection * view * model * vec4(in_position, 1.0);
            }
        """

        fragment_shader = """
            #version 330
            
            uniform vec4 color;
            uniform bool wireframe;
            
            in vec3 normal;
            in vec3 frag_pos;
            
            out vec4 frag_color;
            
            void main() {
                if (wireframe) {
                    frag_color = color;
                } else {
                    vec3 light_pos = vec3(10.0, 10.0, 10.0);
                    vec3 light_color = vec3(1.0, 1.0, 1.0);
                    
                    // Ambient
                    float ambient_strength = 0.3;
                    vec3 ambient = ambient_strength * light_color;
                    
                    // Diffuse
                    vec3 norm = normalize(normal);
                    vec3 light_dir = normalize(light_pos - frag_pos);
                    float diff = max(dot(norm, light_dir), 0.0);
                    vec3 diffuse = diff * light_color;
                    
                    vec3 result = (ambient + diffuse) * color.rgb;
                    frag_color = vec4(result, color.a);
                }
            }
        """

        return self.ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

    def _create_sphere_vao(self):
        """Create vertex array object for a sphere"""
        vertices, indices = create_sphere_mesh(1.0, 32, 16)

        # Create buffers
        vbo = self.ctx.buffer(np.array(vertices, dtype=np.float32))
        ibo = self.ctx.buffer(np.array(indices, dtype=np.uint32))

        # Create VAO
        vao_content = [(vbo, "3f 3f", "in_position", "in_normal")]

        return self.ctx.vertex_array(self.program, vao_content, ibo)

    def _create_cylinder_vao(self):
        """Create vertex array object for a cylinder"""
        vertices, indices = create_cylinder_mesh(1.0, 1.0, 32)

        # Create buffers
        vbo = self.ctx.buffer(np.array(vertices, dtype=np.float32))
        ibo = self.ctx.buffer(np.array(indices, dtype=np.uint32))

        # Create VAO
        vao_content = [(vbo, "3f 3f", "in_position", "in_normal")]

        return self.ctx.vertex_array(self.program, vao_content, ibo)

    def render(
        self,
        shape,
        view_matrix,
        projection_matrix,
        color=(0.8, 0.4, 0.1, 1.0),
        wireframe=False,
    ):
        """Render a cone shape"""
        # Get cone properties
        pos1 = shape.nodes[0].pos
        pos2 = shape.nodes[1].pos
        radii = shape.radii  # Vector2r with two radii

        # Calculate axis direction and length
        axis = pos2 - pos1
        length = np.linalg.norm(axis)
        if length < 1e-10:
            return  # Too short to render

        direction = axis / length

        # Calculate orientation quaternion from direction vector
        # This aligns the cylinder with the axis between the two nodes
        up = Vector3r(0, 1, 0)
        if abs(np.dot(direction, up)) > 0.99:
            up = Vector3r(1, 0, 0)

        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)

        new_up = np.cross(right, direction)

        # Create rotation matrix
        rot_matrix = np.zeros((3, 3), dtype=np.float32)
        rot_matrix[:, 0] = right
        rot_matrix[:, 1] = new_up
        rot_matrix[:, 2] = direction

        # Convert to quaternion
        trace = rot_matrix[0, 0] + rot_matrix[1, 1] + rot_matrix[2, 2]
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rot_matrix[2, 1] - rot_matrix[1, 2]) * s
            y = (rot_matrix[0, 2] - rot_matrix[2, 0]) * s
            z = (rot_matrix[1, 0] - rot_matrix[0, 1]) * s
        else:
            if (
                rot_matrix[0, 0] > rot_matrix[1, 1]
                and rot_matrix[0, 0] > rot_matrix[2, 2]
            ):
                s = 2.0 * np.sqrt(
                    1.0 + rot_matrix[0, 0] - rot_matrix[1, 1] - rot_matrix[2, 2]
                )
                w = (rot_matrix[2, 1] - rot_matrix[1, 2]) / s
                x = 0.25 * s
                y = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                z = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
            elif rot_matrix[1, 1] > rot_matrix[2, 2]:
                s = 2.0 * np.sqrt(
                    1.0 + rot_matrix[1, 1] - rot_matrix[0, 0] - rot_matrix[2, 2]
                )
                w = (rot_matrix[0, 2] - rot_matrix[2, 0]) / s
                x = (rot_matrix[0, 1] + rot_matrix[1, 0]) / s
                y = 0.25 * s
                z = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(
                    1.0 + rot_matrix[2, 2] - rot_matrix[0, 0] - rot_matrix[1, 1]
                )
                w = (rot_matrix[1, 0] - rot_matrix[0, 1]) / s
                x = (rot_matrix[0, 2] + rot_matrix[2, 0]) / s
                y = (rot_matrix[1, 2] + rot_matrix[2, 1]) / s
                z = 0.25 * s

        orientation = np.array([x, y, z, w])

        # Set common uniforms
        self.program["view"].write(view_matrix.astype("f4"))
        self.program["projection"].write(projection_matrix.astype("f4"))
        self.program["color"].value = color
        self.program["wireframe"].value = wireframe

        # Set wireframe mode if needed
        if wireframe:
            self.ctx.wireframe = True

        # Render cylinder for shaft
        # Create model matrix for cylinder
        cylinder_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply rotation
        rot_matrix = pyrr.matrix44.create_from_quaternion(orientation, dtype=np.float32)
        cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, rot_matrix)

        # Apply scale - use average radius for cylinder and adjust length
        avg_radius = (radii[0] + radii[1]) / 2
        scale_matrix = pyrr.matrix44.create_from_scale(
            np.array([avg_radius, length, avg_radius]), dtype=np.float32
        )
        cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, scale_matrix)

        # Apply translation to center of cone
        center = (pos1 + pos2) / 2
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([center[0], center[1], center[2]]), dtype=np.float32
        )
        cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, trans_matrix)

        self.program["model"].write(cylinder_matrix.astype("f4"))
        self.cylinder_vao.render(moderngl.TRIANGLES)

        # Render spheres at ends
        # First end
        sphere1_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply translation
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([pos1[0], pos1[1], pos1[2]]), dtype=np.float32
        )
        sphere1_matrix = pyrr.matrix44.multiply(sphere1_matrix, trans_matrix)

        # Apply scale
        scale_matrix = pyrr.matrix44.create_from_scale(
            np.array([radii[0], radii[0], radii[0]]), dtype=np.float32
        )
        sphere1_matrix = pyrr.matrix44.multiply(sphere1_matrix, scale_matrix)

        self.program["model"].write(sphere1_matrix.astype("f4"))
        self.sphere_vao.render(moderngl.TRIANGLES)

        # Second end
        sphere2_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply translation
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([pos2[0], pos2[1], pos2[2]]), dtype=np.float32
        )
        sphere2_matrix = pyrr.matrix44.multiply(sphere2_matrix, trans_matrix)

        # Apply scale
        scale_matrix = pyrr.matrix44.create_from_scale(
            np.array([radii[1], radii[1], radii[1]]), dtype=np.float32
        )
        sphere2_matrix = pyrr.matrix44.multiply(sphere2_matrix, scale_matrix)

        self.program["model"].write(sphere2_matrix.astype("f4"))
        self.sphere_vao.render(moderngl.TRIANGLES)

        if wireframe:
            self.ctx.wireframe = False

    def cleanup(self):
        """Clean up resources"""
        self.sphere_vao.release()
        self.cylinder_vao.release()
        self.program.release()
