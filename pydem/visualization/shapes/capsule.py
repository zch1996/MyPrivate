import moderngl
import numpy as np
import pyrr
from ..vis_utils import create_sphere_mesh, create_cylinder_mesh
from pydem.src.demmath import Vector3r


class CapsuleRenderer:
    """Renderer for capsule shapes"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = self._create_shader_program()
        self.sphere_vao = self._create_sphere_vao()
        self.cylinder_vao = self._create_cylinder_vao()

    def _create_shader_program(self):
        """Create shader program for rendering capsules"""
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
        color=(0.2, 0.9, 0.5, 1.0),
        wireframe=False,
    ):
        """Render a capsule shape"""
        # Get capsule properties
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        radius = shape.radius
        shaft = shape.shaft

        # Set common uniforms
        self.program["view"].write(view_matrix.astype("f4"))
        self.program["projection"].write(projection_matrix.astype("f4"))
        self.program["color"].value = color
        self.program["wireframe"].value = wireframe

        # Set wireframe mode if needed
        if wireframe:
            self.ctx.wireframe = True

        # Render cylinder for shaft
        if shaft > 0:
            # Create model matrix for cylinder
            cylinder_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

            # Apply rotation from orientation quaternion
            rot_matrix = pyrr.matrix44.create_from_quaternion(
                np.array([ori.x, ori.y, ori.z, ori.w]), dtype=np.float32
            )
            cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, rot_matrix)

            # Apply scale
            scale_matrix = pyrr.matrix44.create_from_scale(
                np.array([radius, shaft, radius]), dtype=np.float32
            )
            cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, scale_matrix)

            # Apply translation
            trans_matrix = pyrr.matrix44.create_from_translation(
                np.array([pos[0], pos[1], pos[2]]), dtype=np.float32
            )
            cylinder_matrix = pyrr.matrix44.multiply(cylinder_matrix, trans_matrix)

            self.program["model"].write(cylinder_matrix.astype("f4"))
            self.cylinder_vao.render(moderngl.TRIANGLES)

        # Render spheres at ends
        # First end
        shaft_vec = Vector3r(shaft / 2, 0, 0)
        end1_pos = pos - ori.rotate(shaft_vec)
        sphere1_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply translation
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([end1_pos[0], end1_pos[1], end1_pos[2]]), dtype=np.float32
        )
        sphere1_matrix = pyrr.matrix44.multiply(sphere1_matrix, trans_matrix)

        # Apply scale
        scale_matrix = pyrr.matrix44.create_from_scale(
            np.array([radius, radius, radius]), dtype=np.float32
        )
        sphere1_matrix = pyrr.matrix44.multiply(sphere1_matrix, scale_matrix)

        self.program["model"].write(sphere1_matrix.astype("f4"))
        self.sphere_vao.render(moderngl.TRIANGLES)

        # Second end
        end2_pos = pos + ori.rotate(shaft_vec)
        sphere2_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply translation
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([end2_pos[0], end2_pos[1], end2_pos[2]]), dtype=np.float32
        )
        sphere2_matrix = pyrr.matrix44.multiply(sphere2_matrix, trans_matrix)

        # Apply scale
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
