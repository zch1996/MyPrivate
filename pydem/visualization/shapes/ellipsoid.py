import moderngl
import numpy as np
import pyrr
from ..vis_utils import create_sphere_mesh


class EllipsoidRenderer:
    """Renderer for ellipsoid shapes"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = self._create_shader_program()
        self.vao = self._create_ellipsoid_vao()

    def _create_shader_program(self):
        """Create shader program for rendering ellipsoids"""
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

    def _create_ellipsoid_vao(self):
        """Create vertex array object for an ellipsoid (using unit sphere)"""
        vertices, indices = create_sphere_mesh(1.0, 32, 16)

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
        color=(0.9, 0.2, 0.5, 1.0),
        wireframe=False,
    ):
        """Render an ellipsoid shape"""
        # Get ellipsoid properties
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        semi_axes = shape.semiAxes

        # Create model matrix
        model_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Apply rotation from orientation quaternion
        rot_matrix = pyrr.matrix44.create_from_quaternion(
            np.array([ori.x, ori.y, ori.z, ori.w]), dtype=np.float32
        )
        model_matrix = pyrr.matrix44.multiply(model_matrix, rot_matrix)

        # Apply scale for semi-axes
        scale_matrix = pyrr.matrix44.create_from_scale(
            np.array([semi_axes[0], semi_axes[1], semi_axes[2]]), dtype=np.float32
        )
        model_matrix = pyrr.matrix44.multiply(model_matrix, scale_matrix)

        # Apply translation
        trans_matrix = pyrr.matrix44.create_from_translation(
            np.array([pos[0], pos[1], pos[2]]), dtype=np.float32
        )
        model_matrix = pyrr.matrix44.multiply(model_matrix, trans_matrix)

        # Set uniforms
        self.program["model"].write(model_matrix.astype("f4"))
        self.program["view"].write(view_matrix.astype("f4"))
        self.program["projection"].write(projection_matrix.astype("f4"))
        self.program["color"].value = color
        self.program["wireframe"].value = wireframe

        # Render
        if wireframe:
            self.ctx.wireframe = True

        self.vao.render(moderngl.TRIANGLES)

        if wireframe:
            self.ctx.wireframe = False

    def cleanup(self):
        """Clean up resources"""
        self.vao.release()
        self.program.release()
