import moderngl
import numpy as np
import pyrr


class FacetRenderer:
    """Renderer for facet shapes"""

    def __init__(self, ctx):
        self.ctx = ctx
        self.program = self._create_shader_program()

    def _create_shader_program(self):
        """Create shader program for rendering facets"""
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

    def render(
        self,
        shape,
        view_matrix,
        projection_matrix,
        color=(0.9, 0.5, 0.2, 1.0),
        wireframe=False,
    ):
        """Render a facet shape"""
        # Get facet vertices
        vertices = []
        indices = []

        # Get the three vertices of the facet
        p1 = shape.nodes[0].pos
        p2 = shape.nodes[1].pos
        p3 = shape.nodes[2].pos

        # Calculate normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)

        # Add vertices
        vertices.extend([p1[0], p1[1], p1[2], normal[0], normal[1], normal[2]])
        vertices.extend([p2[0], p2[1], p2[2], normal[0], normal[1], normal[2]])
        vertices.extend([p3[0], p3[1], p3[2], normal[0], normal[1], normal[2]])

        # Add indices for triangle
        indices.extend([0, 1, 2])

        # Create buffers
        vbo = self.ctx.buffer(np.array(vertices, dtype=np.float32))
        ibo = self.ctx.buffer(np.array(indices, dtype=np.uint32))

        # Create VAO
        vao_content = [(vbo, "3f 3f", "in_position", "in_normal")]

        vao = self.ctx.vertex_array(self.program, vao_content, ibo)

        # Create model matrix (identity for facets as vertices are already in world space)
        model_matrix = pyrr.matrix44.create_identity(dtype=np.float32)

        # Set uniforms
        self.program["model"].write(model_matrix.astype("f4"))
        self.program["view"].write(view_matrix.astype("f4"))
        self.program["projection"].write(projection_matrix.astype("f4"))
        self.program["color"].value = color
        self.program["wireframe"].value = wireframe

        # Render
        if wireframe:
            self.ctx.wireframe = True

        vao.render(moderngl.TRIANGLES)

        if wireframe:
            self.ctx.wireframe = False

        # Clean up
        vao.release()
        vbo.release()
        ibo.release()

    def cleanup(self):
        """Clean up resources"""
        self.program.release()
