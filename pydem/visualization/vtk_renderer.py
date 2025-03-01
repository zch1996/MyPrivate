import time
import threading
import traceback
import numpy as np
import vtk
from vtk.util import numpy_support


class VTKRenderer:
    """VTK-based renderer for PyDEM simulations"""

    def __init__(self, omega=None):
        """Initialize the renderer

        Args:
            omega: Omega instance (defaults to global O)
        """
        from pydem import O

        self.omega = omega or O
        self.running = False

        # VTK objects
        self.renderer = None
        self.render_window = None
        self.render_window_interactor = None

        # Actor dictionaries to keep track of objects
        self.particle_actors = {}  # {particle_id: actor}

        # Rendering options
        self.show_wireframe = False
        self.color_mode = "uniform"  # uniform, velocity, type

        # Scale factor for rendering (DEM units are often very small)
        self.scale_factor = 1.0
        self.auto_scale = True
        self.min_visible_size = 10.0  # 最小可见尺寸（像素）

        # Update frequency (in seconds)
        self.update_interval = 0.05  # 20 FPS

        # Lock for thread safety
        self.lock = threading.RLock()

    def start(self):
        """Start the visualization"""
        self.running = True
        try:
            self._init_vtk()
            self._start_update_timer()
            self.render_window_interactor.Start()
        except Exception as e:
            print(f"Error in renderer: {e}")
            traceback.print_exc()

    def stop(self):
        """Stop the visualization"""
        self.running = False
        if self.render_window_interactor:
            self.render_window_interactor.TerminateApp()

    def _init_vtk(self):
        """Initialize VTK rendering pipeline"""
        # Create renderer
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.8, 0.8, 0.8)  # Light gray background

        # Create render window
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(1280, 720)
        self.render_window.SetWindowName("PyDEM Visualization")

        # Create interactor
        self.render_window_interactor = vtk.vtkRenderWindowInteractor()
        self.render_window_interactor.SetRenderWindow(self.render_window)

        # Set up interactor style (trackball camera)
        class CustomTrackballStyle(vtk.vtkInteractorStyleTrackballCamera):
            def __init__(self, parent=None):
                self.parent = parent
                self.AddObserver("MouseWheelForwardEvent", self.mouse_wheel_forward)
                self.AddObserver("MouseWheelBackwardEvent", self.mouse_wheel_backward)

            def mouse_wheel_forward(self, obj, event):
                camera = self.GetCurrentRenderer().GetActiveCamera()
                camera.Zoom(1.1)
                self.GetCurrentRenderer().ResetCameraClippingRange()
                self.GetInteractor().GetRenderWindow().Render()

            def mouse_wheel_backward(self, obj, event):
                camera = self.GetCurrentRenderer().GetActiveCamera()
                camera.Zoom(0.9)
                self.GetCurrentRenderer().ResetCameraClippingRange()
                self.GetInteractor().GetRenderWindow().Render()

        style = CustomTrackballStyle()
        self.render_window_interactor.SetInteractorStyle(style)

        # Add grid for orientation
        # self._add_grid()

        # Add key press callback for wireframe toggle
        self.render_window_interactor.AddObserver(
            "KeyPressEvent", self._key_press_callback
        )

        # Initialize interactor
        self.render_window_interactor.Initialize()

        # 添加文本显示
        self.text_actor = vtk.vtkTextActor()
        self.text_actor.SetInput(
            "PyDEM Visualization\n按 'W' 切换线框模式\n按 'C' 切换颜色模式\n按 'R' 重置视图\n按 'Q' 退出"
        )
        self.text_actor.GetTextProperty().SetFontSize(14)
        self.text_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.text_actor.SetPosition(10, 10)
        self.renderer.AddActor2D(self.text_actor)

        # 添加状态信息显示
        self.status_actor = vtk.vtkTextActor()
        self.status_actor.SetInput("加载中...")
        self.status_actor.GetTextProperty().SetFontSize(14)
        self.status_actor.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.status_actor.SetPosition(10, self.render_window.GetSize()[1] - 30)
        self.renderer.AddActor2D(self.status_actor)

        print("VTK Renderer initialized successfully")

    def _start_update_timer(self):
        """Start a timer to update the scene periodically"""

        def update_callback(obj, event):
            if not self.running:
                return
            self._update_scene()

        # Create a repeating timer
        self.render_window_interactor.CreateRepeatingTimer(
            int(self.update_interval * 1000)
        )
        self.render_window_interactor.AddObserver("TimerEvent", update_callback)

    def _key_press_callback(self, obj, event):
        """处理按键事件"""
        key = obj.GetKeySym().lower()

        if key == "q":
            # 退出
            self.stop()
        elif key == "w":
            # 切换线框模式
            self.show_wireframe = not self.show_wireframe
            self._update_representation()
        elif key == "c":
            # 循环切换颜色模式
            modes = ["uniform", "velocity", "type"]
            current_idx = modes.index(self.color_mode)
            self.color_mode = modes[(current_idx + 1) % len(modes)]
            self._update_colors()
        elif key == "r":
            # 重置视图
            self._camera_initialized = False  # 强制重新初始化相机
            self._reset_camera()

    def _update_scene(self):
        """更新场景中的当前粒子数据"""
        with self.lock:
            if not self.omega.scene or not self.omega.scene.field:
                return

            # 获取所有粒子
            particles = self.omega.scene.field.particles

            # 跟踪现有粒子ID以检测已移除的粒子
            current_ids = set()

            # 检查是否需要重置相机
            reset_camera = len(self.particle_actors) == 0 and len(particles) > 0

            # 如果启用了自动缩放，计算合适的缩放因子
            if self.auto_scale and particles:
                new_scale = self._calculate_scale_factor()
                if (
                    abs(new_scale - self.scale_factor) / self.scale_factor > 0.1
                ):  # 如果缩放因子变化超过10%
                    self.scale_factor = new_scale
                    print(f"自动调整缩放因子为: {self.scale_factor}")
                    # 需要更新所有现有的actor
                    for pid in self.particle_actors:
                        self._update_particle(pid, particles[pid])

            # 更新或添加每个粒子的actor
            for i, p in enumerate(particles):
                if not p or not p.shape:
                    continue

                current_ids.add(i)

                if i in self.particle_actors:
                    # 更新现有actor
                    self._update_particle(i, p)
                else:
                    # 创建新actor
                    self._add_particle(i, p)
                    print(
                        f"添加粒子 {i}, 类型: {p.shape.__class__.__name__}, 位置: {p.shape.nodes[0].pos}"
                    )

            # 移除不再存在的粒子的actor
            for pid in list(self.particle_actors.keys()):
                if pid not in current_ids:
                    self._remove_particle(pid)

            # 仅在需要时重置相机
            if reset_camera:
                self._reset_camera()

            # 更新状态信息
            if hasattr(self, "status_actor"):
                status_text = f"时间: {self.omega.scene.time:.4f} | "
                status_text += f"步数: {self.omega.scene.step} | "
                status_text += f"粒子数: {len(particles)} | "
                status_text += f"缩放因子: {self.scale_factor:.1f} | "
                status_text += f"颜色模式: {self.color_mode}"

                self.status_actor.SetInput(status_text)

            # 渲染场景
            self.render_window.Render()

    def _add_particle(self, pid, particle):
        """Add a new particle to the scene"""
        shape_type = particle.shape.__class__.__name__

        # Create actor based on shape type
        if shape_type == "Sphere":
            actor = self._create_sphere_actor(particle.shape)
        elif shape_type == "Wall":
            actor = self._create_wall_actor(particle.shape)
        elif shape_type == "Facet":
            actor = self._create_facet_actor(particle.shape)
        elif shape_type == "Capsule":
            actor = self._create_capsule_actor(particle.shape)
        elif shape_type == "Ellipsoid":
            actor = self._create_ellipsoid_actor(particle.shape)
        elif shape_type == "Cone":
            actor = self._create_cone_actor(particle.shape)
        elif shape_type == "InfCylinder":
            actor = self._create_infcylinder_actor(particle.shape)
        else:
            print(f"Warning: Unsupported shape type: {shape_type}")
            return

        # Set color based on current mode
        color = self._get_particle_color(particle, shape_type)
        actor.GetProperty().SetColor(color[0], color[1], color[2])

        # Set representation based on wireframe setting
        if self.show_wireframe:
            actor.GetProperty().SetRepresentationToWireframe()

        # Add actor to renderer
        self.renderer.AddActor(actor)

        # Store actor
        self.particle_actors[pid] = actor

    def _update_particle(self, pid, particle):
        """Update an existing particle"""
        actor = self.particle_actors[pid]
        shape_type = particle.shape.__class__.__name__

        # Update position and orientation
        if shape_type == "Sphere":
            self._update_sphere_actor(actor, particle.shape)
        elif shape_type == "Wall":
            self._update_wall_actor(actor, particle.shape)
        elif shape_type == "Facet":
            self._update_facet_actor(actor, particle.shape)
        elif shape_type == "Capsule":
            self._update_capsule_actor(actor, particle.shape)
        elif shape_type == "Ellipsoid":
            self._update_ellipsoid_actor(actor, particle.shape)
        elif shape_type == "Cone":
            self._update_cone_actor(actor, particle.shape)
        elif shape_type == "InfCylinder":
            self._update_infcylinder_actor(actor, particle.shape)

        # Update color if using velocity mode
        if self.color_mode == "velocity":
            color = self._get_particle_color(particle, shape_type)
            actor.GetProperty().SetColor(color[0], color[1], color[2])

    def _remove_particle(self, pid):
        """Remove a particle from the scene"""
        actor = self.particle_actors.pop(pid)
        self.renderer.RemoveActor(actor)

    def _update_representation(self):
        """Update representation mode for all actors"""
        for actor in self.particle_actors.values():
            if self.show_wireframe:
                actor.GetProperty().SetRepresentationToWireframe()
            else:
                actor.GetProperty().SetRepresentationToSurface()

    def _update_colors(self):
        """Update colors for all particles"""
        if not self.omega.scene or not self.omega.scene.field:
            return

        particles = self.omega.scene.field.particles

        for pid, actor in self.particle_actors.items():
            if pid < len(particles) and particles[pid]:
                particle = particles[pid]
                shape_type = particle.shape.__class__.__name__
                color = self._get_particle_color(particle, shape_type)
                actor.GetProperty().SetColor(color[0], color[1], color[2])

    def _get_particle_color(self, particle, shape_type):
        """Get color for a particle based on current color mode"""
        if self.color_mode == "velocity":
            # Color based on velocity magnitude
            if hasattr(particle, "vel"):
                vel = particle.vel
                vel_mag = np.sqrt(vel[0] ** 2 + vel[1] ** 2 + vel[2] ** 2)
                # Map velocity to color (blue to red)
                r = min(1.0, vel_mag)
                g = 0.2
                b = max(0.0, 1.0 - vel_mag)
                return [r, g, b]

        elif self.color_mode == "type":
            # Color based on shape type
            type_colors = {
                "Sphere": [0.7, 0.7, 0.7],
                "Wall": [0.5, 0.5, 0.5],
                "Facet": [0.9, 0.5, 0.2],
                "Capsule": [0.2, 0.9, 0.5],
                "Ellipsoid": [0.9, 0.2, 0.5],
                "Cone": [0.8, 0.4, 0.1],
                "InfCylinder": [0.3, 0.7, 0.9],
            }
            return type_colors.get(shape_type, [0.7, 0.7, 0.7])

        # Default uniform color
        return [0.7, 0.7, 0.7]

    def _add_axes(self):
        """Add coordinate axes to the scene"""
        axes = vtk.vtkAxesActor()
        axes.SetTotalLength(0.01, 0.01, 0.01)  # Small axes
        axes.SetShaftType(0)  # Cylinder shafts
        axes.SetCylinderRadius(0.03)
        axes.SetConeRadius(0.1)

        # Add labels
        axes.GetXAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetYAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()
        axes.GetZAxisCaptionActor2D().GetTextActor().SetTextScaleModeToNone()

        self.renderer.AddActor(axes)

    def _add_grid(self):
        """Add a grid to the scene for orientation"""
        # Create a grid
        grid = vtk.vtkPlaneSource()
        grid.SetXResolution(20)
        grid.SetYResolution(20)
        grid.SetOrigin(-10, -10, 0)
        grid.SetPoint1(10, -10, 0)
        grid.SetPoint2(-10, 10, 0)

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(grid.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToWireframe()
        actor.GetProperty().SetColor(0.7, 0.7, 0.7)
        actor.GetProperty().SetOpacity(0.5)

        # Add to renderer
        self.renderer.AddActor(actor)

    # Shape-specific actor creation and update methods

    def _create_sphere_actor(self, shape):
        """Create a VTK actor for a sphere"""
        pos = shape.nodes[0].pos
        radius = shape.radius

        # Debug print to see sphere position and radius
        print(f"Creating sphere at position {pos} with radius {radius}")
        print(
            f"Scaled position: {[p * self.scale_factor for p in pos]}, scaled radius: {radius * self.scale_factor}"
        )

        # Create sphere source
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(radius * self.scale_factor)
        sphere.SetPhiResolution(20)
        sphere.SetThetaResolution(20)
        sphere.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        return actor

    def _update_sphere_actor(self, actor, shape):
        """Update a sphere actor"""
        pos = shape.nodes[0].pos

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

    def _create_wall_actor(self, shape):
        """Create a VTK actor for a wall"""
        pos = shape.nodes[0].pos
        axis = shape.axis

        # Create plane source (100x100 units)
        plane = vtk.vtkPlaneSource()
        plane.SetOrigin(-50, -50, 0)
        plane.SetPoint1(50, -50, 0)
        plane.SetPoint2(-50, 50, 0)
        plane.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(plane.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Set orientation based on axis
        if axis == 0:  # X-axis
            actor.RotateY(90)
        elif axis == 1:  # Y-axis
            actor.RotateX(90)
        # Z-axis is default orientation

        return actor

    def _update_wall_actor(self, actor, shape):
        """Update a wall actor"""
        pos = shape.nodes[0].pos

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

    def _create_facet_actor(self, shape):
        """Create a VTK actor for a facet"""
        # Get the three vertices of the facet
        p1 = shape.nodes[0].pos
        p2 = shape.nodes[1].pos
        p3 = shape.nodes[2].pos

        # Create points
        points = vtk.vtkPoints()
        points.InsertNextPoint(
            p1[0] * self.scale_factor,
            p1[1] * self.scale_factor,
            p1[2] * self.scale_factor,
        )
        points.InsertNextPoint(
            p2[0] * self.scale_factor,
            p2[1] * self.scale_factor,
            p2[2] * self.scale_factor,
        )
        points.InsertNextPoint(
            p3[0] * self.scale_factor,
            p3[1] * self.scale_factor,
            p3[2] * self.scale_factor,
        )

        # Create triangle
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, 0)
        triangle.GetPointIds().SetId(1, 1)
        triangle.GetPointIds().SetId(2, 2)

        # Create cell array
        triangles = vtk.vtkCellArray()
        triangles.InsertNextCell(triangle)

        # Create polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetPolys(triangles)

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        return actor

    def _update_facet_actor(self, actor, shape):
        """Update a facet actor"""
        # Get the three vertices of the facet
        p1 = shape.nodes[0].pos
        p2 = shape.nodes[1].pos
        p3 = shape.nodes[2].pos

        # Create new points
        points = vtk.vtkPoints()
        points.InsertNextPoint(
            p1[0] * self.scale_factor,
            p1[1] * self.scale_factor,
            p1[2] * self.scale_factor,
        )
        points.InsertNextPoint(
            p2[0] * self.scale_factor,
            p2[1] * self.scale_factor,
            p2[2] * self.scale_factor,
        )
        points.InsertNextPoint(
            p3[0] * self.scale_factor,
            p3[1] * self.scale_factor,
            p3[2] * self.scale_factor,
        )

        # Update polydata
        polydata = actor.GetMapper().GetInput()
        polydata.SetPoints(points)
        polydata.Modified()

    def _create_capsule_actor(self, shape):
        """Create a VTK actor for a capsule"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        radius = shape.radius
        shaft = shape.shaft

        # Create cylinder for shaft
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(radius * self.scale_factor)
        cylinder.SetHeight(shaft * self.scale_factor)
        cylinder.SetResolution(20)
        cylinder.CappingOff()
        cylinder.Update()

        # Create spheres for ends
        sphere1 = vtk.vtkSphereSource()
        sphere1.SetRadius(radius * self.scale_factor)
        sphere1.SetPhiResolution(20)
        sphere1.SetThetaResolution(20)
        sphere1.Update()

        sphere2 = vtk.vtkSphereSource()
        sphere2.SetRadius(radius * self.scale_factor)
        sphere2.SetPhiResolution(20)
        sphere2.SetThetaResolution(20)
        sphere2.Update()

        # Position spheres at ends of cylinder
        transform1 = vtk.vtkTransform()
        transform1.Translate(0, shaft / 2 * self.scale_factor, 0)

        transform2 = vtk.vtkTransform()
        transform2.Translate(0, -shaft / 2 * self.scale_factor, 0)

        transformFilter1 = vtk.vtkTransformPolyDataFilter()
        transformFilter1.SetInputConnection(sphere1.GetOutputPort())
        transformFilter1.SetTransform(transform1)

        transformFilter2 = vtk.vtkTransformPolyDataFilter()
        transformFilter2.SetInputConnection(sphere2.GetOutputPort())
        transformFilter2.SetTransform(transform2)

        # Combine cylinder and spheres
        append = vtk.vtkAppendPolyData()
        append.AddInputConnection(cylinder.GetOutputPort())
        append.AddInputConnection(transformFilter1.GetOutputPort())
        append.AddInputConnection(transformFilter2.GetOutputPort())

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(append.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Set orientation from quaternion
        self._set_orientation_from_quaternion(actor, ori)

        return actor

    def _update_capsule_actor(self, actor, shape):
        """Update a capsule actor"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Update orientation
        self._set_orientation_from_quaternion(actor, ori)

    def _create_ellipsoid_actor(self, shape):
        """Create a VTK actor for an ellipsoid"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        semi_axes = shape.semiAxes

        # Create sphere source
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(1.0)  # Unit sphere
        sphere.SetPhiResolution(20)
        sphere.SetThetaResolution(20)
        sphere.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Set orientation from quaternion
        self._set_orientation_from_quaternion(actor, ori)

        # Set scale for semi-axes
        actor.SetScale(
            semi_axes[0] * self.scale_factor,
            semi_axes[1] * self.scale_factor,
            semi_axes[2] * self.scale_factor,
        )

        return actor

    def _update_ellipsoid_actor(self, actor, shape):
        """Update an ellipsoid actor"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        semi_axes = shape.semiAxes

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Update orientation
        self._set_orientation_from_quaternion(actor, ori)

        # Update scale
        actor.SetScale(
            semi_axes[0] * self.scale_factor,
            semi_axes[1] * self.scale_factor,
            semi_axes[2] * self.scale_factor,
        )

    def _create_cone_actor(self, shape):
        """Create a VTK actor for a cone"""
        pos = shape.nodes[0].pos
        radius = shape.radius

        # Create cone source
        cone = vtk.vtkConeSource()
        cone.SetRadius(radius * self.scale_factor)
        cone.SetHeight(2 * radius * self.scale_factor)
        cone.SetResolution(20)
        cone.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cone.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        return actor

    def _update_cone_actor(self, actor, shape):
        """Update a cone actor"""
        pos = shape.nodes[0].pos

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

    def _create_infcylinder_actor(self, shape):
        """Create a VTK actor for an infinite cylinder"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori
        radius = shape.radius

        # Create cylinder source (long but finite)
        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(radius * self.scale_factor)
        cylinder.SetHeight(100 * self.scale_factor)  # Very long
        cylinder.SetResolution(20)
        cylinder.Update()

        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(cylinder.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Set position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Set orientation from quaternion
        self._set_orientation_from_quaternion(actor, ori)

        return actor

    def _update_infcylinder_actor(self, actor, shape):
        """Update an infinite cylinder actor"""
        pos = shape.nodes[0].pos
        ori = shape.nodes[0].ori

        # Update position
        actor.SetPosition(
            pos[0] * self.scale_factor,
            pos[1] * self.scale_factor,
            pos[2] * self.scale_factor,
        )

        # Update orientation
        self._set_orientation_from_quaternion(actor, ori)

    def _set_orientation_from_quaternion(self, actor, quaternion):
        """Set actor orientation from a quaternion"""
        # Convert quaternion to axis-angle
        w, x, y, z = quaternion.w, quaternion.x, quaternion.y, quaternion.z

        # Compute angle and axis
        angle = 2.0 * np.arccos(w) * 180.0 / np.pi

        if angle > 0.01:  # Avoid division by zero
            axis_x = x / np.sqrt(1 - w * w)
            axis_y = y / np.sqrt(1 - w * w)
            axis_z = z / np.sqrt(1 - w * w)
        else:
            # Default axis if angle is very small
            axis_x, axis_y, axis_z = 0, 0, 1

        # Reset orientation
        actor.SetOrientation(0, 0, 0)
        actor.RotateWXYZ(angle, axis_x, axis_y, axis_z)

    def _reset_camera(self):
        """重置相机以显示所有actor"""
        # 保存当前相机状态
        camera = self.renderer.GetActiveCamera()
        old_position = camera.GetPosition()
        old_focal_point = camera.GetFocalPoint()
        old_view_up = camera.GetViewUp()

        # 重置相机以显示所有内容
        self.renderer.ResetCamera()

        # 稍微缩小以确保所有内容可见
        camera.Zoom(0.9)

        # 如果这是初始视图，调整为更好的角度
        if not hasattr(self, "_camera_initialized"):
            # 设置为等轴测视图
            camera.Elevation(30)
            camera.Azimuth(30)
            self._camera_initialized = True

        # 更新渲染窗口
        self.render_window.Render()

    def _calculate_scale_factor(self):
        """根据场景中最小颗粒计算合适的缩放因子"""
        if (
            not self.omega.scene
            or not self.omega.scene.field
            or not self.omega.scene.field.particles
        ):
            return 1.0

        # 找到最小半径
        min_radius = float("inf")
        for p in self.omega.scene.field.particles:
            if p and p.shape and hasattr(p.shape, "radius"):
                min_radius = min(min_radius, p.shape.radius)

        if min_radius == float("inf"):
            return 1.0

        # 计算合适的缩放因子，使最小颗粒至少有min_visible_size像素大小
        # 获取当前视口大小
        view_size = self.render_window.GetSize()
        viewport_size = min(view_size[0], view_size[1])

        # 计算缩放因子，使最小颗粒在屏幕上至少有min_visible_size像素
        camera = self.renderer.GetActiveCamera()
        view_angle = camera.GetViewAngle() * np.pi / 180.0  # 转换为弧度
        distance = camera.GetDistance()

        # 在给定距离和视角下，计算视口中1像素对应的世界单位
        world_per_pixel = 2.0 * distance * np.tan(view_angle / 2.0) / viewport_size

        # 计算所需的缩放因子
        required_scale = self.min_visible_size * world_per_pixel / min_radius

        # 限制缩放因子在合理范围内
        return max(1.0, min(required_scale, 10000.0))
