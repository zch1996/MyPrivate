- [ ] OpenMPSimulator 的初始化
- [ ] FunctorFactory 的初始注册
- [ ] Omega类
- [ ] Scene的初始化

DEMField的初始化
引擎初始化

可视化
OpenGL

I/O

Model Selector
Scene背景运行

# Requirements
- coloredlogs
# OpenGL
an excellent way to visualize and interact with your simulations. 


尽管我们设计了一个更简洁的单进程架构，但在某些情况下仍然需要保留多进程功能：
向后兼容性: 现有的脚本和工作流可能依赖于多进程模式。
UI 响应性: 在进行大规模或复杂模拟时，单进程模式可能导致 UI 卡顿，因为模拟计算和渲染共享同一个进程。
隔离性: 多进程提供了更好的隔离性，如果模拟代码崩溃，不会影响可视化界面。
headless 模式: 在无头服务器上运行时，多进程允许主进程继续执行计算，而可视化进程可以独立运行或被禁用