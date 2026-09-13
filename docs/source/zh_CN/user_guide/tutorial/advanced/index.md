# 进阶主题

基础训练流程之外的能力，按需选读：

- [导出 ONNX 策略](export_onnx.md)：把训练好的策略导出为独立的 ONNX 模型，
  用于推理侧集成；
- [Go2 平地行走：从训练到真机部署](motrix_deploy.md)：完整演示训练策略、导出 artifact、
  MuJoCo 检查与真机运行；
- [指令输入架构](input_devices_and_bindings.md)：把"如何读取输入"与
  "策略需要什么目标"解耦，接入键盘等输入设备与任务指令；
- [扩展自定义训练后端](custom_training_backend.md)：接入新的 RL 框架、算法或训练后端。

```{toctree}
:hidden:

export_onnx
motrix_deploy
input_devices_and_bindings
custom_training_backend
```
