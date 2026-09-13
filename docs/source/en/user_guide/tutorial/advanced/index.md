# Advanced Topics

Capabilities beyond the basic training flow, read as needed:

- [Exporting ONNX Policies](export_onnx.md): export a trained policy into a standalone
  ONNX model for inference-side integration;
- [Go2 Flat Walking: from Training to Hardware Deployment](motrix_deploy.md): a complete
  walkthrough of training, artifact export, MuJoCo checks, and hardware runs;
- [Command Input Architecture](input_devices_and_bindings.md): decouples "how input is
  read" from "what targets the policy needs", connecting input devices and task commands;
- [Adding a Custom Training Backend](custom_training_backend.md): integrate a new RL
  framework, algorithm, or training backend.

```{toctree}
:hidden:

export_onnx
motrix_deploy
input_devices_and_bindings
custom_training_backend
```
