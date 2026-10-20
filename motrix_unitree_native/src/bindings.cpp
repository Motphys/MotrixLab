// Copyright Motphys Technology Co., Ltd. 2025, 2026
// SPDX-License-Identifier: Apache-2.0

#include "backend.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using motrix_unitree_native::G1Backend;
using motrix_unitree_native::NativeTimeoutError;
using motrix_unitree_native::StateSnapshot;

namespace {

template <std::size_t N>
std::array<float, N> array_from_numpy(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& value,
    const char* name) {
  if (value.ndim() != 1 || static_cast<std::size_t>(value.shape(0)) != N) {
    throw std::invalid_argument(std::string(name) + " has an unexpected shape");
  }
  std::array<float, N> result{};
  std::memcpy(result.data(), value.data(), sizeof(float) * N);
  for (const float item : result) {
    if (!std::isfinite(item)) throw std::invalid_argument(std::string(name) + " contains a non-finite value");
  }
  return result;
}

py::array_t<float> array_to_numpy(const float* data, std::size_t count) {
  py::array_t<float> result(count);
  std::memcpy(result.mutable_data(), data, sizeof(float) * count);
  return result;
}

py::dict snapshot_to_dict(const StateSnapshot& snapshot) {
  py::dict result;
  result["sequence"] = snapshot.sequence;
  result["receive_time_ns"] = snapshot.receive_time_ns;
  result["joint_position"] = array_to_numpy(snapshot.joint_position.data(), snapshot.joint_position.size());
  result["joint_velocity"] = array_to_numpy(snapshot.joint_velocity.data(), snapshot.joint_velocity.size());
  result["quaternion_xyzw"] = array_to_numpy(snapshot.quaternion_xyzw.data(), snapshot.quaternion_xyzw.size());
  result["gyro"] = array_to_numpy(snapshot.gyro.data(), snapshot.gyro.size());
  result["accelerometer"] = array_to_numpy(snapshot.accelerometer.data(), snapshot.accelerometer.size());
  result["wireless_remote"] = py::bytes(
      reinterpret_cast<const char*>(snapshot.wireless_remote.data()), snapshot.wireless_remote.size());
  return result;
}

}  // namespace

PYBIND11_MODULE(_native, module) {
  module.doc() = "Native Unitree SDK2 transport for Motrix deployment";

  // Keep the Python contract explicit: callers can catch builtins.TimeoutError.
  py::register_exception<NativeTimeoutError>(module, "TimeoutError", PyExc_TimeoutError);

  py::class_<G1Backend>(module, "G1Backend")
      .def(
          py::init<std::string, int, std::string, std::string, std::vector<int>, int, int, int, double, double,
                   double, double, bool, int>(),
          py::arg("network_interface"), py::arg("domain_id"), py::arg("lowcmd_topic"),
          py::arg("lowstate_topic"), py::arg("motor_indices"), py::arg("mode_pr") = -1,
          py::arg("expected_mode_machine") = -1, py::arg("motor_mode") = 1,
          py::arg("damping_kd") = 3.0, py::arg("command_period_s") = 0.001,
          py::arg("command_timeout_s") = 0.1, py::arg("state_timeout_s") = 0.1,
          py::arg("validate_crc") = true, py::arg("subscriber_queue_depth") = 10)
      .def("open", [](G1Backend& backend, double connect_timeout_s) {
        py::gil_scoped_release release;
        backend.open(connect_timeout_s);
      }, py::arg("connect_timeout_s") = 5.0)
      .def("read_state", [](G1Backend& backend, double timeout_s, std::int64_t after_sequence) {
        StateSnapshot snapshot;
        {
          py::gil_scoped_release release;
          snapshot = backend.read_state(timeout_s, after_sequence);
        }
        return snapshot_to_dict(snapshot);
      }, py::arg("timeout_s") = 0.1, py::arg("after_sequence") = -1)
      .def("write_command", [](
          G1Backend& backend,
          const py::array_t<float, py::array::c_style | py::array::forcecast>& position,
          const py::array_t<float, py::array::c_style | py::array::forcecast>& velocity,
          const py::array_t<float, py::array::c_style | py::array::forcecast>& torque,
          const py::array_t<float, py::array::c_style | py::array::forcecast>& kp,
          const py::array_t<float, py::array::c_style | py::array::forcecast>& kd) {
        backend.write_command(
            array_from_numpy<29>(position, "joint_position"), array_from_numpy<29>(velocity, "joint_velocity"),
            array_from_numpy<29>(torque, "feedforward_torque"), array_from_numpy<29>(kp, "kp"),
            array_from_numpy<29>(kd, "kd"));
      }, py::arg("joint_position"), py::arg("joint_velocity"), py::arg("feedforward_torque"), py::arg("kp"),
      py::arg("kd"))
      .def("write_torque", [](G1Backend& backend,
                               const py::array_t<float, py::array::c_style | py::array::forcecast>& torque) {
        backend.write_torque(array_from_numpy<29>(torque, "torque"));
      })
      .def("set_enabled", &G1Backend::set_enabled, py::arg("enabled"))
      .def("stop", [](G1Backend& backend) {
        py::gil_scoped_release release;
        backend.stop();
      })
      .def("close", [](G1Backend& backend) {
        py::gil_scoped_release release;
        backend.close();
      })
      .def("health", [](const G1Backend& backend) {
        py::dict result;
        result["healthy"] = backend.healthy();
        result["reason"] = backend.health_reason();
        result["last_successful_communication_ns"] = backend.last_communication_ns();
        return result;
      });
}
