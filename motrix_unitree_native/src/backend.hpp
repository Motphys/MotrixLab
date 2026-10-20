// Copyright Motphys Technology Co., Ltd. 2025, 2026
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

namespace motrix_unitree_native {

/// Raised when no fresh LowState (or no first LowState) arrives in time.
/// Registered as a subclass of Python ``TimeoutError``.
class NativeTimeoutError : public std::runtime_error {
 public:
  explicit NativeTimeoutError(const std::string& message) : std::runtime_error(message) {}
};

struct StateSnapshot {
  std::uint64_t sequence = 0;
  std::int64_t receive_time_ns = 0;
  std::array<float, 29> joint_position{};
  std::array<float, 29> joint_velocity{};
  std::array<float, 4> quaternion_xyzw{};
  std::array<float, 3> gyro{};
  std::array<float, 3> accelerometer{};
  std::array<std::uint8_t, 40> wireless_remote{};
};

class G1Backend {
 public:
  G1Backend(
      std::string network_interface, int domain_id, std::string lowcmd_topic,
      std::string lowstate_topic, std::vector<int> motor_indices, int mode_pr,
      int expected_mode_machine, int motor_mode, double damping_kd,
      double command_period_s, double command_timeout_s, double state_timeout_s,
      bool validate_crc, int subscriber_queue_depth);
  ~G1Backend();

  void open(double connect_timeout_s);
  StateSnapshot read_state(double timeout_s, std::int64_t after_sequence);
  void write_command(const std::array<float, 29>& position,
                     const std::array<float, 29>& velocity,
                     const std::array<float, 29>& torque,
                     const std::array<float, 29>& kp,
                     const std::array<float, 29>& kd);
  void write_torque(const std::array<float, 29>& torque);
  void set_enabled(bool enabled);
  void stop();
  void close();

  bool healthy() const;
  std::string health_reason() const;
  std::int64_t last_communication_ns() const;

 private:
  using LowCmd = unitree_hg::msg::dds_::LowCmd_;
  using LowState = unitree_hg::msg::dds_::LowState_;
  using Publisher = unitree::robot::ChannelPublisher<LowCmd>;
  using Subscriber = unitree::robot::ChannelSubscriber<LowState>;

  static std::int64_t now_ns();
  static std::uint32_t crc32_core(std::uint32_t* ptr, std::uint32_t length);
  void lowstate_handler(const void* message);
  void command_writer_loop();
  void initialize_command_locked();
  void set_damping_locked();
  /// CRC and publish one command without holding mutex_.
  bool publish(const LowCmd& command);
  bool valid_crc(const LowState& state) const;
  bool valid_mode(const LowState& state) const;

  std::string network_interface_;
  int domain_id_;
  std::string lowcmd_topic_;
  std::string lowstate_topic_;
  std::array<int, 29> motor_indices_{};
  int mode_pr_;
  int expected_mode_machine_;
  int motor_mode_;
  float damping_kd_;
  std::chrono::nanoseconds command_period_;
  std::chrono::nanoseconds command_timeout_;
  std::chrono::nanoseconds state_timeout_;
  bool validate_crc_;
  int subscriber_queue_depth_;

  std::shared_ptr<Publisher> publisher_;
  std::shared_ptr<Subscriber> subscriber_;
  LowCmd command_{};

  mutable std::mutex mutex_;
  std::condition_variable state_condition_;
  StateSnapshot latest_state_{};
  bool have_state_ = false;
  bool mode_ok_ = false;
  bool invalid_mode_at_connect_ = false;
  std::uint64_t sequence_ = 0;
  std::int64_t last_state_receive_ns_ = 0;
  std::int64_t last_command_update_ns_ = 0;
  std::int64_t last_communication_ns_ = 0;
  std::string state_error_;
  std::string publish_error_;
  int publish_failures_ = 0;
  bool enabled_ = false;
  bool stopped_ = false;
  bool opened_ = false;
  bool closed_ = false;

  std::atomic<bool> writer_stop_{false};
  std::thread writer_thread_;
};

}  // namespace motrix_unitree_native
