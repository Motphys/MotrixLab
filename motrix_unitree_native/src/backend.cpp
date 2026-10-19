#include "backend.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <unitree/robot/channel/channel_factory.hpp>

namespace motrix_unitree_native {

namespace {
constexpr int kMotorCount = 29;
constexpr int kMotorSlotCount = 35;
constexpr float kPositionStop = 0.0F;
constexpr float kVelocityStop = 0.0F;
// One dropped DDS sample is a hiccup, not a failure: only a sustained run of
// failed publishes makes the backend unhealthy.
constexpr int kPublishFailureLimit = 20;
}  // namespace

G1Backend::G1Backend(
    std::string network_interface, int domain_id, std::string lowcmd_topic,
    std::string lowstate_topic, std::vector<int> motor_indices, int mode_pr,
    int expected_mode_machine, int motor_mode, double damping_kd,
    double command_period_s, double command_timeout_s, double state_timeout_s,
    bool validate_crc, int subscriber_queue_depth)
    : network_interface_(std::move(network_interface)),
      domain_id_(domain_id),
      lowcmd_topic_(std::move(lowcmd_topic)),
      lowstate_topic_(std::move(lowstate_topic)),
      mode_pr_(mode_pr),
      expected_mode_machine_(expected_mode_machine),
      motor_mode_(motor_mode),
      damping_kd_(static_cast<float>(damping_kd)),
      command_period_(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(command_period_s))),
      command_timeout_(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(command_timeout_s))),
      state_timeout_(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::duration<double>(state_timeout_s))),
      validate_crc_(validate_crc),
      subscriber_queue_depth_(subscriber_queue_depth) {
  if (motor_indices.size() != kMotorCount) {
    throw std::invalid_argument("G1 native backend requires exactly 29 motor indices");
  }
  std::copy(motor_indices.begin(), motor_indices.end(), motor_indices_.begin());
  std::array<bool, kMotorSlotCount> seen{};
  for (const int index : motor_indices_) {
    if (index < 0 || index >= kMotorSlotCount || seen[static_cast<std::size_t>(index)]) {
      throw std::invalid_argument("G1 motor indices must be unique values in [0, 34]");
    }
    seen[static_cast<std::size_t>(index)] = true;
  }
  if (command_period_ <= std::chrono::nanoseconds::zero() ||
      command_timeout_ <= std::chrono::nanoseconds::zero() ||
      state_timeout_ <= std::chrono::nanoseconds::zero()) {
    throw std::invalid_argument("native backend timing values must be positive");
  }
  if (!std::isfinite(damping_kd) || damping_kd_ <= 0.0F) {
    throw std::invalid_argument("native backend damping_kd must be positive and finite");
  }
}

G1Backend::~G1Backend() { close(); }

std::int64_t G1Backend::now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::uint32_t G1Backend::crc32_core(std::uint32_t* ptr, std::uint32_t length) {
  // Unitree wire CRC: polynomial 0x04c11db7, init 0xFFFFFFFF, over
  // (sizeof(message) >> 2) - 1 little-endian words (the trailing crc word is
  // excluded). Implemented here so the build does not depend on an optional
  // SDK header.
  std::uint32_t crc = 0xFFFFFFFFU;
  constexpr std::uint32_t kPolynomial = 0x04c11db7U;
  for (std::uint32_t i = 0; i < length; ++i) {
    std::uint32_t data = ptr[i];
    std::uint32_t mask = 1U << 31;
    for (int bit = 0; bit < 32; ++bit) {
      if ((crc & 0x80000000U) != 0U) {
        crc = (crc << 1) ^ kPolynomial;
      } else {
        crc <<= 1;
      }
      if ((data & mask) != 0U) {
        crc ^= kPolynomial;
      }
      mask >>= 1;
    }
  }
  return crc;
}

void G1Backend::open(double connect_timeout_s) {
  if (connect_timeout_s <= 0.0 || !std::isfinite(connect_timeout_s)) {
    throw std::invalid_argument("connect_timeout_s must be positive and finite");
  }
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_ && !closed_) {
      throw std::runtime_error("native Unitree backend is already open");
    }
    unitree::robot::ChannelFactory::Instance()->Init(domain_id_, network_interface_);
    subscriber_ = std::make_shared<Subscriber>(lowstate_topic_);
    subscriber_->InitChannel(
        [this](const void* message) { lowstate_handler(message); }, subscriber_queue_depth_);
    publisher_ = std::make_shared<Publisher>(lowcmd_topic_);
    publisher_->InitChannel();
    initialize_command_locked();
    opened_ = true;
    closed_ = false;
    stopped_ = false;
    enabled_ = false;
    state_error_.clear();
    publish_error_.clear();
    publish_failures_ = 0;
    have_state_ = false;
    mode_ok_ = false;
    invalid_mode_at_connect_ = false;
    sequence_ = 0;
    last_command_update_ns_ = now_ns();
  }

  std::unique_lock<std::mutex> lock(mutex_);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(connect_timeout_s));
  while (!have_state_ && !invalid_mode_at_connect_) {
    if (state_condition_.wait_until(lock, deadline) == std::cv_status::timeout) {
      lock.unlock();
      close();
      throw NativeTimeoutError("no valid G1 LowState received before connect timeout");
    }
  }
  if (invalid_mode_at_connect_) {
    lock.unlock();
    close();
    throw std::runtime_error("G1 LowState mode_pr/mode_machine does not match the configured contract");
  }
  lock.unlock();

  writer_stop_.store(false);
  writer_thread_ = std::thread(&G1Backend::command_writer_loop, this);
}

void G1Backend::lowstate_handler(const void* message) {
  const auto* incoming = static_cast<const LowState*>(message);
  if (incoming == nullptr) return;
  LowState state = *incoming;
  if (validate_crc_ && !valid_crc(state)) return;
  if (!valid_mode(state)) {
    // Recoverable: the robot may return to low-level mode. Keep publishing
    // damping while unhealthy and clear this on the next contract-valid frame.
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ok_ = false;
    if (!have_state_) invalid_mode_at_connect_ = true;
    state_error_ = "G1 LowState mode_pr/mode_machine does not match the configured contract";
    state_condition_.notify_all();
    return;
  }

  StateSnapshot snapshot;
  snapshot.receive_time_ns = now_ns();
  for (int i = 0; i < kMotorCount; ++i) {
    const auto motor_index = static_cast<std::size_t>(motor_indices_[static_cast<std::size_t>(i)]);
    snapshot.joint_position[static_cast<std::size_t>(i)] = state.motor_state()[motor_index].q();
    snapshot.joint_velocity[static_cast<std::size_t>(i)] = state.motor_state()[motor_index].dq();
  }
  const auto& q = state.imu_state().quaternion();
  const float w = q[0];
  const float x = q[1];
  const float y = q[2];
  const float z = q[3];
  const float norm = std::sqrt(w * w + x * x + y * y + z * z);
  if (!std::isfinite(norm) || norm <= 1.0e-8F) {
    // Drop the sample instead of latching an error: one corrupt IMU frame must
    // not end the session.
    std::lock_guard<std::mutex> lock(mutex_);
    state_error_ = "G1 LowState contains an invalid IMU quaternion";
    state_condition_.notify_all();
    return;
  }
  snapshot.quaternion_xyzw = {x / norm, y / norm, z / norm, w / norm};
  for (int i = 0; i < 3; ++i) {
    snapshot.gyro[static_cast<std::size_t>(i)] = state.imu_state().gyroscope()[i];
    snapshot.accelerometer[static_cast<std::size_t>(i)] = state.imu_state().accelerometer()[i];
  }
  const auto& remote = state.wireless_remote();
  const auto remote_count = std::min<std::size_t>(snapshot.wireless_remote.size(), remote.size());
  for (std::size_t i = 0; i < remote_count; ++i) snapshot.wireless_remote[i] = remote[i];

  {
    std::lock_guard<std::mutex> lock(mutex_);
    snapshot.sequence = ++sequence_;
    latest_state_ = snapshot;
    have_state_ = true;
    mode_ok_ = true;
    state_error_.clear();
    last_state_receive_ns_ = snapshot.receive_time_ns;
    last_communication_ns_ = snapshot.receive_time_ns;
  }
  state_condition_.notify_all();
}

bool G1Backend::valid_crc(const LowState& state) const {
  auto copy = state;
  return state.crc() == crc32_core(reinterpret_cast<std::uint32_t*>(&copy),
                                   (sizeof(LowState) >> 2U) - 1U);
}

bool G1Backend::valid_mode(const LowState& state) const {
  if (mode_pr_ >= 0 && state.mode_pr() != mode_pr_) return false;
  return expected_mode_machine_ < 0 || state.mode_machine() == expected_mode_machine_;
}

StateSnapshot G1Backend::read_state(double timeout_s, std::int64_t after_sequence) {
  if (timeout_s <= 0.0 || !std::isfinite(timeout_s)) {
    throw std::invalid_argument("state timeout must be positive and finite");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                            std::chrono::duration<double>(timeout_s));
  const auto fresh = [this, after_sequence] {
    return have_state_ &&
           (after_sequence < 0 || static_cast<std::int64_t>(latest_state_.sequence) > after_sequence);
  };
  while (!fresh()) {
    if (state_condition_.wait_until(lock, deadline) == std::cv_status::timeout) {
      std::string message = "no fresh G1 LowState received before timeout";
      const std::string& reason = !state_error_.empty() ? state_error_ : publish_error_;
      if (!reason.empty()) message += ": " + reason;
      throw NativeTimeoutError(message);
    }
  }
  return latest_state_;
}

void G1Backend::write_command(const std::array<float, 29>& position,
                              const std::array<float, 29>& velocity,
                              const std::array<float, 29>& torque,
                              const std::array<float, 29>& kp,
                              const std::array<float, 29>& kd) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!opened_ || closed_) throw std::runtime_error("native Unitree backend is not open");
  for (int i = 0; i < kMotorCount; ++i) {
    auto& motor = command_.motor_cmd()[static_cast<std::size_t>(motor_indices_[static_cast<std::size_t>(i)])];
    motor.mode() = motor_mode_;
    motor.q() = position[static_cast<std::size_t>(i)];
    motor.dq() = velocity[static_cast<std::size_t>(i)];
    motor.kp() = kp[static_cast<std::size_t>(i)];
    motor.kd() = kd[static_cast<std::size_t>(i)];
    motor.tau() = torque[static_cast<std::size_t>(i)];
  }
  enabled_ = true;
  stopped_ = false;
  last_command_update_ns_ = now_ns();
}

void G1Backend::write_torque(const std::array<float, 29>& torque) {
  std::array<float, 29> zeros{};
  write_command(zeros, zeros, torque, zeros, zeros);
}

void G1Backend::set_enabled(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  enabled_ = enabled;
  stopped_ = false;
  if (enabled) {
    // Never publish a kp=0/kd=0 frame between enabling and the first command.
    set_damping_locked();
  }
  last_command_update_ns_ = now_ns();
}

void G1Backend::stop() {
  LowCmd outgoing;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    enabled_ = false;
    stopped_ = true;
    set_damping_locked();
    last_command_update_ns_ = now_ns();
    outgoing = command_;
  }
  // Publish the first damping frame synchronously. The periodic writer keeps
  // repeating it afterwards, but shutdown should not depend on its next tick.
  publish(outgoing);
}

void G1Backend::close() {
  LowCmd outgoing;
  bool publish_damping = false;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_) return;
    if (opened_) {
      enabled_ = false;
      stopped_ = true;
      set_damping_locked();
      outgoing = command_;
      publish_damping = true;
    }
    writer_stop_.store(true);
  }
  if (publish_damping) publish(outgoing);
  if (writer_thread_.joinable()) writer_thread_.join();
  std::lock_guard<std::mutex> lock(mutex_);
  publisher_.reset();
  subscriber_.reset();
  opened_ = false;
  closed_ = true;
}

bool G1Backend::healthy() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!opened_ || closed_ || stopped_) return false;
  if (!state_error_.empty() || !publish_error_.empty()) return false;
  return have_state_ && mode_ok_ && now_ns() - last_state_receive_ns_ <= state_timeout_.count();
}

std::string G1Backend::health_reason() const {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!publish_error_.empty()) return publish_error_;
  if (!state_error_.empty()) return state_error_;
  if (closed_) return "backend is closed";
  if (!opened_) return "backend is not open";
  if (stopped_) return "backend is stopped";
  if (!have_state_ || now_ns() - last_state_receive_ns_ > state_timeout_.count()) return "LowState is stale";
  return "";
}

std::int64_t G1Backend::last_communication_ns() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_communication_ns_;
}

void G1Backend::initialize_command_locked() {
  if (mode_pr_ >= 0) command_.mode_pr() = mode_pr_;
  if (expected_mode_machine_ >= 0) command_.mode_machine() = expected_mode_machine_;
  for (std::size_t i = 0; i < command_.motor_cmd().size(); ++i) {
    auto& motor = command_.motor_cmd()[i];
    motor.mode() = 0;
    motor.q() = kPositionStop;
    motor.dq() = kVelocityStop;
    motor.kp() = 0.0F;
    motor.kd() = 0.0F;
    motor.tau() = 0.0F;
  }
  for (const int index : motor_indices_) {
    auto& motor = command_.motor_cmd()[static_cast<std::size_t>(index)];
    motor.mode() = motor_mode_;
    // The first frame published after open()/enable() must already damp.
    motor.kd() = damping_kd_;
  }
}

void G1Backend::set_damping_locked() {
  for (int i = 0; i < kMotorCount; ++i) {
    const auto index = static_cast<std::size_t>(motor_indices_[static_cast<std::size_t>(i)]);
    auto& motor = command_.motor_cmd()[index];
    motor.mode() = motor_mode_;
    motor.q() = have_state_ ? latest_state_.joint_position[static_cast<std::size_t>(i)] : 0.0F;
    motor.dq() = 0.0F;
    motor.kp() = 0.0F;
    motor.kd() = damping_kd_;
    motor.tau() = 0.0F;
  }
}

bool G1Backend::publish(const LowCmd& command) {
  const std::shared_ptr<Publisher> publisher = publisher_;
  if (!publisher) return false;
  LowCmd message = command;
  message.crc() = crc32_core(reinterpret_cast<std::uint32_t*>(&message),
                             (sizeof(LowCmd) >> 2U) - 1U);
  const bool ok = publisher->Write(message, 0);
  std::lock_guard<std::mutex> lock(mutex_);
  if (ok) {
    last_communication_ns_ = now_ns();
    publish_failures_ = 0;
    publish_error_.clear();
  } else if (++publish_failures_ >= kPublishFailureLimit) {
    publish_error_ = "G1 LowCmd DDS publish failed";
  }
  return ok;
}

void G1Backend::command_writer_loop() {
  auto next = std::chrono::steady_clock::now();
  while (!writer_stop_.load()) {
    next += command_period_;
    LowCmd outgoing;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      const auto now = now_ns();
      const bool state_stale = !have_state_ || now - last_state_receive_ns_ > state_timeout_.count();
      const bool command_stale = now - last_command_update_ns_ > command_timeout_.count();
      if (!enabled_ || stopped_ || !mode_ok_ || state_stale || command_stale) {
        set_damping_locked();
      }
      outgoing = command_;
    }
    // CRC and the DDS write happen outside the lock: the policy loop and
    // read_state() must never wait on transport I/O.
    publish(outgoing);
    std::this_thread::sleep_until(next);
    if (std::chrono::steady_clock::now() > next + command_period_) next = std::chrono::steady_clock::now();
  }
}

}  // namespace motrix_unitree_native
