/**
 * @file shared_autonomy.hpp
 * @brief Shared-autonomy layer that sits between the SONIC policy output and
 *        the low-level robot command.
 *
 * Pipeline position:
 *
 *   Human Input -> SONIC policy -> sonic_action -> SharedAutonomyWrapper
 *               -> final_action -> MotorCommand -> LowCmd_ -> robot
 *
 * The wrapper is intentionally a **no-op in this revision**: even when enabled
 * it leaves `final_action` bit-identical to `sonic_action` and the Dex3 hand
 * command bit-identical to the operator's command.  Its purpose right now is
 * to establish the seam, the runtime switch, and the intervention logging that
 * a real assistance policy will later plug into.
 *
 * ## Guarantees
 *
 * - Default constructed state is `disabled`.  With the wrapper disabled the
 *   `Apply*` calls return immediately and the deploy binary behaves exactly as
 *   it did before this file existed.
 * - `Apply*` never allocates, never locks, and never touches the filesystem.
 * - Intervention magnitude (L2 norm of `final - original`) is measured by the
 *   wrapper itself on every `Apply*` call, so it stays correct regardless of
 *   what the caller does with the buffers afterwards.
 * - CSV persistence runs on a dedicated background thread fed by a
 *   pre-allocated ring buffer, so the 50 Hz control thread performs no
 *   blocking I/O.
 *
 * ## Numerics
 *
 * The deploy target is built with `-O3 -ffast-math`, which makes
 * `std::isnan` / `std::isfinite` unreliable (see `motor_gain_scaling.cpp`).
 * Non-finite detection here is therefore done on the raw bit pattern.
 */

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

/**
 * @class SharedAutonomyWrapper
 * @brief Post-policy modulation seam for body and hand commands.
 */
class SharedAutonomyWrapper {
 public:
  static constexpr std::size_t kNumBodyJoints = 29;  ///< SONIC action width (IsaacLab order).
  static constexpr std::size_t kNumHandJoints = 7;   ///< Dex3-1 DoF per hand.

  using BodyAction = std::array<float, kNumBodyJoints>;
  using HandAction = std::array<double, kNumHandJoints>;

  /// L2 norms of the most recent intervention, per channel.
  struct Metrics {
    double body = 0.0;
    double left_hand = 0.0;
    double right_hand = 0.0;
  };

  SharedAutonomyWrapper() = default;

  ~SharedAutonomyWrapper() { StopLogging(); }

  SharedAutonomyWrapper(const SharedAutonomyWrapper&) = delete;
  SharedAutonomyWrapper& operator=(const SharedAutonomyWrapper&) = delete;
  SharedAutonomyWrapper(SharedAutonomyWrapper&&) = delete;
  SharedAutonomyWrapper& operator=(SharedAutonomyWrapper&&) = delete;

  // ---------------------------------------------------------------------
  // Runtime switch
  // ---------------------------------------------------------------------

  void setEnabled(bool enabled) { enabled_.store(enabled, std::memory_order_relaxed); }

  bool isEnabled() const { return enabled_.load(std::memory_order_relaxed); }

  // ---------------------------------------------------------------------
  // Action modulation (control thread, 50 Hz)
  // ---------------------------------------------------------------------

  /**
   * @brief Modulate the 29-DoF SONIC body action in place.
   *
   * @param action  In: `sonic_action` (IsaacLab joint order, residual joint
   *                position as emitted by the decoder).  Out: `final_action`.
   *
   * Current revision performs no modification.  The intervention norm is still
   * measured so the baseline (expected: exactly 0) can be verified from logs.
   */
  void ApplyBodyAction(BodyAction& action) {
    if (!isEnabled()) {
      metrics_.body = 0.0;
      return;
    }

    const BodyAction original = action;

    // ---- assistance goes here (intentionally empty in this revision) ----
    // final_action = sonic_action
    // --------------------------------------------------------------------

    metrics_.body = L2Diff(action, original);
    nonfinite_body_ += CountNonFinite(action);
  }

  /**
   * @brief Modulate one Dex3 hand command in place.
   *
   * @param is_left      True for the left hand, false for the right.
   * @param hand_action  In: operator command (7 DoF).  Out: final command.
   *
   * Downstream, `Dex3Hands::writeOnce()` still applies its max-close-ratio
   * clipping and per-tick delta rate limit; this wrapper does not bypass them.
   */
  void ApplyHandAction(bool is_left, HandAction& hand_action) {
    double& slot = is_left ? metrics_.left_hand : metrics_.right_hand;
    if (!isEnabled()) {
      slot = 0.0;
      return;
    }

    const HandAction original = hand_action;

    // ---- grasp assistance goes here (intentionally empty in this revision) ----
    // final_hand_action = human_hand_action
    // --------------------------------------------------------------------------

    slot = L2Diff(hand_action, original);
    (is_left ? nonfinite_left_hand_ : nonfinite_right_hand_) += CountNonFinite(hand_action);
  }

  /// Intervention norms produced by the most recent `Apply*` calls.
  const Metrics& metrics() const { return metrics_; }

  /// Running count of non-finite values seen leaving the wrapper (should stay 0).
  std::uint64_t nonfinite_count() const {
    return nonfinite_body_ + nonfinite_left_hand_ + nonfinite_right_hand_;
  }

  // ---------------------------------------------------------------------
  // Logging
  // ---------------------------------------------------------------------

  /**
   * @brief Start CSV logging to @p csv_path.
   *
   * Allocates the ring buffer and spawns the writer thread up-front so that
   * `LogStep()` never allocates.  Returns false if the file cannot be opened,
   * in which case logging stays off and control is unaffected.
   */
  bool StartLogging(const std::string& csv_path, std::size_t ring_capacity = 4096) {
    if (logging_.load(std::memory_order_acquire)) { return true; }
    if (csv_path.empty() || ring_capacity == 0) { return false; }

    file_.open(csv_path, std::ios::out | std::ios::trunc);
    if (!file_.good()) {
      std::cerr << "[SharedAutonomy] Failed to open log file: " << csv_path << std::endl;
      return false;
    }
    WriteHeader();

    ring_.assign(ring_capacity, Record{});
    capacity_ = ring_capacity;
    write_seq_.store(0, std::memory_order_relaxed);
    read_seq_ = 0;
    dropped_ = 0;
    log_start_ = std::chrono::steady_clock::now();

    logging_.store(true, std::memory_order_release);
    writer_thread_ = std::thread(&SharedAutonomyWrapper::WriterLoop, this);
    std::cout << "[SharedAutonomy] Logging to " << csv_path << " (ring capacity "
              << ring_capacity << ")" << std::endl;
    return true;
  }

  bool IsLogging() const { return logging_.load(std::memory_order_acquire); }

  /**
   * @brief Record one control step.  Lock-free, allocation-free, no I/O.
   *
   * Safe to call unconditionally; returns immediately when logging is off.
   */
  void LogStep(std::uint64_t index,
               const BodyAction& sonic_action,
               const BodyAction& final_action,
               const HandAction& left_hand_human,
               const HandAction& left_hand_final,
               const HandAction& right_hand_human,
               const HandAction& right_hand_final) {
    if (!logging_.load(std::memory_order_acquire)) { return; }

    const std::uint64_t seq = write_seq_.load(std::memory_order_relaxed);
    Record& r = ring_[static_cast<std::size_t>(seq % capacity_)];

    const auto now = std::chrono::steady_clock::now();
    r.index = index;
    r.t_ms = std::chrono::duration<double, std::milli>(now - log_start_).count();
    r.t_realtime_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();
    r.enabled = isEnabled() ? 1u : 0u;
    r.sonic_action = sonic_action;
    r.final_action = final_action;
    r.left_hand_human = left_hand_human;
    r.left_hand_final = left_hand_final;
    r.right_hand_human = right_hand_human;
    r.right_hand_final = right_hand_final;
    r.body_intervention = metrics_.body;
    r.left_hand_intervention = metrics_.left_hand;
    r.right_hand_intervention = metrics_.right_hand;

    write_seq_.store(seq + 1, std::memory_order_release);
  }

  /// Flush and join the writer thread.  Idempotent.
  void StopLogging() {
    if (!logging_.load(std::memory_order_acquire)) { return; }
    logging_.store(false, std::memory_order_release);
    if (writer_thread_.joinable()) { writer_thread_.join(); }
    Drain();
    if (file_.is_open()) {
      file_.flush();
      file_.close();
    }
    if (dropped_ > 0) {
      std::cerr << "[SharedAutonomy] WARNING: dropped " << dropped_
                << " log records (writer could not keep up)" << std::endl;
    }
  }

 private:
  // Fixed-size POD record; no heap members so the control thread only memcpys.
  struct Record {
    std::uint64_t index = 0;
    double t_ms = 0.0;
    double t_realtime_ms = 0.0;
    std::uint8_t enabled = 0;
    BodyAction sonic_action{};
    BodyAction final_action{};
    HandAction left_hand_human{};
    HandAction left_hand_final{};
    HandAction right_hand_human{};
    HandAction right_hand_final{};
    double body_intervention = 0.0;
    double left_hand_intervention = 0.0;
    double right_hand_intervention = 0.0;
  };

  // ---- numeric helpers (fast-math safe) ----

  static bool IsNonFinite(float v) {
    std::uint32_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7F800000u) == 0x7F800000u;  // exponent all ones -> inf or NaN
  }

  static bool IsNonFinite(double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & 0x7FF0000000000000ull) == 0x7FF0000000000000ull;
  }

  template <typename T, std::size_t N>
  static std::uint64_t CountNonFinite(const std::array<T, N>& a) {
    std::uint64_t n = 0;
    for (std::size_t i = 0; i < N; ++i) {
      if (IsNonFinite(a[i])) { ++n; }
    }
    return n;
  }

  template <typename T, std::size_t N>
  static double L2Diff(const std::array<T, N>& a, const std::array<T, N>& b) {
    double acc = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
      const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
      acc += d * d;
    }
    return std::sqrt(acc);
  }

  // ---- CSV writer (background thread) ----

  void WriteHeader() {
    file_ << "index,time_ms,time_realtime_ms,shared_autonomy_enabled"
          << ",body_intervention,left_hand_intervention,right_hand_intervention";
    for (std::size_t i = 0; i < kNumBodyJoints; ++i) { file_ << ",sonic_action_" << i; }
    for (std::size_t i = 0; i < kNumBodyJoints; ++i) { file_ << ",final_action_" << i; }
    for (std::size_t i = 0; i < kNumHandJoints; ++i) { file_ << ",left_hand_human_" << i; }
    for (std::size_t i = 0; i < kNumHandJoints; ++i) { file_ << ",left_hand_final_" << i; }
    for (std::size_t i = 0; i < kNumHandJoints; ++i) { file_ << ",right_hand_human_" << i; }
    for (std::size_t i = 0; i < kNumHandJoints; ++i) { file_ << ",right_hand_final_" << i; }
    file_ << '\n';
  }

  void WriteRecord(const Record& r) {
    file_.setf(std::ios::fixed, std::ios::floatfield);
    file_ << r.index << ',' << std::setprecision(3) << r.t_ms << ','
          << std::setprecision(3) << r.t_realtime_ms << ','
          << static_cast<int>(r.enabled);
    file_ << std::setprecision(9) << ',' << r.body_intervention << ','
          << r.left_hand_intervention << ',' << r.right_hand_intervention;
    for (float v : r.sonic_action) { file_ << ',' << v; }
    for (float v : r.final_action) { file_ << ',' << v; }
    for (double v : r.left_hand_human) { file_ << ',' << v; }
    for (double v : r.left_hand_final) { file_ << ',' << v; }
    for (double v : r.right_hand_human) { file_ << ',' << v; }
    for (double v : r.right_hand_final) { file_ << ',' << v; }
    file_ << '\n';
  }

  /// Consume everything committed so far.  Writer thread / shutdown only.
  void Drain() {
    const std::uint64_t committed = write_seq_.load(std::memory_order_acquire);
    if (committed - read_seq_ > capacity_) {
      // Producer lapped us: skip the records that were overwritten.
      const std::uint64_t lost = committed - read_seq_ - capacity_;
      dropped_ += lost;
      read_seq_ = committed - capacity_;
    }
    while (read_seq_ < committed) {
      WriteRecord(ring_[static_cast<std::size_t>(read_seq_ % capacity_)]);
      ++read_seq_;
    }
  }

  void WriterLoop() {
    while (logging_.load(std::memory_order_acquire)) {
      Drain();
      file_.flush();
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  }

  // ---- state ----

  std::atomic<bool> enabled_{false};
  Metrics metrics_{};
  std::uint64_t nonfinite_body_ = 0;
  std::uint64_t nonfinite_left_hand_ = 0;
  std::uint64_t nonfinite_right_hand_ = 0;

  // Logging (single producer = control thread, single consumer = writer thread)
  std::atomic<bool> logging_{false};
  std::vector<Record> ring_;
  std::size_t capacity_ = 0;
  std::atomic<std::uint64_t> write_seq_{0};
  std::uint64_t read_seq_ = 0;
  std::uint64_t dropped_ = 0;
  std::chrono::steady_clock::time_point log_start_{};
  std::ofstream file_;
  std::thread writer_thread_;
};
