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
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "math_utils.hpp"
#include "policy_parameters.hpp"

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

  /// Rigid-body pose.  Quaternion is scalar-first (w,x,y,z), per SONIC convention.
  struct Pose {
    std::array<double, 3> position{0.0, 0.0, 0.0};
    std::array<double, 4> orientation{1.0, 0.0, 0.0, 0.0};
  };

  /**
   * @brief Geometric state available to an assistance policy.
   *
   * ## Frame
   *
   * Everything here lives in the **pelvis frame**: origin at the `pelvis` body
   * origin, axes aligned with the pelvis link (MuJoCo convention: +X forward,
   * +Y left, +Z up when the robot stands upright).
   *
   * That frame is chosen because it is the only one this stack can compute
   * exactly: wrist poses come from joint encoders through forward kinematics,
   * and the deploy binary has **no odometry**, so the pelvis has no known world
   * position.  A consequence worth being explicit about: an object pose that is
   * constant in this frame is fixed *relative to the robot's pelvis*, not to the
   * world.  It is meaningful while the base stays put; it is not a world-frame
   * object while the robot walks.  Distances between wrist and object are
   * frame-invariant as long as both sides come from here.
   */
  /**
   * @brief One hand's grasp target and the current wrist's error against it.
   *
   * `relative` is the target in the **object frame** (what the config declares);
   * `pose` is the same target composed into the pelvis frame via
   * `T_target = T_object * T_relative`.
   */
  struct GraspTarget {
    Pose relative;                          ///< Target in object frame (from config).
    Pose pose;                              ///< Target in pelvis frame (composed).
    std::array<double, 3> position_error{0.0, 0.0, 0.0};  ///< target_pos - wrist_pos, pelvis frame (m).
    double position_error_norm = 0.0;       ///< |position_error| (m).
    double orientation_error_rad = 0.0;     ///< Geodesic angle between wrist and target rotation (rad).
    /// Rotation that takes the wrist onto the target, as an axis-angle vector in
    /// the pelvis frame (direction = axis, magnitude = angle in rad).
    std::array<double, 3> orientation_error{0.0, 0.0, 0.0};
  };

  /// Max arm joints per side that the assist may touch (shoulder x3, elbow, wrist x3).
  static constexpr std::size_t kMaxArmJoints = 7;

  /**
   * @brief Translational Jacobian of one wrist w.r.t. that arm's joints.
   *
   * Column j is d(wrist_position)/d(q_j) in the pelvis frame, m/rad.  Only arm
   * joints appear here -- legs, waist and torso are filtered out upstream, so
   * the assist structurally cannot reach them.
   */
  struct ArmJacobian {
    bool valid = false;
    int count = 0;
    std::array<int, kMaxArmJoints> joint_hw_index{};            ///< Hardware joint index per column.
    std::array<std::array<double, 3>, kMaxArmJoints> column{};  ///< d(p_wrist)/d(q_j), pelvis frame, m/rad.
    /// d(wrist angular velocity)/d(qdot_j): for a revolute joint this is simply
    /// the joint axis in the pelvis frame, dimensionless.
    std::array<std::array<double, 3>, kMaxArmJoints> angular{};
  };

  /**
   * @brief Toy bimanual pick sequence.
   *
   *   MANUAL -> ALIGN -> GRASP -> LIFT -> MANUAL
   *
   * MANUAL is pure SONIC; the assist contributes nothing. Every other state has
   * a timeout that falls back to MANUAL, and LIFT additionally latches into
   * ABORT on an anomaly, which disables the assist outright until the operator
   * re-arms it with the U key.
   */
  enum class TaskState : int {
    MANUAL = 0,  ///< SONIC alone. No correction, no hand override.
    ALIGN = 1,   ///< Both wrists near their targets; arm assist pulls them in.
    GRASP = 2,   ///< Position error small enough; hands commanded closed.
    LIFT = 3,    ///< Grasp confirmed; target ramps upward by lift_height.
    ABORT = 4,   ///< Latched fault. Assist is off until re-armed.
  };

  static const char* TaskStateName(TaskState s) {
    switch (s) {
      case TaskState::MANUAL: return "MANUAL";
      case TaskState::ALIGN: return "ALIGN";
      case TaskState::GRASP: return "GRASP";
      case TaskState::LIFT: return "LIFT";
      case TaskState::ABORT: return "ABORT";
    }
    return "?";
  }

  /// Tunables for the toy pick sequence.
  struct TaskConfig {
    bool enabled = false;            ///< Off by default: the assist can run without the sequence.
    double align_enter_radius = 0.10;///< m. Both wrists inside this -> ALIGN.
    double align_exit_radius = 0.20;  ///< m. Either wrist outside this -> back to MANUAL.
    double grasp_radius = 0.025;      ///< m. Both wrists inside this -> GRASP.
    double align_timeout_s = 5.0;     ///< ALIGN gives up after this.
    double grasp_settle_s = 0.6;      ///< Hold the close command this long before judging.
    double grasp_timeout_s = 2.0;     ///< GRASP gives up after this.
    double grasp_contact_margin = 0.15;///< rad. Finger travel shortfall that counts as "holding".
    double lift_height = 0.05;        ///< m. How far up the target is ramped, pelvis +Z.
    double lift_duration_s = 2.0;     ///< Ramp + hold time before declaring success.
  };

  /// Live state of the pick sequence.  Logged verbatim.
  struct TaskStatus {
    TaskState state = TaskState::MANUAL;
    double time_in_state_s = 0.0;
    bool grasp_trigger = false;      ///< Hands are being commanded closed this tick.
    bool lift_trigger = false;       ///< Lift offset is being applied this tick.
    bool grasp_success = false;      ///< Fingers stalled short of the command -> something is held.
    bool task_success = false;       ///< A LIFT completed without losing the grasp.
    double lift_offset_m = 0.0;      ///< Current upward target offset (pelvis +Z).
    double left_grasp_residual = 0.0;  ///< rad, |commanded - measured| on the left hand.
    double right_grasp_residual = 0.0; ///< rad, same for the right.
    std::uint32_t completed_picks = 0; ///< How many LIFTs finished successfully this run.
    bool retry_blocked = false;      ///< Waiting for the operator to withdraw after a failure.
    std::string last_reason;         ///< Why the last transition happened (for the console).
  };

  /// Tunables for the arm alignment assist.  All come from the config file.
  struct AssistConfig {
    bool enabled = false;               ///< Separate from setEnabled(): allows observe-only runs.
    double alpha = 0.3;                 ///< Fixed blend weight (no adaptation in this revision).
    double engage_radius = 0.10;        ///< m. Beyond this, alpha is 0 -- SONIC is on its own.
    double full_radius = 0.02;          ///< m. Inside this, the gate is fully open.
    double max_joint_correction = 0.05; ///< rad. Hard cap on |correction| for any single joint.
    double alpha_slew = 0.02;           ///< Max alpha change per control tick (engage/disengage ramp).
    /**
     * @brief How much of the correction budget goes to orientation, 0..1.
     *
     * Position and orientation gradients carry different units (m^2/rad vs
     * rad), so they are each normalised to unit maximum before blending;
     * this weight is then a plain, unit-free split of the same per-joint
     * budget.  0 reproduces the position-only behaviour exactly.
     */
    double orientation_weight = 0.0;
  };

  /// What the assist actually did this tick, per side.  Logged verbatim.
  struct AssistState {
    double alpha = 0.0;                                  ///< Gated, slew-limited blend weight.
    double distance = 0.0;                               ///< Wrist-to-grasp-target distance (m).
    int count = 0;                                       ///< Joints corrected.
    std::array<int, kMaxArmJoints> joint_hw_index{};     ///< Hardware indices touched.
    std::array<double, kMaxArmJoints> correction_rad{};  ///< Joint-space correction applied (rad).
    double max_abs_correction_rad = 0.0;                 ///< Largest |correction| this tick.
  };

  struct GeometryState {
    bool valid = false;             ///< False until UpdateGeometry() has run with FK available.
    bool object_valid = false;      ///< False until an object pose has been configured.
    bool grasp_targets_valid = false;  ///< False until grasp targets have been configured.
    Pose left_wrist;                ///< left_wrist_yaw_link, pelvis frame.
    Pose right_wrist;               ///< right_wrist_yaw_link, pelvis frame.
    Pose object;                    ///< Configured object pose, pelvis frame.
    double left_wrist_to_object = 0.0;   ///< Euclidean distance (m); 0 when object_valid is false.
    double right_wrist_to_object = 0.0;  ///< Euclidean distance (m); 0 when object_valid is false.
    GraspTarget left_grasp;         ///< Left-hand grasp target and error.
    GraspTarget right_grasp;        ///< Right-hand grasp target and error.
    ArmJacobian left_jacobian;      ///< Left wrist translational Jacobian, arm joints only.
    ArmJacobian right_jacobian;     ///< Right wrist translational Jacobian, arm joints only.
  };

  /// Everything the object config file declares.
  struct ObjectConfig {
    Pose object;                              ///< Object pose, pelvis frame.
    std::array<double, 3> size{0.0, 0.0, 0.0};///< Box extents (m); {0,0,0} if not declared.
    bool has_grasp_targets = false;           ///< True when grasp targets were given or derived.
    Pose left_grasp_relative;                 ///< Left target, object frame.
    Pose right_grasp_relative;                ///< Right target, object frame.
    AssistConfig assist;                      ///< Arm assist tunables (defaults if absent).
    TaskConfig task;                          ///< Pick-sequence tunables (defaults if absent).
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

    if (assist_.enabled && assist_runtime_enabled_.load(std::memory_order_relaxed) &&
        geometry_.valid && geometry_.object_valid && geometry_.grasp_targets_valid &&
        TaskWantsArmAssist()) {
      ApplyArmAssist(action, true);
      ApplyArmAssist(action, false);
    } else {
      DecayAssist(left_assist_);
      DecayAssist(right_assist_);
    }

    metrics_.body = L2Diff(action, original);
    nonfinite_body_ += CountNonFinite(action);
  }

  const AssistConfig& assist_config() const { return assist_; }

  /// Runtime on/off for the assist, independent of the config flag (U key).
  void SetAssistRuntimeEnabled(bool on) {
    assist_runtime_enabled_.store(on, std::memory_order_relaxed);
  }
  bool assist_runtime_enabled() const {
    return assist_runtime_enabled_.load(std::memory_order_relaxed);
  }
  /// True only when the feature is configured AND not switched off at runtime.
  bool assist_active() const { return assist_.enabled && assist_runtime_enabled(); }
  const AssistState& left_assist() const { return left_assist_; }
  const AssistState& right_assist() const { return right_assist_; }

  /**
   * @brief Install assist tunables, clamped to the hard safety caps.
   *
   * The caps are compiled in, not configurable: a typo in the config file must
   * not be able to hand the arms a large correction.
   */
  void SetAssistConfig(const AssistConfig& cfg) {
    assist_ = cfg;
    assist_.alpha = Clamp(assist_.alpha, 0.0, kMaxAlpha);
    assist_.max_joint_correction = Clamp(assist_.max_joint_correction, 0.0, kMaxJointCorrectionRad);
    assist_.engage_radius = std::max(assist_.engage_radius, 0.0);
    assist_.full_radius = Clamp(assist_.full_radius, 0.0, assist_.engage_radius);
    assist_.alpha_slew = Clamp(assist_.alpha_slew, 1e-4, 1.0);
    assist_.orientation_weight = Clamp(assist_.orientation_weight, 0.0, 1.0);
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

    // The sequence takes the hands over in GRASP and holds them through LIFT.
    // Outside those states the operator's command passes through untouched.
    // Dex3Hands::writeOnce() still applies its close-ratio clip and per-tick
    // rate limit downstream, so the takeover cannot step the fingers.
    if (task_cfg_.enabled && task_status_.grasp_trigger) {
      hand_action = ClosedHandPose(is_left);
    }

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
  // Geometric state (pelvis frame — see GeometryState)
  // ---------------------------------------------------------------------

  /// Install the object pose.  Marks the object valid; no effect on control.
  void SetObjectPose(const Pose& pose) {
    geometry_.object = pose;
    geometry_.object_valid = true;
  }

  /**
   * @brief Drop the object pose, disengaging anything that depends on it.
   *
   * Used when a streamed pose goes stale.  A pose that arrived while the robot
   * was somewhere else is worse than no pose at all: it is expressed in the
   * pelvis frame, so it silently drifts as the base moves and the assist would
   * pull toward a position the object never occupied.  Clearing it makes the
   * gate close and the sequence fall back to MANUAL.
   */
  void InvalidateObjectPose() {
    geometry_.object_valid = false;
    geometry_.left_wrist_to_object = 0.0;
    geometry_.right_wrist_to_object = 0.0;
  }

  /**
   * @brief Install the two grasp targets, expressed in the **object frame**.
   *
   * They are composed into the pelvis frame on every UpdateGeometry() call, so
   * a later moving object pose propagates automatically.
   */
  void SetGraspTargets(const Pose& left_relative, const Pose& right_relative) {
    geometry_.left_grasp.relative = left_relative;
    geometry_.right_grasp.relative = right_relative;
    geometry_.grasp_targets_valid = true;
  }

  /// Apply a parsed config in one shot: object pose plus grasp targets if present.
  void ApplyObjectConfig(const ObjectConfig& config) {
    SetObjectPose(config.object);
    if (config.has_grasp_targets) {
      SetGraspTargets(config.left_grasp_relative, config.right_grasp_relative);
    }
    SetAssistConfig(config.assist);
    SetTaskConfig(config.task);
  }

  /**
   * @brief Refresh wrist poses and recompute wrist-to-object distances.
   *
   * Called once per control tick by the deploy loop after forward kinematics.
   * Pure bookkeeping: allocation-free, and it does not touch any action.
   */
  void UpdateGeometry(const Pose& left_wrist, const Pose& right_wrist) {
    UpdateGeometry(left_wrist, right_wrist, ArmJacobian{}, ArmJacobian{});
  }

  /// Overload carrying the arm Jacobians, which the assist needs.
  void UpdateGeometry(const Pose& left_wrist, const Pose& right_wrist,
                      const ArmJacobian& left_jacobian, const ArmJacobian& right_jacobian) {
    geometry_.left_wrist = left_wrist;
    geometry_.right_wrist = right_wrist;
    geometry_.left_jacobian = left_jacobian;
    geometry_.right_jacobian = right_jacobian;
    geometry_.valid = true;
    if (geometry_.object_valid) {
      geometry_.left_wrist_to_object = Distance(left_wrist.position, geometry_.object.position);
      geometry_.right_wrist_to_object = Distance(right_wrist.position, geometry_.object.position);
    } else {
      geometry_.left_wrist_to_object = 0.0;
      geometry_.right_wrist_to_object = 0.0;
    }

    if (geometry_.object_valid && geometry_.grasp_targets_valid) {
      UpdateGraspTarget(geometry_.left_grasp, left_wrist);
      UpdateGraspTarget(geometry_.right_grasp, right_wrist);
    }
  }

  const GeometryState& geometry() const { return geometry_; }

  /// Feed measured Dex3 joint positions; needed to tell a real grasp from a closed empty hand.
  void UpdateHandState(const HandAction& left_measured, const HandAction& right_measured) {
    left_hand_measured_ = left_measured;
    right_hand_measured_ = right_measured;
    hand_state_valid_ = true;
  }

  void SetTaskConfig(const TaskConfig& cfg) {
    task_cfg_ = cfg;
    task_cfg_.align_exit_radius = std::max(task_cfg_.align_exit_radius, task_cfg_.align_enter_radius);
    task_cfg_.grasp_radius = Clamp(task_cfg_.grasp_radius, 0.0, task_cfg_.align_enter_radius);
    task_cfg_.lift_height = Clamp(task_cfg_.lift_height, 0.0, kMaxLiftHeight);
  }

  const TaskConfig& task_config() const { return task_cfg_; }
  const TaskStatus& task_status() const { return task_status_; }

  /**
   * @brief Advance the pick sequence by one control tick.
   *
   * Call once per tick, after UpdateGeometry()/UpdateHandState() and before
   * ApplyBodyAction(). Never touches an action itself: it only decides which
   * state is current, what target offset applies, and whether the hands are
   * being overridden. The Apply* calls read that decision.
   */
  void StepTask(double dt_s) {
    const bool assist_live = isEnabled() && assist_active();

    // Re-arming (U off -> on) is the only way out of a latched ABORT.
    if (task_status_.state == TaskState::ABORT) {
      if (!assist_runtime_enabled()) { abort_rearm_pending_ = true; }
      else if (abort_rearm_pending_) {
        abort_rearm_pending_ = false;
        retry_blocked_ = true;   // withdraw before the sequence may engage again
        EnterState(TaskState::MANUAL, "re-armed by operator");
      }
    }

    // object_valid belongs here: without an object the grasp targets stop being
    // recomputed, so staying in GRASP would keep the hands clamped shut around
    // a pose that is no longer being updated.
    if (!task_cfg_.enabled || !assist_live || !geometry_.valid ||
        !geometry_.object_valid || !geometry_.grasp_targets_valid) {
      if (task_status_.state != TaskState::MANUAL && task_status_.state != TaskState::ABORT) {
        EnterState(TaskState::MANUAL, "assist or geometry unavailable");
      }
      task_lift_offset_ = 0.0;
      task_status_.lift_offset_m = 0.0;
      task_status_.grasp_trigger = false;
      task_status_.lift_trigger = false;
      return;
    }

    task_status_.time_in_state_s += dt_s;
    const double dl = geometry_.left_grasp.position_error_norm;
    const double dr = geometry_.right_grasp.position_error_norm;
    UpdateGraspResiduals();

    switch (task_status_.state) {
      case TaskState::MANUAL:
        // After a failure the sequence will not grab at the same spot again
        // until the operator has pulled clear -- otherwise a timeout inside the
        // engage radius would retry forever.
        if (retry_blocked_) {
          if (dl > task_cfg_.align_exit_radius && dr > task_cfg_.align_exit_radius) {
            retry_blocked_ = false;
            std::cout << "[SharedAutonomy] task re-armed: wrists withdrew, ALIGN can retry"
                      << std::endl;
          }
        } else if (dl <= task_cfg_.align_enter_radius && dr <= task_cfg_.align_enter_radius) {
          EnterState(TaskState::ALIGN, "both wrists entered the engage radius");
        }
        break;

      case TaskState::ALIGN:
        if (dl > task_cfg_.align_exit_radius || dr > task_cfg_.align_exit_radius) {
          EnterState(TaskState::MANUAL, "wrists left the engage radius");
        } else if (dl <= task_cfg_.grasp_radius && dr <= task_cfg_.grasp_radius) {
          EnterState(TaskState::GRASP, "both wrists within the grasp radius");
        } else if (task_status_.time_in_state_s > task_cfg_.align_timeout_s) {
          retry_blocked_ = true;
          EnterState(TaskState::MANUAL, "ALIGN timed out");
        }
        break;

      case TaskState::GRASP:
        if (task_status_.time_in_state_s >= task_cfg_.grasp_settle_s) {
          if (GraspHolding()) {
            task_status_.grasp_success = true;
            EnterState(TaskState::LIFT, "fingers stalled on the object");
          } else if (task_status_.time_in_state_s > task_cfg_.grasp_timeout_s) {
            task_status_.grasp_success = false;
            retry_blocked_ = true;
            EnterState(TaskState::MANUAL, "GRASP timed out with empty hands");
          }
        }
        break;

      case TaskState::LIFT: {
        // Anomaly: the object slipped out, or the geometry went bad.
        if (!GraspHolding()) {
          EnterState(TaskState::ABORT, "grasp lost during LIFT");
          setEnabled(false);
          break;
        }
        const double ramp = task_cfg_.lift_duration_s > 1e-6
                                ? Clamp(task_status_.time_in_state_s / task_cfg_.lift_duration_s,
                                        0.0, 1.0)
                                : 1.0;
        task_lift_offset_ = ramp * task_cfg_.lift_height;
        if (task_status_.time_in_state_s >= task_cfg_.lift_duration_s) {
          task_status_.task_success = true;
          ++task_status_.completed_picks;
          EnterState(TaskState::MANUAL, "LIFT completed");
        }
        break;
      }

      case TaskState::ABORT:
        break;
    }

    if (task_status_.state != TaskState::LIFT) { task_lift_offset_ = 0.0; }
    task_status_.lift_offset_m = task_lift_offset_;
    // Derived from the state that is current *after* this tick's transitions, so
    // entering GRASP closes the hands on the same tick rather than one later.
    task_status_.grasp_trigger =
        task_status_.state == TaskState::GRASP || task_status_.state == TaskState::LIFT;
    task_status_.lift_trigger = task_status_.state == TaskState::LIFT;
    task_status_.retry_blocked = retry_blocked_;
  }

  /// Hand pose commanded during GRASP/LIFT: the midpoint of each Dex3 joint's
  /// range, matching Dex3Hands::close() so the automatic grasp and the manual
  /// helper agree.
  static const HandAction& ClosedHandPose(bool is_left) {
    static const HandAction kLeft = {0.0, 0.163, 0.875, -0.785, -0.875, -0.785, -0.875};
    static const HandAction kRight = {0.0, -0.154, -0.875, 0.785, 0.875, 0.785, 0.875};
    return is_left ? kLeft : kRight;
  }

  /**
   * @brief Parse the object / grasp-target config file.
   *
   * Recognised keys (`#` starts a comment, blank lines ignored):
   * @code
   *   object_frame:      pelvis            # must be "pelvis"; guards against frame mix-ups
   *   object_position:   0.45 0.0 0.10     # x y z, metres, pelvis frame
   *   object_quaternion: 1.0 0.0 0.0 0.0   # w x y z, pelvis frame
   *   object_size:       0.20 0.30 0.15    # box extents x y z, metres (optional)
   *
   *   # Optional explicit grasp targets, in the OBJECT frame.  When omitted and
   *   # object_size is given, they default to the centres of the box's left
   *   # (+Y) and right (-Y) faces with the object's own orientation.
   *   left_grasp_position:    0.0  0.15 0.0
   *   left_grasp_quaternion:  1.0  0.0  0.0 0.0
   *   right_grasp_position:   0.0 -0.15 0.0
   *   right_grasp_quaternion: 1.0  0.0  0.0 0.0
   * @endcode
   *
   * @param[out] config  Filled on success.
   * @param[out] error   Human-readable reason on failure.
   * @return True if the file parsed and object_position / object_quaternion were present.
   */
  static bool LoadObjectConfig(const std::string& path, ObjectConfig& config, std::string& error) {
    std::ifstream in(path);
    if (!in.good()) {
      error = "cannot open '" + path + "'";
      return false;
    }

    bool have_position = false;
    bool have_quaternion = false;
    bool have_size = false;
    bool have_left_pos = false, have_left_quat = false;
    bool have_right_pos = false, have_right_quat = false;
    std::string line;
    int line_no = 0;

    auto read3 = [&](std::istringstream& v, std::array<double, 3>& out, const char* key) {
      v >> out[0] >> out[1] >> out[2];
      if (v.fail()) {
        error = "line " + std::to_string(line_no) + ": " + key + " needs 3 numbers (x y z)";
        return false;
      }
      return true;
    };
    auto read_quat = [&](std::istringstream& v, std::array<double, 4>& out, const char* key) {
      v >> out[0] >> out[1] >> out[2] >> out[3];
      if (v.fail()) {
        error = "line " + std::to_string(line_no) + ": " + key + " needs 4 numbers (w x y z)";
        return false;
      }
      const double n = std::sqrt(out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]);
      if (!(n > 1e-6)) {
        error = "line " + std::to_string(line_no) + ": " + key + " has zero norm";
        return false;
      }
      for (double& c : out) { c /= n; }
      return true;
    };

    while (std::getline(in, line)) {
      ++line_no;
      const auto hash = line.find('#');
      if (hash != std::string::npos) { line.erase(hash); }
      const auto colon = line.find(':');
      if (colon == std::string::npos) {
        if (line.find_first_not_of(" \t\r\n") != std::string::npos) {
          error = "line " + std::to_string(line_no) + ": expected 'key: value'";
          return false;
        }
        continue;
      }
      const std::string key = Trim(line.substr(0, colon));
      std::istringstream value(line.substr(colon + 1));

      if (key == "object_frame") {
        std::string frame;
        value >> frame;
        if (frame != "pelvis") {
          error = "line " + std::to_string(line_no) + ": object_frame must be 'pelvis' (got '" +
                  frame + "'); the shared-autonomy layer only computes poses in the pelvis frame";
          return false;
        }
      } else if (key == "object_position") {
        if (!read3(value, config.object.position, "object_position")) { return false; }
        have_position = true;
      } else if (key == "object_quaternion") {
        if (!read_quat(value, config.object.orientation, "object_quaternion")) { return false; }
        have_quaternion = true;
      } else if (key == "object_size") {
        if (!read3(value, config.size, "object_size")) { return false; }
        if (config.size[0] < 0.0 || config.size[1] < 0.0 || config.size[2] < 0.0) {
          error = "line " + std::to_string(line_no) + ": object_size must be non-negative";
          return false;
        }
        have_size = true;
      } else if (key == "left_grasp_position") {
        if (!read3(value, config.left_grasp_relative.position, "left_grasp_position")) { return false; }
        have_left_pos = true;
      } else if (key == "left_grasp_quaternion") {
        if (!read_quat(value, config.left_grasp_relative.orientation, "left_grasp_quaternion")) { return false; }
        have_left_quat = true;
      } else if (key == "right_grasp_position") {
        if (!read3(value, config.right_grasp_relative.position, "right_grasp_position")) { return false; }
        have_right_pos = true;
      } else if (key == "assist_enabled") {
        std::string flag;
        value >> flag;
        if (flag == "true" || flag == "1") {
          config.assist.enabled = true;
        } else if (flag == "false" || flag == "0") {
          config.assist.enabled = false;
        } else {
          error = "line " + std::to_string(line_no) + ": assist_enabled must be true or false";
          return false;
        }
      } else if (key == "task_enabled") {
        std::string flag;
        value >> flag;
        if (flag == "true" || flag == "1") {
          config.task.enabled = true;
        } else if (flag == "false" || flag == "0") {
          config.task.enabled = false;
        } else {
          error = "line " + std::to_string(line_no) + ": task_enabled must be true or false";
          return false;
        }
      } else if (key == "task_align_enter_radius" || key == "task_align_exit_radius" ||
                 key == "task_grasp_radius" || key == "task_align_timeout" ||
                 key == "task_grasp_settle" || key == "task_grasp_timeout" ||
                 key == "task_grasp_contact_margin" || key == "task_lift_height" ||
                 key == "task_lift_duration") {
        double v = 0.0;
        value >> v;
        if (value.fail() || v < 0.0) {
          error = "line " + std::to_string(line_no) + ": " + key + " needs one non-negative number";
          return false;
        }
        if (key == "task_align_enter_radius") { config.task.align_enter_radius = v; }
        else if (key == "task_align_exit_radius") { config.task.align_exit_radius = v; }
        else if (key == "task_grasp_radius") { config.task.grasp_radius = v; }
        else if (key == "task_align_timeout") { config.task.align_timeout_s = v; }
        else if (key == "task_grasp_settle") { config.task.grasp_settle_s = v; }
        else if (key == "task_grasp_timeout") { config.task.grasp_timeout_s = v; }
        else if (key == "task_grasp_contact_margin") { config.task.grasp_contact_margin = v; }
        else if (key == "task_lift_height") { config.task.lift_height = v; }
        else { config.task.lift_duration_s = v; }
      } else if (key == "assist_alpha" || key == "assist_engage_radius" ||
                 key == "assist_full_radius" || key == "assist_max_joint_correction" ||
                 key == "assist_alpha_slew" || key == "assist_orientation_weight") {
        double v = 0.0;
        value >> v;
        if (value.fail() || v < 0.0) {
          error = "line " + std::to_string(line_no) + ": " + key + " needs one non-negative number";
          return false;
        }
        if (key == "assist_alpha") { config.assist.alpha = v; }
        else if (key == "assist_engage_radius") { config.assist.engage_radius = v; }
        else if (key == "assist_full_radius") { config.assist.full_radius = v; }
        else if (key == "assist_max_joint_correction") { config.assist.max_joint_correction = v; }
        else if (key == "assist_orientation_weight") { config.assist.orientation_weight = v; }
        else { config.assist.alpha_slew = v; }
      } else if (key == "right_grasp_quaternion") {
        if (!read_quat(value, config.right_grasp_relative.orientation, "right_grasp_quaternion")) { return false; }
        have_right_quat = true;
      } else if (!key.empty()) {
        error = "line " + std::to_string(line_no) + ": unknown key '" + key + "'";
        return false;
      }
    }

    if (!have_position || !have_quaternion) {
      error = "'" + path + "' must define both object_position and object_quaternion";
      return false;
    }

    // Grasp targets: explicit positions win; otherwise derive the box side
    // centres from object_size.  Orientations default to the object's own
    // frame (identity relative rotation) unless given.
    if (have_left_pos != have_right_pos) {
      error = "'" + path + "': define both left_grasp_position and right_grasp_position, or neither";
      return false;
    }
    if (have_left_pos) {
      config.has_grasp_targets = true;
    } else if (have_size) {
      const double half_y = 0.5 * config.size[1];
      config.left_grasp_relative.position = {0.0, half_y, 0.0};    // +Y face centre
      config.right_grasp_relative.position = {0.0, -half_y, 0.0};  // -Y face centre
      config.has_grasp_targets = true;
    }
    if (!config.has_grasp_targets && (have_left_quat || have_right_quat)) {
      error = "'" + path + "': grasp quaternions given without object_size or grasp positions";
      return false;
    }
    if (config.assist.full_radius > config.assist.engage_radius) {
      error = "'" + path + "': assist_full_radius must not exceed assist_engage_radius";
      return false;
    }
    if (config.task.enabled && !config.assist.enabled) {
      error = "'" + path + "': task_enabled needs assist_enabled (the sequence drives the assist)";
      return false;
    }
    if (config.task.grasp_radius > config.task.align_enter_radius) {
      error = "'" + path + "': task_grasp_radius must not exceed task_align_enter_radius";
      return false;
    }
    if (config.task.align_exit_radius < config.task.align_enter_radius) {
      error = "'" + path + "': task_align_exit_radius must be >= task_align_enter_radius";
      return false;
    }
    if (config.task.grasp_timeout_s < config.task.grasp_settle_s) {
      error = "'" + path + "': task_grasp_timeout must be >= task_grasp_settle";
      return false;
    }
    if (config.assist.enabled && !config.has_grasp_targets) {
      error = "'" + path + "': assist_enabled needs grasp targets (object_size or explicit positions)";
      return false;
    }
    return true;
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
    r.geometry_valid = geometry_.valid ? 1u : 0u;
    r.object_valid = geometry_.object_valid ? 1u : 0u;
    r.left_wrist = geometry_.left_wrist;
    r.right_wrist = geometry_.right_wrist;
    r.object = geometry_.object;
    r.left_wrist_to_object = geometry_.left_wrist_to_object;
    r.right_wrist_to_object = geometry_.right_wrist_to_object;
    r.grasp_targets_valid = geometry_.grasp_targets_valid ? 1u : 0u;
    r.left_grasp = geometry_.left_grasp;
    r.right_grasp = geometry_.right_grasp;
    r.assist_enabled = assist_active() ? 1u : 0u;
    r.left_assist = left_assist_;
    r.right_assist = right_assist_;
    for (std::size_t i = 0; i < kNumBodyJoints; ++i) {
      r.correction[i] = final_action[i] - sonic_action[i];
    }
    r.task_state = static_cast<std::int32_t>(task_status_.state);
    r.grasp_trigger = task_status_.grasp_trigger ? 1u : 0u;
    r.lift_trigger = task_status_.lift_trigger ? 1u : 0u;
    r.grasp_success = task_status_.grasp_success ? 1u : 0u;
    r.task_success = task_status_.task_success ? 1u : 0u;
    r.time_in_state_s = task_status_.time_in_state_s;
    r.lift_offset_m = task_status_.lift_offset_m;
    r.left_grasp_residual = task_status_.left_grasp_residual;
    r.right_grasp_residual = task_status_.right_grasp_residual;
    // What the hands were actually told this tick (the sequence's command when
    // it has taken over, otherwise the operator's own command).
    r.left_hand_close_cmd = left_hand_final;
    r.right_hand_close_cmd = right_hand_final;

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
    // Geometry snapshot (pelvis frame)
    std::uint8_t geometry_valid = 0;
    std::uint8_t object_valid = 0;
    Pose left_wrist{};
    Pose right_wrist{};
    Pose object{};
    double left_wrist_to_object = 0.0;
    double right_wrist_to_object = 0.0;
    std::uint8_t grasp_targets_valid = 0;
    GraspTarget left_grasp{};
    GraspTarget right_grasp{};
    std::uint8_t assist_enabled = 0;
    AssistState left_assist{};
    AssistState right_assist{};
    BodyAction correction{};   ///< final_action - sonic_action, IsaacLab order.
    std::int32_t task_state = 0;
    std::uint8_t grasp_trigger = 0;
    std::uint8_t lift_trigger = 0;
    std::uint8_t grasp_success = 0;
    std::uint8_t task_success = 0;
    double time_in_state_s = 0.0;
    double lift_offset_m = 0.0;
    double left_grasp_residual = 0.0;
    double right_grasp_residual = 0.0;
    HandAction left_hand_close_cmd{};
    HandAction right_hand_close_cmd{};
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

  /**
   * @brief Compose one grasp target into the pelvis frame and score the wrist error.
   *
   * Composition is the usual rigid-body product `T_target = T_object * T_relative`:
   *   p_target = p_object + R_object * p_relative
   *   q_target = q_object (x) q_relative
   *
   * Orientation error is the geodesic angle between the wrist and target
   * rotations, `2*acos(|<q_wrist, q_target>|)`, which is sign-agnostic (q and
   * -q are the same rotation) and lands in [0, pi].
   */
  /**
   * @brief Nudge one arm's joints toward its grasp target.
   *
   * ## Gate
   * `alpha` is 0 beyond `engage_radius`, ramps linearly to `alpha` at
   * `full_radius`, and is slew-limited so engaging or dropping out cannot step
   * the correction in a single tick.  SONIC does the approach unaided; this
   * only touches the last few centimetres.
   *
   * ## Direction
   * Steepest descent on 1/2|e|^2, i.e. the Jacobian **transpose** step
   * `dq_j = J_j . e`.  No matrix inverse, no singularity handling, no iteration
   * -- this is one gradient step per control tick, not an IK solve.  Its
   * direction is right; its magnitude is not metric, so the magnitude is set
   * entirely by the normalisation below rather than by the gradient's scale.
   *
   * ## Magnitude
   * The step is rescaled so the largest single-joint correction equals
   * `alpha * max_joint_correction` radians.  Every joint is therefore bounded
   * by a compile-time cap regardless of error size or configuration.
   *
   * ## Units
   * The policy action is not radians:
   *   `q_target[hw] = default_angles[hw] + action[il] * g1_action_scale[hw]`
   * so a correction of `dq` radians at hardware joint `hw` is applied as
   * `action[il] += dq / g1_action_scale[hw]`.
   *
   * Only arm joints appear in the Jacobian, so legs, waist and torso are
   * untouchable here by construction.
   */
  void ApplyArmAssist(BodyAction& action, bool is_left) {
    const GraspTarget& grasp = is_left ? geometry_.left_grasp : geometry_.right_grasp;
    const ArmJacobian& jac = is_left ? geometry_.left_jacobian : geometry_.right_jacobian;
    AssistState& st = is_left ? left_assist_ : right_assist_;

    const double previous_alpha = st.alpha;
    st = AssistState{};
    st.distance = grasp.position_error_norm;

    if (!jac.valid || jac.count <= 0) {
      st.alpha = SlewToward(previous_alpha, 0.0);
      return;
    }

    // Gate: 0 outside engage_radius, full inside full_radius, linear between.
    double gate = 0.0;
    const double span = assist_.engage_radius - assist_.full_radius;
    if (st.distance <= assist_.full_radius) {
      gate = 1.0;
    } else if (st.distance < assist_.engage_radius && span > 1e-9) {
      gate = (assist_.engage_radius - st.distance) / span;
    }
    st.alpha = SlewToward(previous_alpha, Clamp(gate, 0.0, 1.0) * assist_.alpha);
    if (st.alpha <= 0.0) { return; }

    // Jacobian-transpose direction, position and orientation blended.
    //
    // The two gradients have different units (m^2/rad against rad), so they are
    // each normalised to unit maximum first; orientation_weight is then a plain
    // split of the same per-joint budget rather than a quantity that would need
    // retuning whenever the scene scale changes.  Weight 0 reproduces the
    // position-only step bit for bit.
    std::array<double, kMaxArmJoints> step_pos{};
    std::array<double, kMaxArmJoints> step_rot{};
    double max_pos = 0.0;
    double max_rot = 0.0;
    for (int j = 0; j < jac.count; ++j) {
      const std::size_t sj = static_cast<std::size_t>(j);
      const auto& c = jac.column[sj];
      step_pos[sj] = c[0] * grasp.position_error[0] + c[1] * grasp.position_error[1] +
                     c[2] * grasp.position_error[2];
      max_pos = std::max(max_pos, std::abs(step_pos[sj]));
      const auto& a = jac.angular[sj];
      step_rot[sj] = a[0] * grasp.orientation_error[0] + a[1] * grasp.orientation_error[1] +
                     a[2] * grasp.orientation_error[2];
      max_rot = std::max(max_rot, std::abs(step_rot[sj]));
    }

    const double w_rot = Clamp(assist_.orientation_weight, 0.0, 1.0);
    const double w_pos = 1.0 - w_rot;
    std::array<double, kMaxArmJoints> step{};
    double max_abs = 0.0;
    for (int j = 0; j < jac.count; ++j) {
      const std::size_t sj = static_cast<std::size_t>(j);
      double v = 0.0;
      if (w_pos > 0.0 && max_pos > 1e-12) { v += w_pos * step_pos[sj] / max_pos; }
      if (w_rot > 0.0 && max_rot > 1e-12) { v += w_rot * step_rot[sj] / max_rot; }
      step[sj] = v;
      max_abs = std::max(max_abs, std::abs(v));
    }
    if (!(max_abs > 1e-12)) { return; }

    // Rescale so the largest joint step is exactly alpha * max_joint_correction.
    const double scale = st.alpha * assist_.max_joint_correction / max_abs;
    st.count = jac.count;
    for (int j = 0; j < jac.count; ++j) {
      const std::size_t sj = static_cast<std::size_t>(j);
      const int hw = jac.joint_hw_index[sj];
      if (hw < 0 || hw >= static_cast<int>(kNumBodyJoints)) { continue; }
      const double dq = step[sj] * scale;
      const double scale_hw = g1_action_scale[static_cast<std::size_t>(hw)];
      if (!(std::abs(scale_hw) > 1e-12)) { continue; }
      action[static_cast<std::size_t>(isaaclab_to_mujoco[static_cast<std::size_t>(hw)])] +=
          static_cast<float>(dq / scale_hw);
      st.joint_hw_index[sj] = hw;
      st.correction_rad[sj] = dq;
      st.max_abs_correction_rad = std::max(st.max_abs_correction_rad, std::abs(dq));
    }
  }

  void EnterState(TaskState next, const std::string& reason) {
    if (next == task_status_.state) { return; }
    std::cout << "[SharedAutonomy] task " << TaskStateName(task_status_.state) << " -> "
              << TaskStateName(next) << " (" << reason << ")" << std::endl;
    task_status_.state = next;
    task_status_.time_in_state_s = 0.0;
    task_status_.last_reason = reason;
    if (next == TaskState::MANUAL || next == TaskState::ABORT) {
      task_lift_offset_ = 0.0;
      if (next == TaskState::MANUAL) { task_status_.grasp_success = false; }
    }
  }

  /// How far each hand fell short of the commanded closure, in radians.
  void UpdateGraspResiduals() {
    if (!hand_state_valid_) {
      task_status_.left_grasp_residual = 0.0;
      task_status_.right_grasp_residual = 0.0;
      return;
    }
    task_status_.left_grasp_residual = L2Diff(left_hand_measured_, ClosedHandPose(true));
    task_status_.right_grasp_residual = L2Diff(right_hand_measured_, ClosedHandPose(false));
  }

  /**
   * @brief Is something actually held?
   *
   * The only grasp evidence this stack has is joint feedback: there is no force
   * or tactile sensing on the Dex3 here. Fingers commanded closed that stall
   * well short of the command are being blocked by something, which is taken as
   * a grasp. Fingers that reach the commanded pose closed on air.
   *
   * That makes a "successful grasp" mean *an obstruction of roughly object
   * size*, not a verified stable hold. Good enough for a toy sequence; it is
   * not a substitute for real contact sensing.
   */
  bool GraspHolding() const {
    if (!hand_state_valid_) { return false; }
    return task_status_.left_grasp_residual > task_cfg_.grasp_contact_margin &&
           task_status_.right_grasp_residual > task_cfg_.grasp_contact_margin;
  }

  /// True while the sequence wants the arm assist to pull toward the target.
  bool TaskWantsArmAssist() const {
    if (!task_cfg_.enabled) { return true; }  // assist runs standalone when the task is off
    return task_status_.state == TaskState::ALIGN || task_status_.state == TaskState::GRASP ||
           task_status_.state == TaskState::LIFT;
  }

  /// Ramp an assist channel down instead of dropping it in one tick.
  void DecayAssist(AssistState& st) {
    const double previous_alpha = st.alpha;
    st = AssistState{};
    st.alpha = SlewToward(previous_alpha, 0.0);
  }

  double SlewToward(double current, double target) const {
    const double delta = Clamp(target - current, -assist_.alpha_slew, assist_.alpha_slew);
    return Clamp(current + delta, 0.0, kMaxAlpha);
  }

  static double Clamp(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
  }

  void UpdateGraspTarget(GraspTarget& target, const Pose& wrist) const {
    const auto rotated = quat_rotate_d(geometry_.object.orientation, target.relative.position);
    target.pose.position = {geometry_.object.position[0] + rotated[0],
                            geometry_.object.position[1] + rotated[1],
                            geometry_.object.position[2] + rotated[2]};
    // LIFT raises the target along pelvis +Z. That is true vertical only while
    // the pelvis is upright; the toy sequence assumes a standing robot.
    target.pose.position[2] += task_lift_offset_;
    target.pose.orientation =
        quat_unit_d(quat_mul_d(geometry_.object.orientation, target.relative.orientation));

    target.position_error = {target.pose.position[0] - wrist.position[0],
                             target.pose.position[1] - wrist.position[1],
                             target.pose.position[2] - wrist.position[2]};
    target.position_error_norm = std::sqrt(target.position_error[0] * target.position_error[0] +
                                           target.position_error[1] * target.position_error[1] +
                                           target.position_error[2] * target.position_error[2]);
    target.orientation_error_rad = OrientationError(wrist.orientation, target.pose.orientation);
    target.orientation_error = OrientationErrorVector(wrist.orientation, target.pose.orientation);
  }

  /**
   * @brief Rotation taking the wrist onto the target, as an axis-angle vector.
   *
   * q_err = q_target * conj(q_wrist), sign-normalised so the rotation is the
   * short way round, then converted to axis * angle in the pelvis frame.  That
   * vector is exactly what the angular Jacobian transpose consumes.
   */
  static std::array<double, 3> OrientationErrorVector(const std::array<double, 4>& wrist,
                                                      const std::array<double, 4>& target) {
    std::array<double, 4> q = quat_mul_d(target, quat_conjugate_d(wrist));
    if (q[0] < 0.0) { for (double& c : q) { c = -c; } }   // shortest arc
    const double vnorm = std::sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(vnorm > 1e-12)) { return {0.0, 0.0, 0.0}; }
    const double w = Clamp(q[0], -1.0, 1.0);
    const double angle = 2.0 * std::atan2(vnorm, w);
    const double k = angle / vnorm;
    return {q[1] * k, q[2] * k, q[3] * k};
  }

  /// Geodesic angle (rad) between two unit quaternions, in [0, pi].
  static double OrientationError(const std::array<double, 4>& a, const std::array<double, 4>& b) {
    double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    if (dot < 0.0) { dot = -dot; }          // q and -q denote the same rotation
    if (dot > 1.0) { dot = 1.0; }           // guard acos against rounding
    return 2.0 * std::acos(dot);
  }

  static double Distance(const std::array<double, 3>& a, const std::array<double, 3>& b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  static std::string Trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) { return std::string(); }
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
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
    file_ << ",geometry_valid,object_valid"
          << ",left_wrist_to_object,right_wrist_to_object";
    static const char* kXyz[3] = {"x", "y", "z"};
    static const char* kWxyz[4] = {"qw", "qx", "qy", "qz"};
    for (const char* b : {"left_wrist", "right_wrist", "object"}) {
      for (const char* c : kXyz) { file_ << ',' << b << '_' << c; }
      for (const char* c : kWxyz) { file_ << ',' << b << '_' << c; }
    }
    file_ << ",grasp_targets_valid";
    for (const char* b : {"left_grasp_target", "right_grasp_target"}) {
      for (const char* c : kXyz) { file_ << ',' << b << '_' << c; }
      for (const char* c : kWxyz) { file_ << ',' << b << '_' << c; }
    }
    for (const char* b : {"left_grasp", "right_grasp"}) {
      for (const char* c : kXyz) { file_ << ',' << b << "_pos_err_" << c; }
      file_ << ',' << b << "_pos_err_norm" << ',' << b << "_ori_err_rad";
      for (const char* c : kXyz) { file_ << ',' << b << "_ori_err_" << c; }
    }
    file_ << ",assist_enabled";
    for (const char* b : {"left", "right"}) {
      file_ << ',' << b << "_alpha" << ',' << b << "_assist_max_corr_rad" << ',' << b
             << "_assist_joint_count";
      for (std::size_t i = 0; i < kMaxArmJoints; ++i) { file_ << ',' << b << "_corr_rad_" << i; }
      for (std::size_t i = 0; i < kMaxArmJoints; ++i) { file_ << ',' << b << "_corr_hw_" << i; }
    }
    for (std::size_t i = 0; i < kNumBodyJoints; ++i) { file_ << ",correction_" << i; }
    file_ << ",correction_norm";
    file_ << ",current_state,time_in_state_s,grasp_trigger,lift_trigger"
          << ",grasp_success,task_success,lift_offset_m"
          << ",left_grasp_residual,right_grasp_residual";
    for (const char* b : {"left", "right"}) {
      for (std::size_t i = 0; i < kNumHandJoints; ++i) {
        file_ << ',' << b << "_hand_close_command_" << i;
      }
    }
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
    file_ << ',' << static_cast<int>(r.geometry_valid) << ','
          << static_cast<int>(r.object_valid);
    file_ << ',' << r.left_wrist_to_object << ',' << r.right_wrist_to_object;
    for (const Pose* p : {&r.left_wrist, &r.right_wrist, &r.object}) {
      for (double v : p->position) { file_ << ',' << v; }
      for (double v : p->orientation) { file_ << ',' << v; }
    }
    file_ << ',' << static_cast<int>(r.grasp_targets_valid);
    for (const GraspTarget* g : {&r.left_grasp, &r.right_grasp}) {
      for (double v : g->pose.position) { file_ << ',' << v; }
      for (double v : g->pose.orientation) { file_ << ',' << v; }
    }
    for (const GraspTarget* g : {&r.left_grasp, &r.right_grasp}) {
      for (double v : g->position_error) { file_ << ',' << v; }
      file_ << ',' << g->position_error_norm << ',' << g->orientation_error_rad;
      for (double v : g->orientation_error) { file_ << ',' << v; }
    }
    file_ << ',' << static_cast<int>(r.assist_enabled);
    for (const AssistState* a : {&r.left_assist, &r.right_assist}) {
      file_ << ',' << a->alpha << ',' << a->max_abs_correction_rad << ',' << a->count;
      for (double v : a->correction_rad) { file_ << ',' << v; }
      for (int v : a->joint_hw_index) { file_ << ',' << v; }
    }
    double corr_norm = 0.0;
    for (float v : r.correction) {
      file_ << ',' << v;
      corr_norm += static_cast<double>(v) * static_cast<double>(v);
    }
    file_ << ',' << std::sqrt(corr_norm);
    file_ << ',' << r.task_state << ',' << r.time_in_state_s << ','
          << static_cast<int>(r.grasp_trigger) << ',' << static_cast<int>(r.lift_trigger) << ','
          << static_cast<int>(r.grasp_success) << ',' << static_cast<int>(r.task_success) << ','
          << r.lift_offset_m << ',' << r.left_grasp_residual << ',' << r.right_grasp_residual;
    for (double v : r.left_hand_close_cmd) { file_ << ',' << v; }
    for (double v : r.right_hand_close_cmd) { file_ << ',' << v; }
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

  // Hard safety caps.  Compiled in, deliberately not configurable.
  static constexpr double kMaxAlpha = 1.0;
  static constexpr double kMaxJointCorrectionRad = 0.15;

  std::atomic<bool> enabled_{false};
  Metrics metrics_{};
  GeometryState geometry_{};
  AssistConfig assist_{};
  std::atomic<bool> assist_runtime_enabled_{true};
  static constexpr double kMaxLiftHeight = 0.15;  ///< Hard cap, not configurable.
  TaskConfig task_cfg_{};
  TaskStatus task_status_{};
  double task_lift_offset_ = 0.0;
  bool abort_rearm_pending_ = false;
  bool retry_blocked_ = false;
  HandAction left_hand_measured_{};
  HandAction right_hand_measured_{};
  bool hand_state_valid_ = false;
  AssistState left_assist_{};
  AssistState right_assist_{};
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
