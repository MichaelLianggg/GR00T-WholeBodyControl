/**
 * @file dex1_grippers.hpp
 * @brief DDS driver for a pair of Unitree Dex1-1 parallel grippers.
 */
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <memory>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

class Dex1Grippers {
 public:
  void initialize(const std::string& network_interface) {
    if (!network_interface.empty()) {
      unitree::robot::ChannelFactory::Instance()->Init(0, network_interface.c_str());
    }
    initializeSide(left_, true);
    initializeSide(right_, false);
  }

  void SetMaxCloseRatio(double ratio) {
    max_close_ratio_ = std::clamp(ratio, 0.2, 1.0);
  }
  double GetMaxCloseRatio() const { return max_close_ratio_; }

  // Dex3 targets are retained at the input boundary for protocol compatibility.
  // Finger-flexion joints 3..6 have a nominal closed magnitude of 1.5 rad.
  void setAllJointsCommand(bool is_left, const std::array<double, 7>& q) {
    double closure = 0.0;
    for (int i = 3; i < 7; ++i) closure = std::max(closure, std::abs(q[i]) / 1.5);
    setClosure(is_left, std::clamp(closure, 0.0, 1.0));
  }

  void setClosure(bool is_left, double closure) {
    Side& side = is_left ? left_ : right_;
    auto cmd = side.command.GetDataWithTime().data;
    unitree_go::msg::dds_::MotorCmds_ next = cmd ? *cmd : makeCommand();
    // Calibration defines q=0 at fully closed; increasing q opens Dex1.
    next.cmds()[0].q(static_cast<float>(dex1_max_q_ *
        (1.0 - max_close_ratio_ * std::clamp(closure, 0.0, 1.0))));
    side.command.SetData(std::move(next));
  }

  void open(bool is_left) { setClosure(is_left, 0.0); }
  void close(bool is_left) { setClosure(is_left, 1.0); }

  void writeOnce() {
    writeSide(left_);
    writeSide(right_);
  }

  std::shared_ptr<const unitree_go::msg::dds_::MotorStates_> getState(bool is_left) const {
    return (is_left ? left_ : right_).state.GetDataWithTime().data;
  }

 private:
  struct Side {
    unitree::robot::ChannelPublisherPtr<unitree_go::msg::dds_::MotorCmds_> publisher;
    unitree::robot::ChannelSubscriberPtr<unitree_go::msg::dds_::MotorStates_> subscriber;
    DataBuffer<unitree_go::msg::dds_::MotorCmds_> command;
    DataBuffer<unitree_go::msg::dds_::MotorStates_> state;
  };

  static unitree_go::msg::dds_::MotorCmds_ makeCommand() {
    unitree_go::msg::dds_::MotorCmds_ cmd;
    cmd.cmds().resize(1);
    cmd.cmds()[0].mode(1);
    cmd.cmds()[0].q(dex1_max_q_);  // Start open; calibrated fully-closed position is q=0.
    cmd.cmds()[0].dq(0.0f);
    cmd.cmds()[0].tau(0.0f);
    cmd.cmds()[0].kp(5.0f);
    cmd.cmds()[0].kd(0.05f);
    return cmd;
  }

  void initializeSide(Side& side, bool is_left) {
    const std::string base = is_left ? "rt/dex1/left" : "rt/dex1/right";
    side.command.SetData(makeCommand());
    side.publisher = std::make_shared<unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>>(base + "/cmd");
    side.subscriber = std::make_shared<unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>>(base + "/state");
    side.publisher->InitChannel();
    side.subscriber->InitChannel([this, is_left](const void* message) {
      auto& target = is_left ? left_ : right_;
      target.state.SetData(*static_cast<const unitree_go::msg::dds_::MotorStates_*>(message));
    }, 1);
  }

  static void writeSide(Side& side) {
    auto cmd = side.command.GetDataWithTime().data;
    if (side.publisher && cmd) side.publisher->Write(*cmd);
  }

  Side left_;
  Side right_;
  double max_close_ratio_ = 1.0;
  // Official test program sweeps approximately 0.5..5.5 rad; q=0 is the
  // calibrated fully-closed position and 5.1 rad is the open endpoint.
  double dex1_max_q_ = 5.1;
};
