#ifndef DEX1_GRIPPERS_HPP
#define DEX1_GRIPPERS_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory>
#include <string>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

/** Controller for a pair of Unitree DEX1-1 parallel grippers.
 *
 * The serial-to-DDS service supplied by Unitree must already be running.  It
 * exposes one M4010 motor per gripper on rt/dex1/{left,right}/{cmd,state}.
 * SONIC supplies seven DEX3 joints, so the four finger-flexion joints are
 * reduced to a single normalized closure value.
 */
class Dex1Grippers
{
public:
    static constexpr std::size_t SONIC_HAND_DOF = 7;

    void initialize()
    {
        initializeSide(left_, "rt/dex1/left", true);
        initializeSide(right_, "rt/dex1/right", false);
    }

    void SetMaxCloseRatio(double ratio)
    {
        max_close_ratio_.store(std::clamp(ratio, 0.2, 1.0));
    }

    double GetMaxCloseRatio() const { return max_close_ratio_.load(); }

    void setAllJointsCommand(bool is_left,
                             const std::array<double, SONIC_HAND_DOF> &q)
    {
        // DEX3 joints 3..6 are the two flexion axes of the index and middle
        // fingers. Their full travel is approximately 1.57/1.75 rad.
        constexpr std::array<double, 4> ranges = {1.57, 1.75, 1.57, 1.75};
        double closure = 0.0;
        for (std::size_t i = 0; i < ranges.size(); ++i)
        {
            closure += std::clamp(std::abs(q[i + 3]) / ranges[i], 0.0, 1.0);
        }
        closure /= static_cast<double>(ranges.size());
        (is_left ? left_ : right_).closure.store(closure);
    }

    void open(bool is_left) { (is_left ? left_ : right_).closure.store(0.0); }
    void close(bool is_left) { (is_left ? left_ : right_).closure.store(1.0); }

    void writeOnce()
    {
        publishSide(left_);
        publishSide(right_);
    }

    bool getState(bool is_left, double &q, double &dq) const
    {
        const auto state = (is_left ? left_ : right_).state.GetDataWithTime().data;
        if (!state || state->states().empty()) { return false; }
        q = state->states()[0].q();
        dq = state->states()[0].dq();
        return true;
    }

private:
    using Command = unitree_go::msg::dds_::MotorCmds_;
    using State = unitree_go::msg::dds_::MotorStates_;

    struct Side
    {
        unitree::robot::ChannelPublisherPtr<Command> publisher;
        unitree::robot::ChannelSubscriberPtr<State> subscriber;
        DataBuffer<State> state;
        std::atomic<double> closure {0.0};
    };

    void initializeSide(Side &side, const std::string &topic, bool is_left)
    {
        side.publisher.reset(new unitree::robot::ChannelPublisher<Command>(topic + "/cmd"));
        side.subscriber.reset(new unitree::robot::ChannelSubscriber<State>(topic + "/state"));
        side.publisher->InitChannel();
        side.subscriber->InitChannel(
            [this, is_left](const void *message) { onState(is_left, message); }, 1);
    }

    void onState(bool is_left, const void *message)
    {
        (is_left ? left_ : right_).state.SetData(*static_cast<const State *>(message));
    }

    void publishSide(Side &side)
    {
        if (!side.publisher) { return; }

        // Do not send an absolute position before feedback is available.  The
        // gripper may be anywhere in its travel when sonic_deploy starts.
        const auto state = side.state.GetDataWithTime().data;
        if (!state || state->states().empty()) { return; }

        const double desired_q = DEX1_OPEN_Q +
            (DEX1_CLOSED_Q - DEX1_OPEN_Q) * side.closure.load() * max_close_ratio_.load();
        const double current_q = state->states()[0].q();
        const double safe_q = std::clamp(desired_q,
                                         current_q - DEX1_MAX_DELTA_Q,
                                         current_q + DEX1_MAX_DELTA_Q);

        Command command;
        command.cmds().resize(1);
        auto &motor = command.cmds()[0];
        motor.mode(1);
        motor.q(static_cast<float>(safe_q));
        motor.dq(0.0f);
        motor.kp(DEX1_KP);
        motor.kd(DEX1_KD);
        motor.tau(0.0f);
        side.publisher->Write(command);
    }

    // Unitree calibration defines q=0 as closed; 5.4 rad opens the 9 cm stroke.
    static constexpr double DEX1_CLOSED_Q = 0.0;
    static constexpr double DEX1_OPEN_Q = 5.1;
    static constexpr double DEX1_MAX_DELTA_Q = 0.18;
    static constexpr float DEX1_KP = 5.0f;
    static constexpr float DEX1_KD = 0.05f;

    Side left_;
    Side right_;
    std::atomic<double> max_close_ratio_ {1.0};
};

#endif // DEX1_GRIPPERS_HPP
