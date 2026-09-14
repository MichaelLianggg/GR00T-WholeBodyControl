/**
 * @file fk.hpp
 * @brief Forward kinematics (FK) for the G1 robot, parsed from a MuJoCo XML model.
 *
 * RobotFK loads a MuJoCo MJCF XML file (e.g. `g1/g1_29dof.xml`), extracts the
 * kinematic tree (joint axes, translations, rest rotations), and provides a
 * `DoFK()` method that computes world-frame positions and orientations for
 * every body given a root pose and joint angles.
 *
 * The FK result is used by MotionSequence::ComputeFK() to populate body-part
 * positions and quaternions from joint-angle data.
 */

#pragma once

#include <string>
#include <vector>
#include <array>

struct XMLNode;

/**
 * @class RobotFK
 * @brief Forward kinematics solver for the G1 robot, initialised from a MuJoCo XML.
 */
class RobotFK
{
    public:

        RobotFK(const std::string &xmlfile);

        void DoFK(
            std::array<double, 3> *positions_world,
            std::array<double, 4> *rotations_world,
            const std::array<double, 3> &root_translation,
            const std::array<double, 4> &root_rotation,
            const double *joint_angles
        ) const;

        int NumJoints() const { return node_children_.size(); }

        /**
         * @brief Look up a body by its MJCF `name` attribute.
         *
         * The returned index addresses the `positions_world` / `rotations_world`
         * arrays filled by DoFK(): index 0 is the root body (pelvis) and index
         * `k >= 1` is the body actuated by MuJoCo/hardware joint `k - 1`.
         *
         * @return Body index, or -1 if no body carries that name.
         */
        int FindBodyIndex(const std::string &name) const;

        /// MJCF name of body @p idx, or an empty string if out of range.
        const std::string &BodyName(int idx) const;

        /// Parent body index of @p idx; -1 for the root and for out-of-range input.
        int Parent(int idx) const;

        /**
         * @brief Joint axis of body @p idx as declared in the MJCF, in the body's
         *        own (post-rest-rotation) frame.
         *
         * The same axis in world coordinates is `quat_rotate(rotations_world[idx],
         * JointAxis(idx))`, using the rotations DoFK() produced.
         */
        const std::array<double, 3> &JointAxis(int idx) const;

    private:

        void FKChildren(
            std::array<double, 3> *positions_world,
            std::array<double, 4> *rotations_world,
            int parent_idx,
            const double *joint_angles
        ) const;

        void AddNode(XMLNode *node);

        std::vector< std::array<double, 3> > axes_;
        std::vector< std::array<double, 3> > translations_;
        std::vector< std::array<double, 4> > rest_rotations_;
        std::vector< std::vector<int> > node_children_;
        std::vector< int > node_parents_;
        std::vector< std::string > node_names_;
};
