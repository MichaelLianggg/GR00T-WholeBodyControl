#include <gtest/gtest.h>

#include "dex1_grippers.hpp"

// No initialize() calls: these tests exercise commands without opening DDS or
// sending anything to a robot.
TEST(Dex1Grippers, OpenCloseAndIndependentSides) {
  Dex1Grippers grippers;
  EXPECT_EQ(grippers.getCommand(true), nullptr);
  grippers.open(true);
  grippers.close(false);
  auto left = grippers.getCommand(true);
  auto right = grippers.getCommand(false);
  ASSERT_EQ(left->cmds().size(), 1);
  ASSERT_EQ(right->cmds().size(), 1);
  EXPECT_FLOAT_EQ(left->cmds()[0].q(), 5.1f);
  EXPECT_FLOAT_EQ(right->cmds()[0].q(), 0.0f);
  EXPECT_EQ(left->cmds()[0].mode(), 1);
  EXPECT_FLOAT_EQ(left->cmds()[0].kp(), 3.0f);
  EXPECT_FLOAT_EQ(left->cmds()[0].kd(), 0.05f);
  EXPECT_FLOAT_EQ(left->cmds()[0].dq(), 0.0f);
  EXPECT_FLOAT_EQ(left->cmds()[0].tau(), 0.0f);
}

TEST(Dex1Grippers, ClosureBoundsAndLimit) {
  Dex1Grippers grippers;
  grippers.setClosure(true, -1.0);
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 5.1f);
  grippers.setClosure(true, 2.0);
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 0.0f);
  grippers.SetMaxCloseRatio(0.5);
  grippers.close(true);
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 2.55f);
  grippers.open(true);
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 5.1f);
  grippers.SetMaxCloseRatio(-1.0);
  EXPECT_DOUBLE_EQ(grippers.GetMaxCloseRatio(), 0.2);
  grippers.SetMaxCloseRatio(2.0);
  EXPECT_DOUBLE_EQ(grippers.GetMaxCloseRatio(), 1.0);
}

TEST(Dex1Grippers, PicoTargetsAndMirroredRightHand) {
  Dex1Grippers grippers;
  // Half-trigger index gesture from G1GripperInverseKinematicsSolver.
  std::array<double, 7> left = {-0.25, 0.35, 0.35, -0.75, -0.75, -0.3, -0.75};
  std::array<double, 7> right = left;
  for (auto& q : right) q = -q;
  grippers.setAllJointsCommand(true, left);
  grippers.setAllJointsCommand(false, right);
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 2.55f);
  EXPECT_FLOAT_EQ(grippers.getCommand(false)->cmds()[0].q(), 2.55f);
  grippers.setAllJointsCommand(true, {});
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 5.1f);
  // Unrelated thumb joints must not command closure.
  grippers.setAllJointsCommand(true, {1.5, 1.5, 1.5, 0, 0, 0, 0});
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 5.1f);
  grippers.setAllJointsCommand(true, {0, 0, 0, -3, 0, 0, 0});
  EXPECT_FLOAT_EQ(grippers.getCommand(true)->cmds()[0].q(), 0.0f);
}
