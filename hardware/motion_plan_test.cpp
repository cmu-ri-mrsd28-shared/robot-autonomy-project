#include "motion_plan.hpp"

#include <iostream>
#include <string>

namespace {

using namespace catchbox;

const Joints kStart{0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0};
const Point kCenter{0.40, 0.0, 0.35};
const Eigen::Matrix3d kRotation = Eigen::Matrix3d::Identity();

Transform test_pose(const Joints& q) {
  Transform pose = Transform::Identity();
  pose.topRightCorner<3, 1>() =
      kCenter + Point(q[0], q[1] - kStart[1], q[2]);
  // Joint 7 can cancel the yaw introduced when moving joint 1.
  pose.topLeftCorner<3, 3>() =
      Eigen::AngleAxisd(q[0] + q[6], Point::UnitZ()).toRotationMatrix();
  return pose;
}

Jacobian test_jacobian(const Joints&) {
  Jacobian jacobian = Jacobian::Zero();
  jacobian.topLeftCorner<3, 3>().setIdentity();
  jacobian(5, 0) = 1.0;
  jacobian(5, 6) = 1.0;
  return jacobian;
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Operation>
void require_rejection(const Operation& operation, const char* expected) {
  try {
    operation();
  } catch (const std::runtime_error& error) {
    require(std::string(error.what()).find(expected) != std::string::npos, error.what());
    return;
  }
  throw std::runtime_error("The invalid plan was accepted");
}

void test_planner() {
  const Point target = kCenter + Point(0.031, -0.024, 0.012);
  const auto plan = plan_move(kStart, target, kCenter, kRotation, kFr3Limits,
                              test_pose, test_jacobian);
  require(plan.samples.front() == kStart, "First command must equal the start");
  require(plan.duration_s > 0.0 && plan.samples.size() > 1, "Move needs intermediate commands");
  const Transform final_pose = test_pose(plan.samples.back());
  require(within_radius(final_pose.topRightCorner<3, 1>(), target, kIkPositionToleranceM),
          "Final position must reach the target");
  require(Eigen::AngleAxisd(final_pose.topLeftCorner<3, 3>()).angle() <= kIkAngleToleranceRad,
          "Final orientation must match the start");
  require(plan.max_radius_m <= kPlannedRadiusM, "Plan must stay inside the workspace");
  require(plan_move(kStart, kCenter, kCenter, kRotation, kFr3Limits, test_pose, test_jacobian)
              .samples.size() == 1,
          "An already reached target requires no motion");

  require_rejection(
      [&] {
        plan_move(kStart, kCenter + Point(0.19, 0.0, 0.0), kCenter, kRotation, kFr3Limits,
                  test_pose, test_jacobian);
      },
      "Target is outside");
  require_rejection(
      [&] {
        Joints outside = kStart;
        outside[5] = 0.5;
        plan_move(outside, target, kCenter, kRotation, kFr3Limits, test_pose, test_jacobian);
      },
      "Starting joint position");
  require_rejection(
      [&] {
        plan_move(kStart, target, kCenter, kRotation, kFr3Limits, test_pose,
                  [](const Joints&) -> Jacobian { return Jacobian::Zero(); });
      },
      "IK could not reach");

  // Keep the endpoints inside the sphere while forcing the path through a bulge.
  const auto bulging_pose = [](const Joints& q) {
    Transform pose = test_pose(q);
    pose(1, 3) += 0.22 * std::max(0.0, 1.0 - std::abs(q[0] - 0.025) / 0.01);
    return pose;
  };
  require_rejection(
      [&] {
        plan_move(kStart, kCenter + Point(0.05, 0.0, 0.0), kCenter, kRotation, kFr3Limits,
                  bulging_pose, test_jacobian);
      },
      "Planned tool path leaves");
}

}  // namespace

int main() {
  try {
    test_planner();
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
