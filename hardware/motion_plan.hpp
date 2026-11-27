#pragma once

#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <ruckig/ruckig.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace catchbox {

using Joints = std::array<double, 7>;
using Point = Eigen::Vector3d;
using Transform = Eigen::Matrix4d;
using Jacobian = Eigen::Matrix<double, 6, 7>;

constexpr double kWorkspaceRadiusM = 0.20;
constexpr double kPlannedRadiusM = kWorkspaceRadiusM - 0.02;
constexpr double kMeasuredRadiusM = kWorkspaceRadiusM - 0.01;
constexpr double kCycleS = 0.001;
constexpr double kIkPositionToleranceM = 0.0005;
constexpr double kIkAngleToleranceRad = 0.005;

struct JointLimits {
  Joints min;
  Joints max;
};

// Manufacturer's FER position limits with a small margin at each end.
inline constexpr JointLimits kPandaLimits{
    {-2.8773, -1.7428, -2.8773, -3.0518, -2.8773, 0.0025, -2.8773},
    {2.8773, 1.7428, 2.8773, -0.0898, 2.8773, 3.7325, 2.8773}};

// Manufacturer's FR3 rectangular bounds keep the velocity limit uniform.
inline constexpr JointLimits kFr3Limits{
    {-2.3476, -1.5454, -2.4937, -2.7714, -2.5100, 0.7773, -2.7045},
    {2.3476, 1.5454, 2.4937, -0.4226, 2.5100, 4.2841, 2.7045}};

struct MotionPlan {
  // One position command per 1 ms, including the initial and final commands.
  std::vector<Joints> samples;
  double duration_s;
  double max_radius_m;
};

inline bool within_radius(const Point& point, const Point& center, double radius) {
  return (point - center).squaredNorm() <= radius * radius;
}

template <typename Forward, typename Differential>
Joints solve_ik(Joints joints, const Point& target, const Eigen::Matrix3d& rotation,
                const JointLimits& limits, const Forward& forward,
                const Differential& differential) {
  for (int iteration = 0; iteration < 100; ++iteration) {
    const Transform pose = forward(joints);
    const Eigen::AngleAxisd angular_error(rotation * pose.topLeftCorner<3, 3>().transpose());
    Eigen::Matrix<double, 6, 1> error;
    error.head<3>() = target - pose.topRightCorner<3, 1>();
    error.tail<3>() = angular_error.axis() * angular_error.angle();
    if (error.head<3>().norm() <= kIkPositionToleranceM &&
        error.tail<3>().norm() <= kIkAngleToleranceRad) {
      return joints;
    }

    // The zero Jacobian and both error vectors are expressed in the base frame.
    const Jacobian jacobian = differential(joints);
    Eigen::Matrix<double, 6, 6> system = jacobian * jacobian.transpose();
    system.diagonal().array() += 0.0025 * 0.0025;
    Eigen::Matrix<double, 7, 1> step = jacobian.transpose() * system.ldlt().solve(error);
    if (!step.allFinite()) {
      break;
    }
    step *= 0.05 / std::max(0.05, step.cwiseAbs().maxCoeff());
    for (std::size_t i = 0; i < joints.size(); ++i) {
      joints[i] = std::clamp(joints[i] + step[i], limits.min[i], limits.max[i]);
    }
  }
  throw std::runtime_error("IK could not reach the target with the starting tool orientation");
}

template <typename Forward, typename Differential>
MotionPlan plan_move(const Joints& start, const Point& target, const Point& center,
                     const Eigen::Matrix3d& rotation, const JointLimits& limits,
                     const Forward& forward, const Differential& differential) {
  if (!within_radius(target, center, kPlannedRadiusM)) {
    throw std::runtime_error("Target is outside the 0.18 m planning radius");
  }
  for (std::size_t i = 0; i < start.size(); ++i) {
    if (!(start[i] >= limits.min[i] && start[i] <= limits.max[i])) {
      throw std::runtime_error("Starting joint position is outside the planning limits");
    }
  }
  if (!within_radius(forward(start).template topRightCorner<3, 1>(), center, kPlannedRadiusM)) {
    throw std::runtime_error("Starting tool position is outside the 0.18 m planning radius");
  }

  ruckig::InputParameter<7> input;
  input.current_position = start;
  input.target_position = solve_ik(start, target, rotation, limits, forward, differential);
  // Plan from rest to rest at low joint speeds for hardware bring-up.
  input.current_velocity.fill(0.0);
  input.current_acceleration.fill(0.0);
  input.target_velocity.fill(0.0);
  input.target_acceleration.fill(0.0);
  input.max_velocity.fill(0.15);
  input.max_acceleration.fill(0.5);
  input.max_jerk.fill(2.0);
  input.duration_discretization = ruckig::DurationDiscretization::Discrete;

  ruckig::Ruckig<7> generator(kCycleS);
  ruckig::Trajectory<7> trajectory;
  if (!generator.validate_input(input, true, true) ||
      generator.calculate(input, trajectory) != ruckig::Result::Working) {
    throw std::runtime_error("Ruckig could not plan a joint trajectory");
  }

  // Check continuous joint extrema and the tool path at each command tick.
  std::array<ruckig::Bound, 7> extrema;
  trajectory.get_position_extrema(extrema);
  for (std::size_t i = 0; i < extrema.size(); ++i) {
    if (extrema[i].min < limits.min[i] || extrema[i].max > limits.max[i]) {
      throw std::runtime_error("Planned path exceeds joint position limits");
    }
  }
  const double duration = trajectory.get_duration();
  const auto steps = static_cast<std::size_t>(std::llround(duration / kCycleS));
  MotionPlan plan{{}, duration, 0.0};
  plan.samples.reserve(steps + 1);
  for (std::size_t i = 0; i <= steps; ++i) {
    Joints command;
    trajectory.at_time(std::min(i * kCycleS, duration), command);
    const Transform pose = forward(command);
    const double radius = (pose.topRightCorner<3, 1>() - center).norm();
    if (!(radius <= kPlannedRadiusM)) {
      throw std::runtime_error("Planned tool path leaves the 0.18 m planning radius");
    }
    plan.max_radius_m = std::max(plan.max_radius_m, radius);
    plan.samples.push_back(command);
  }
  // Match the exact controller starting position, including a zero-duration plan.
  plan.samples.front() = start;
  return plan;
}

}  // namespace catchbox
