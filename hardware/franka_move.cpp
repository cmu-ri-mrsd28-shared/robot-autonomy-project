#include "motion_plan.hpp"

#include <franka/model.h>
#include <franka/robot.h>

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>

namespace {

using namespace catchbox;

struct Options {
  std::string robot = "localhost";
  std::string model;
  std::optional<Point> center;
  std::optional<Point> target;
  std::optional<Point> offset;
  bool execute = false;
};

void usage() {
  std::cout << "Usage: franka_move [--robot HOST] [--model fr3|panda]\n"
               "                   [--center X Y Z] [--target X Y Z | --offset DX DY DZ]\n"
               "                   [--execute]\n"
               "FCI host defaults to localhost. Set FRANKA_HOST or pass --robot HOST to override.\n"
               "Coordinates and offsets are in meters along the Franka base axes.\n"
               "With no target or offset, print the measured tool position.\n"
               "With a target or offset, require a model and check the plan without moving.\n"
               "Motion requires --execute and an explicit --center in addition to the plan.\n";
}

Point parse_point(int argc, char* argv[], int& i, const std::string& option) {
  if (i + 3 >= argc) {
    throw std::invalid_argument(option + " requires three coordinates");
  }
  Point point;
  for (int axis = 0; axis < 3; ++axis) {
    const std::string value = argv[++i];
    std::size_t end = 0;
    try {
      point[axis] = std::stod(value, &end);
    } catch (const std::logic_error&) {
      throw std::invalid_argument(option + " requires three finite coordinates");
    }
    if (end != value.size() || !std::isfinite(point[axis])) {
      throw std::invalid_argument(option + " requires three finite coordinates");
    }
  }
  return point;
}

Options parse_options(int argc, char* argv[]) {
  Options options;
  if (const char* host = std::getenv("FRANKA_HOST"); host && *host) {
    options.robot = host;
  }
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if ((option == "--robot" || option == "--model") && i + 1 < argc &&
        std::string(argv[i + 1]).rfind("--", 0) != 0) {
      (option == "--robot" ? options.robot : options.model) = argv[++i];
    } else if (option == "--center") {
      options.center = parse_point(argc, argv, i, option);
    } else if (option == "--target") {
      options.target = parse_point(argc, argv, i, option);
    } else if (option == "--offset") {
      options.offset = parse_point(argc, argv, i, option);
    } else if (option == "--execute") {
      options.execute = true;
    } else {
      throw std::invalid_argument("Unknown or incomplete option: " + option);
    }
  }
  if (options.robot.empty()) {
    throw std::invalid_argument("--robot requires a hostname or IP address");
  }
  if (options.target && options.offset) {
    throw std::invalid_argument("Choose either --target or --offset");
  }
  if ((!options.model.empty() || options.target || options.offset) &&
      options.model != "fr3" && options.model != "panda") {
    throw std::invalid_argument("Select the connected arm with --model fr3 or --model panda");
  }
  if (options.execute && (!(options.target || options.offset) || !options.center)) {
    throw std::invalid_argument("--execute requires --center and either --target or --offset");
  }
  return options;
}

Point measured_tool(const franka::RobotState& state) {
  return Eigen::Map<const Point>(state.O_T_EE.data() + 12);
}

bool positions_match(const Joints& a, const Joints& b, double tolerance) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (!(std::abs(a[i] - b[i]) <= tolerance)) {
      return false;
    }
  }
  return true;
}

bool is_stationary(const franka::RobotState& state) {
  const Joints zero{};
  return positions_match(state.dq, zero, 0.01) &&
         positions_match(state.dq_d, zero, 1e-6) &&
         positions_match(state.ddq_d, zero, 1e-6) &&
         positions_match(state.q, state.q_d, 0.01);
}

void execute_move(franka::Robot& robot, const franka::RobotState& start,
                  const Point& target, const Point& center, const MotionPlan& plan) {
  std::size_t tick = 0;
  std::size_t settle_ticks = 0;
  double max_measured_radius = 0.0;
  const Point start_tool = measured_tool(start);

  // The callback only checks feedback and streams precomputed joint positions.
  // libfranka cancels its motion when a callback throws.
  robot.control(
      [&](const franka::RobotState& state, franka::Duration period) {
        const Point tool = measured_tool(state);
        const double radius = (tool - center).norm();
        if (!(radius <= kMeasuredRadiusM)) {
          throw std::runtime_error("Measured tool reached the workspace stop threshold");
        }
        max_measured_radius = std::max(max_measured_radius, radius);
        if (period.toMSec() > 5) {
          throw std::runtime_error("FCI feedback gap exceeded 5 ms");
        }
        if (tick == 0 &&
            (!is_stationary(state) || !positions_match(start.q_d, state.q_d, 1e-6) ||
             !within_radius(tool, start_tool, 0.002) || state.F_T_EE != start.F_T_EE ||
             state.EE_T_K != start.EE_T_K)) {
          throw std::runtime_error("Robot state changed after planning. Run a fresh plan");
        }

        const franka::JointPositions command(plan.samples[tick]);
        if (tick + 1 < plan.samples.size()) {
          ++tick;
        } else if (++settle_ticks >= 150) {
          return franka::MotionFinished(command);
        }
        return command;
      },
      franka::ControllerMode::kJointImpedance, true);

  const Point reached = measured_tool(robot.readOnce());
  const double final_error = (reached - target).norm();
  std::cout << "Final tool XYZ (m)      " << reached.transpose() << '\n'
            << "Final position error (m) " << final_error << '\n'
            << "Max measured radius (m)  " << max_measured_radius << '\n';
  if (!(final_error <= 0.005)) {
    throw std::runtime_error("Final position error exceeded 5 mm");
  }
}

int run(const Options& options) {
  std::cout << "Connecting to FCI at " << options.robot << std::endl;
  franka::Robot robot(options.robot);
  const auto start = robot.readOnce();
  const Point tool = measured_tool(start);
  if (!tool.allFinite()) {
    throw std::runtime_error("Robot did not report a finite tool position");
  }
  std::cout << std::fixed << std::setprecision(6)
            << "Measured tool XYZ (m)  " << tool.transpose() << '\n';
  if (!options.target && !options.offset) {
    return 0;
  }
  if (start.robot_mode != franka::RobotMode::kIdle || !is_stationary(start)) {
    throw std::runtime_error("Place the robot in FCI idle at rest before planning");
  }

  const Point center = options.center.value_or(tool);
  const Point target = options.target ? *options.target : Point(tool + *options.offset);
  if (!within_radius(tool, center, kPlannedRadiusM)) {
    throw std::runtime_error("Measured starting tool is outside the 0.18 m planning radius");
  }
  if (!within_radius(target, center, kPlannedRadiusM)) {
    throw std::runtime_error("Target is outside the 0.18 m planning radius");
  }
  const auto model = robot.loadModel();
  const auto forward = [&](const Joints& q) -> Transform {
    const auto pose = model.pose(franka::Frame::kEndEffector, q, start.F_T_EE, start.EE_T_K);
    return Eigen::Map<const Transform>(pose.data());
  };
  const auto differential = [&](const Joints& q) -> Jacobian {
    const auto data = model.zeroJacobian(franka::Frame::kEndEffector, q, start.F_T_EE, start.EE_T_K);
    return Eigen::Map<const Jacobian>(data.data());
  };
  const Transform measured_pose = Eigen::Map<const Transform>(start.O_T_EE.data());
  const Eigen::Matrix3d rotation = measured_pose.topLeftCorner<3, 3>();
  if (!within_radius(forward(start.q).topRightCorner<3, 1>(), tool, 0.005)) {
    throw std::runtime_error("Robot model and measured tool position differ by more than 5 mm");
  }

  const auto& limits = options.model == "fr3" ? kFr3Limits : kPandaLimits;
  const MotionPlan plan = plan_move(start.q_d, target, center, rotation, limits, forward, differential);
  std::cout << "Workspace center (m)   " << center.transpose() << '\n'
            << "Target tool XYZ (m)    " << target.transpose() << '\n'
            << "Planned duration (s)   " << plan.duration_s << '\n'
            << "Max planned radius (m) " << plan.max_radius_m << '\n';
  if (!options.execute) {
    std::cout << "Plan passed. No motion commanded.\n";
    return 0;
  }
  if (plan.samples.size() == 1) {
    if (!within_radius(tool, target, 0.005)) {
      throw std::runtime_error("Zero-duration plan with measured tool error above 5 mm");
    }
    std::cout << "Measured tool is already within 5 mm of the target.\n";
    return 0;
  }
  std::cout << "Executing one move using Franka joint impedance control.\n" << std::flush;
  execute_move(robot, start, target, center, plan);
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    usage();
    return 0;
  }
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "Franka move: " << error.what() << '\n';
    return 1;
  }
}
