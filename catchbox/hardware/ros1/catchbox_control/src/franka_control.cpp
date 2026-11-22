// ROS 1 ros_control joint-position controller. Runs in franka_control's RT loop.
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <controller_interface/multi_interface_controller.h>
#include <franka/rate_limiting.h>
#include <franka_hw/franka_state_interface.h>
#include <hardware_interface/joint_command_interface.h>
#include <pluginlib/class_list_macros.h>
#include <realtime_tools/realtime_buffer.h>
#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <urdf/model.h>

namespace catchbox_control {
struct Target { std::array<double,7> q{}; ros::Time received; bool valid=false; };

class FrankaControl : public controller_interface::MultiInterfaceController<
    hardware_interface::PositionJointInterface, franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* hw, ros::NodeHandle& nh) override {
    const std::string arm="panda";
    for (int i=1;i<=7;++i) names_.push_back(arm+"_joint"+std::to_string(i));
    double max_velocity, max_acceleration, max_jerk;
    std::string robot_description;
    if (!nh.getParam("enable_motion", enabled_) ||
        !nh.getParam("max_velocity",max_velocity) ||
        !nh.getParam("max_acceleration",max_acceleration) ||
        !nh.getParam("max_jerk",max_jerk) ||
        !nh.getParam("command_timeout",timeout_) ||
        !ros::param::get("/robot_description",robot_description) ||
        !std::isfinite(timeout_) || timeout_<=0 ||
        !std::isfinite(max_velocity) || max_velocity<=0 ||
        !std::isfinite(max_acceleration) || max_acceleration<=0 ||
        !std::isfinite(max_jerk) || max_jerk<=0) {
      ROS_ERROR("catchbox: missing or invalid parameters or robot_description"); return false;
    }
    urdf::Model model;
    if (!model.initString(robot_description)) {
      ROS_ERROR("catchbox: could not parse robot_description"); return false;
    }
    for (size_t i=0;i<7;++i) {
      const auto joint=model.getJoint(names_[i]);
      if (!joint || joint->type != urdf::Joint::REVOLUTE || !joint->limits ||
          !std::isfinite(joint->limits->lower) || !std::isfinite(joint->limits->upper) ||
          !std::isfinite(joint->limits->velocity) || joint->limits->velocity<=0 ||
          max_velocity > joint->limits->velocity) {
        ROS_ERROR_STREAM("catchbox: invalid URDF joint limits for "<<names_[i]); return false;
      }
      // Keep a small margin inside the URDF position limits.
      lo_[i]=joint->limits->lower+.01;hi_[i]=joint->limits->upper-.01;
      if (!(lo_[i]<hi_[i])) return false;
      vmax_[i]=max_velocity;amax_[i]=max_acceleration;jmax_[i]=max_jerk;
    }
    auto* joints=hw->get<hardware_interface::PositionJointInterface>();
    auto* state=hw->get<franka_hw::FrankaStateInterface>();
    if (!joints || !state) return false;
    try {
      for (auto& name:names_) handles_.push_back(joints->getHandle(name));
      state_.reset(new franka_hw::FrankaStateHandle(state->getHandle(arm+"_robot")));
    } catch (const hardware_interface::HardwareInterfaceException& e) {
      ROS_ERROR_STREAM("catchbox interface: "<<e.what()); return false;
    }
    target_.writeFromNonRT(Target{});
    sub_=nh.subscribe("joint_target",1,&FrankaControl::command,this,
                      ros::TransportHints().tcpNoDelay());
    return true;
  }
  void starting(const ros::Time&) override { target_.writeFromNonRT(Target{}); }
  void update(const ros::Time& time,const ros::Duration& period) override {
    const auto state=state_->getRobotState();
    std::array<double,7> requested=state.q_d;
    const Target* target=target_.readFromRT();
    const double age=(time-target->received).toSec();
    if (enabled_ && target->valid && age>=0 && age<=timeout_ &&
        period.toSec()>0.0005 && period.toSec()<0.0015) requested=target->q;
    // libfranka's limiter assumes 1 ms and is based on the robot's last desired
    // position, velocity and acceleration, including packet-loss compensation.
    auto limited=franka::limitRate(vmax_,amax_,jmax_,requested,state.q_d,state.dq_d,state.ddq_d);
    for (size_t i=0;i<7;++i) handles_[i].setCommand(limited[i]);
  }
 private:
  void command(const sensor_msgs::JointStateConstPtr& msg) {
    if (msg->name.size()!=7 || msg->position.size()!=7) return;
    Target t;
    for (size_t i=0;i<7;++i) {
      if (msg->name[i]!=names_[i] || !std::isfinite(msg->position[i]) ||
          msg->position[i]<lo_[i] || msg->position[i]>hi_[i]) return;
      t.q[i]=msg->position[i];
    }
    t.received=ros::Time::now();t.valid=true;target_.writeFromNonRT(t);
  }
  std::vector<std::string> names_;std::vector<hardware_interface::JointHandle> handles_;
  std::unique_ptr<franka_hw::FrankaStateHandle> state_;
  std::array<double,7> lo_{},hi_{},vmax_{},amax_{},jmax_{};
  bool enabled_=false;double timeout_=0;ros::Subscriber sub_;realtime_tools::RealtimeBuffer<Target> target_;
};
}
PLUGINLIB_EXPORT_CLASS(catchbox_control::FrankaControl, controller_interface::ControllerBase)
