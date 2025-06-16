#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>

using namespace std::chrono_literals;

class MazeSolver : public rclcpp::Node {
public:
  MazeSolver(int scene_number)
      : Node("pid_maze_solver"), scene_number_(scene_number), got_odom_(false),
        wp_reached_(false), init_(true), paused_(false), target_wp_(0) {
    twist_pub_ =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&MazeSolver::odomCallback, this, std::placeholders::_1));
    timer_ = this->create_wall_timer(
        200ms, std::bind(&MazeSolver::executeCallback, this));

    SelectWaypoints();
    clock_ = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  }

private:
  void SelectWaypoints() {
    // Waypoints [dx, dy, dphi] in robot frame
    switch (scene_number_) {
    case 1: // Simulation
      waypoints_ = {{
          {0.335, 0.0, 0.0}, // w1
      }};
      break;

    case 2: // CyberWorld
      waypoints_ = {{
          {0.0, 0.0, -0.483}, // w1
          {0.0, 0.0, -0.812}, // w2
          {0.0, 0.0, 1.291},  // w3
      }};
      break;

    default:
      RCLCPP_ERROR(this->get_logger(), "Invalid Scene Number: %d",
                   scene_number_);
    }
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    auto orientation = msg->pose.pose.orientation;
    tf2::Quaternion q(orientation.x, orientation.y, orientation.z,
                      orientation.w);
    current_pose_(0) = msg->pose.pose.position.x;
    current_pose_(1) = msg->pose.pose.position.y;
    current_pose_(2) = tf2::impl::getYaw(q);
    got_odom_ = true;
  }

  void executeCallback() {
    if (!got_odom_) {
      RCLCPP_WARN(this->get_logger(), "Odom data not received!");
      return;
    }

    // Paused state
    if (paused_) {
      rclcpp::Time now = clock_->now();
      // Check if 2 seconds have elapsed
      if ((now - pause_time_).seconds() >= 2.0) {
        paused_ = false;
      } else {
        auto msg = geometry_msgs::msg::Twist();
        twist_pub_->publish(msg);
        return;
      }
    }

    // Update target waypoint
    if (wp_reached_ || init_) {
      target_pose_ = current_pose_ + waypoints_[target_wp_];

      wp_reached_ = false;
      init_ = false;
      size_t twp = target_wp_ + 1;
      RCLCPP_INFO(this->get_logger(), "Moving to next waypoint: %ld", twp);
    }

    // Error vector
    Eigen::Vector3f error_pose = target_pose_ - current_pose_;

    // Check distance to target
    if (error_pose.norm() < 0.02) {
      auto msg = geometry_msgs::msg::Twist();
      twist_pub_->publish(msg);

      prev_error_ = {0.0, 0.0, 0.0};
      integral_error_ = {0.0, 0.0, 0.0};
      wp_reached_ = true;
      target_wp_++;

      if (target_wp_ >= waypoints_.size()) {
        RCLCPP_INFO(this->get_logger(), "Maze finished!");
        rclcpp::shutdown();
      } else {
        // Start the pause timer
        pause_time_ = clock_->now();
        paused_ = true;
        RCLCPP_INFO(this->get_logger(),
                    "Waypoint reached, stopping briefly...");
      }
      return;
    }

    // PID Controller (Proportional + Integral + Derivative)

    integral_error_ += error_pose; // Sum of errors over time
    integral_error_(0) =
        std::clamp(integral_error_(0), -int_limit_, int_limit_);
    integral_error_(1) =
        std::clamp(integral_error_(1), -int_limit_, int_limit_);
    integral_error_(2) =
        std::clamp(integral_error_(2), -int_limit_, int_limit_);

    Eigen::Vector3f V = Kp_ * error_pose + Kd_ * (error_pose - prev_error_) +
                        Ki_ * integral_error_;
    prev_error_ = error_pose; // for next iteration

    // Publish velocities
    auto cmd_vel = geometry_msgs::msg::Twist();
    cmd_vel.linear.x = std::clamp(V(0), -max_lin_vel_, max_lin_vel_);
    cmd_vel.linear.y = std::clamp(V(1), -max_lin_vel_, max_lin_vel_);
    cmd_vel.angular.z = std::clamp(V(2), -max_ang_vel_, max_ang_vel_);
    twist_pub_->publish(cmd_vel);
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time pause_time_;
  int scene_number_;
  bool got_odom_, wp_reached_, init_, paused_;
  size_t target_wp_;
  std::shared_ptr<rclcpp::Clock> clock_;
  Eigen::Vector3f current_pose_{0.0, 0.0, 0.0};
  Eigen::Vector3f target_pose_{0.0, 0.0, 0.0};
  Eigen::Vector3f prev_error_{0.0, 0.0, 0.0};
  Eigen::Vector3f integral_error_{0.0, 0.0, 0.0};
  std::array<Eigen::Vector3f, 14> waypoints_;

  // PID Gains
  const float Kp_ = 1.2;
  const float Ki_ = 0.02, int_limit_ = 5.0;
  const float Kd_ = 0.8;
  const float max_lin_vel_ = 0.5;
  const float max_ang_vel_ = 0.8;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // Check if a scene number argument is provided
  int scene_number = 1; // Default scene number to simulation
  if (argc > 1) {
    scene_number = std::atoi(argv[1]);
  }

  auto controller_node = std::make_shared<MazeSolver>(scene_number);
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(controller_node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
