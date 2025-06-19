#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/impl/utils.h>
#include <yaml-cpp/yaml.h>

using namespace std::chrono_literals;

class MazeSolver : public rclcpp::Node {
public:
  MazeSolver(int scene_number)
      : Node("pid_maze_solver"), scene_number_(scene_number), got_odom_(false),
        turn_phase_(false), wp_reached_(false), init_(true), paused_(false),
        target_wp_(0) {
    twist_pub_ =
        this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
        "/odometry/filtered", 10,
        std::bind(&MazeSolver::odomCallback, this, std::placeholders::_1));
    scan_sub_ = this->create_subscription<sensor_msgs::msg::LaserScan>(
        "/scan_filtered", 10,
        std::bind(&MazeSolver::scanCallback, this, std::placeholders::_1));
    timer_ = this->create_wall_timer(
        200ms, std::bind(&MazeSolver::executeCallback, this));

    SelectWaypoints();
    clock_ = std::make_shared<rclcpp::Clock>(RCL_ROS_TIME);
  }

private:
  void SelectWaypoints() {
    // Waypoints [dx, dy, dphi] in robot frame
    switch (scene_number_) {

    case 1: { // Simulation
      std::string yaml_path =
          "/home/user/ros2_ws/src/pid_maze_solver/waypoints/waypoints_sim.yaml";
      YAML::Node config = YAML::LoadFile(yaml_path);
      auto waypoints =
          config["pid_maze_solver"]["ros__parameters"]["waypoints_sim"];
      for (size_t i = 0; i < 14; ++i) {
        waypoints_[i] = Eigen::Vector3f(waypoints[i * 3].as<float>(),
                                        waypoints[i * 3 + 1].as<float>(),
                                        waypoints[i * 3 + 2].as<float>());
      }
      break;
    }
    case 2: { // CyberWorld
      std::string yaml_path = "/home/user/ros2_ws/src/pid_maze_solver/"
                              "waypoints/waypoints_real.yaml";
      YAML::Node config = YAML::LoadFile(yaml_path);
      auto waypoints =
          config["pid_maze_solver"]["ros__parameters"]["waypoints_real"];
      for (size_t i = 0; i < 14; ++i) {
        waypoints_[i] = Eigen::Vector3f(waypoints[i * 3].as<float>(),
                                        waypoints[i * 3 + 1].as<float>(),
                                        waypoints[i * 3 + 2].as<float>());
      }
      break;
    }

    case 3: { // Simulation reverse
      std::string yaml_path =
          "/home/user/ros2_ws/src/pid_maze_solver/waypoints/"
          "reverse_waypoints_sim.yaml";
      YAML::Node config = YAML::LoadFile(yaml_path);
      auto waypoints =
          config["pid_maze_solver"]["ros__parameters"]["waypoints_sim"];
      for (size_t i = 0; i < 14; ++i) {
        waypoints_[i] = Eigen::Vector3f(waypoints[i * 3].as<float>(),
                                        waypoints[i * 3 + 1].as<float>(),
                                        waypoints[i * 3 + 2].as<float>());
      }
      break;
    }
    case 4: { // CyberWorld reverse
      std::string yaml_path =
          "/home/user/ros2_ws/src/pid_maze_solver/waypoints/"
          "reverse_waypoints_real.yaml";
      YAML::Node config = YAML::LoadFile(yaml_path);
      auto waypoints =
          config["pid_maze_solver"]["ros__parameters"]["waypoints_real"];
      for (size_t i = 0; i < 14; ++i) {
        waypoints_[i] = Eigen::Vector3f(waypoints[i * 3].as<float>(),
                                        waypoints[i * 3 + 1].as<float>(),
                                        waypoints[i * 3 + 2].as<float>());
      }
      break;
    }
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

  void scanCallback(const sensor_msgs::msg::LaserScan::SharedPtr msg) {
    laser_ranges_ = msg->ranges;
  }

  void executeCallback() {
    if (!got_odom_) {
      RCLCPP_WARN(this->get_logger(), "Odom data not received!");
      return;
    }
    if (laser_ranges_.empty()) {
      RCLCPP_WARN(this->get_logger(), "Scan data missing!!");
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
    if (error_pose(2) > M_PI) {
      error_pose(2) -= 2 * M_PI;
    } else if (error_pose(2) < -M_PI) {
      error_pose(2) += 2 * M_PI;
    }

    // Check distance to target waypoint
    if (std::hypot(error_pose(0), error_pose(1)) < 0.02) {
      auto msg = geometry_msgs::msg::Twist();
      twist_pub_->publish(msg);

      // Perform wall avoidance
      if (performWallAvoidance()) {
        return;
      }

      if (std::abs(error_pose(2)) > 0.02) {
        // PID Controller - Angular
        integral_error_(2) += error_pose(2);
        integral_error_(2) =
            std::clamp(integral_error_(2), -int_limit_, int_limit_);

        float omega = Kp_ * error_pose(2) +
                      Kd_ * (error_pose(2) - prev_error_(2)) +
                      Ki_ * integral_error_(2);
        prev_error_(2) = error_pose(2);

        // Publish velocities
        auto cmd_vel = geometry_msgs::msg::Twist();
        cmd_vel.angular.z = std::clamp(omega, -max_ang_vel_, max_ang_vel_);
        twist_pub_->publish(cmd_vel);
      } else {
        prev_error_ = {0.0, 0.0, 0.0};
        integral_error_ = {0.0, 0.0, 0.0};
        wp_reached_ = true;
        target_wp_++;

        if (target_wp_ >= waypoints_.size()) {
          if (scene_number_ == 1 || scene_number_ == 2)
            RCLCPP_INFO(this->get_logger(), "Maze finished!");
          else if (scene_number_ == 3 || scene_number_ == 4) {
            RCLCPP_INFO(this->get_logger(), "Reverse-Maze finished!");
          }

          rclcpp::shutdown();
        } else {
          // Start the pause timer
          pause_time_ = clock_->now();
          paused_ = true;
          RCLCPP_INFO(this->get_logger(),
                      "Waypoint reached, stopping briefly...");
        }
      }
      return;
    }

    // PID Controller - Linear
    integral_error_ += error_pose;
    integral_error_(0) =
        std::clamp(integral_error_(0), -int_limit_, int_limit_);
    integral_error_(1) =
        std::clamp(integral_error_(1), -int_limit_, int_limit_);

    Eigen::Vector3f V = Kp_ * error_pose + Kd_ * (error_pose - prev_error_) +
                        Ki_ * integral_error_;
    prev_error_ = error_pose;
    V = recomputeTwist(V);

    // Apply course correction
    V = performCourseCorrection(V);

    // Publish linear velocities
    auto cmd_vel = geometry_msgs::msg::Twist();
    cmd_vel.linear.x = std::clamp(V(0), -max_lin_vel_, max_lin_vel_);
    cmd_vel.linear.y = std::clamp(V(1), -max_lin_vel_, max_lin_vel_);
    cmd_vel.angular.z = 0.0;
    twist_pub_->publish(cmd_vel);
  }

  Eigen::Vector3f recomputeTwist(const Eigen::Vector3f &V) {

    float dphi = current_pose_(2);
    Eigen::Matrix3f R{{std::cos(dphi), std::sin(dphi), 0.0},
                      {-std::sin(dphi), std::cos(dphi), 0.0},
                      {0.0, 0.0, 1.0}};
    // Velocity command vector transposed into robot frame
    Eigen::Vector3f nu = R * V;
    return nu;
  }

  bool performWallAvoidance() {

    float front = laser_ranges_[0];
    float back = laser_ranges_[459];
    float left = laser_ranges_[229];
    float right = laser_ranges_[689];

    auto cmd_vel = geometry_msgs::msg::Twist();
    bool avoided = false;

    // Thresholds
    const float min_dist = 0.2;
    const float backoff_vel = 0.1;

    if (front < min_dist) {
      cmd_vel.linear.x = -backoff_vel;
      avoided = true;
    } else if (back < min_dist) {
      cmd_vel.linear.x = backoff_vel;
      avoided = true;
    }

    if (left < min_dist) {
      cmd_vel.linear.y = -backoff_vel;
      avoided = true;
    } else if (right < min_dist) {
      cmd_vel.linear.y = backoff_vel;
      avoided = true;
    }

    if (avoided) {
      RCLCPP_WARN(this->get_logger(), "Wall too close. Avoiding...");
      twist_pub_->publish(cmd_vel);
      return true;
    }

    return false;
  }

  Eigen::Vector3f performCourseCorrection(const Eigen::Vector3f &cmd_vel) {

    float left = laser_ranges_[229];
    float right = laser_ranges_[689];

    const float critical_dist = 0.15;
    const float min_valid = 0.05;
    const float max_valid = 5.0;

    Eigen::Vector3f vel_updt = cmd_vel;

    bool left_valid = (left > min_valid && left < max_valid);
    bool right_valid = (right > min_valid && right < max_valid);

    // Apply only if moving linearly
    if (std::abs(cmd_vel(0)) > 0.01 || std::abs(cmd_vel(1)) > 0.01) {
      if (left_valid && left < critical_dist) {
        vel_updt(1) += 0.05;
      } else if (right_valid && right < critical_dist) {
        vel_updt(1) -= 0.05;
      } else if (left_valid && right_valid) {
        float drift_error = right - left;
        if (std::abs(drift_error) > 0.15) {
          vel_updt(1) += -0.01 * (drift_error > 0 ? 1 : -1);
        }
      }
    }
    return vel_updt;
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Time pause_time_;
  int scene_number_;
  bool got_odom_, turn_phase_, wp_reached_, init_, paused_;
  size_t target_wp_;
  std::shared_ptr<rclcpp::Clock> clock_;
  Eigen::Vector3f current_pose_{0.0, 0.0, 0.0};
  Eigen::Vector3f target_pose_{0.0, 0.0, 0.0};
  Eigen::Vector3f prev_error_{0.0, 0.0, 0.0};
  Eigen::Vector3f integral_error_{0.0, 0.0, 0.0};
  std::array<Eigen::Vector3f, 14> waypoints_;
  std::vector<float> laser_ranges_;

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
  int scene_number = 3; // Default scene number to simulation
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
