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
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
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

    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

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

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr /*msg*/) {
    // Use tf2 to get transform from odom to base_link
    geometry_msgs::msg::TransformStamped transformStamped;
    try {
      transformStamped =
          tf_buffer_->lookupTransform("odom", "base_link", tf2::TimePointZero);

      // Translation
      current_pose_(0) = transformStamped.transform.translation.x;
      current_pose_(1) = transformStamped.transform.translation.y;

      // Yaw from quaternion
      tf2::Quaternion q(transformStamped.transform.rotation.x,
                        transformStamped.transform.rotation.y,
                        transformStamped.transform.rotation.z,
                        transformStamped.transform.rotation.w);
      current_pose_(2) = tf2::impl::getYaw(q);

      got_odom_ = true;
    } catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform odom->base_link: %s",
                  ex.what());
      got_odom_ = false;
    }
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
          RCLCPP_INFO(this->get_logger(), "Maze finished!");
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

    // Transpose velocity command vector into robot frame
    V = recomputeTwist(V);

    // Apply simple obstacle avoidance
    // if (std::abs(V(0)) > 0.01 || std::abs(V(1)) > 0.01) {
    V = performObstacleAvoidance(V);
    // }

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
    Eigen::Vector3f nu = R * V;
    return nu;
  }

  Eigen::Vector3f performObstacleAvoidance(const Eigen::Vector3f &cmd_vel) {

    const float front = laser_ranges_[0];
    const float left = laser_ranges_[179];
    const float back = laser_ranges_[359];
    const float right = laser_ranges_[579];

    const float min_valid = 0.05;
    const float max_valid = 5.0;
    const float critical_dist = 0.21;

    bool left_valid = (left > min_valid && left < max_valid);
    bool right_valid = (right > min_valid && right < max_valid);
    bool back_valid = (back > min_valid && back < max_valid);
    bool front_valid = (front > min_valid && front < max_valid);

    Eigen::Vector3f vel_update = cmd_vel;
    Eigen::Vector2f target_correction = Eigen::Vector2f::Zero();

    // Perform course correction
    if (left_valid && left < critical_dist) {
      RCLCPP_WARN(this->get_logger(), "Course correcting to right...");
      vel_update(1) -= 0.05;
      target_correction(1) -= 0.003; // update target to right

    } else if (right_valid && right < critical_dist - 0.01) {
      RCLCPP_WARN(this->get_logger(), "Course correcting to left...");
      vel_update(1) += 0.05;
      target_correction(1) += 0.003; // update target to left
    }

    // Perform wall avoidance
    if (front_valid && front < critical_dist - 0.01) {
      RCLCPP_WARN(this->get_logger(), "Avoiding front wall...");
      vel_update(0) = -0.05;
      target_correction(0) -= 0.003; // update target to back

    } else if (back_valid && back < critical_dist + 0.02) {
      RCLCPP_WARN(this->get_logger(), "Avoiding back wall...");
      vel_update(0) = 0.05;
      target_correction(0) += 0.003; // update target to front
    }

    // Transform corrections
    if (target_correction.norm() > 0.0) {
      const float dphi = current_pose_(2);
      Eigen::Matrix2f R{{std::cos(dphi), std::sin(dphi)},
                        {-std::sin(dphi), std::cos(dphi)}};
      Eigen::Vector2f tranformed_target = R * target_correction;
      target_pose_.head<2>() += tranformed_target;
    }

    return vel_update;
  }

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr scan_sub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
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
  const float Kp_ = 0.35;
  const float Ki_ = 0.005, int_limit_ = 5.0;
  const float Kd_ = 0.32;
  const float max_lin_vel_ = 0.18;
  const float max_ang_vel_ = 0.5;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  // Check if a scene number argument is provided
  int scene_number = 4; // Default scene number to reverse-cyberworld
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