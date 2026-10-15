#include <memory>
#include "rclcpp/rclcpp.hpp"

class JointController : public rclcpp::Node
{
public:
    JointController()
        : Node("joint_controller")
    {
        RCLCPP_INFO(
            this->get_logger(),
            "Hexapod joint controller started!"
        );
    }
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<JointController>();

    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}
