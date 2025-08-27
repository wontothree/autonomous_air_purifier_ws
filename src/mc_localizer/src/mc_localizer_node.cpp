#include <mc_localizer/mc_localizer_ros.hpp>

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    std::cout << "package mc_localizer is started\n";

    // entry point
    rclcpp::spin(std::make_shared<mc_localizer::MCLocalizerROS>());

    rclcpp::shutdown();
    std::cout << "shutdown" << std::endl;

    return 0;
}
