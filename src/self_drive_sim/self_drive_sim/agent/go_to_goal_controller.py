import math

class GoToGoalController:
    def __init__(self,
            max_linear_velocity=1.0,
            max_angular_velocity=2.0,
            linear_velocity_gain=1.0,
            angular_velocity_gain=2.0,
            angle_threshold=0.05
        ) -> None:
        self.max_linear_velocity = max_linear_velocity
        self.max_angular_velocity = max_angular_velocity
        self.linear_velocity_gain = linear_velocity_gain
        self.angular_velocity_gain = angular_velocity_gain
        self.angle_threshold = angle_threshold

    def go_to_goal_controller(self, 
            current_robot_pose, 
            target_position,
            is_obstacle=False,
        ) -> tuple[float, float]:
        """
        Returns
        -------
        - linear_velocity
        - angular_velocity

        Usings
        ------
        - self.angle_threshold
        - self.max_angular_velocity
        - self.max_linear_velocity
        - self.linear_velocity_gain
        - self.angular_velocity_gain
        """
        current_robot_x, current_robot_y, current_robot_yaw = current_robot_pose
        target_x, target_y = target_position

        dx = current_robot_x - target_x
        dy = current_robot_y - target_y
        d_distance = math.hypot(dx, dy)

        target_angle = math.atan2(-dy, -dx)  
        d_yaw = math.atan2(math.sin(target_angle - current_robot_yaw),
                           math.cos(target_angle - current_robot_yaw))

        if abs(d_yaw) > self.angle_threshold:
            linear_velocity = 0.0
            angular_velocity = max(-self.max_angular_velocity,
                min(self.max_angular_velocity, self.angular_velocity_gain * d_yaw))
        elif is_obstacle:
            linear_velocity = 0.0
            angular_velocity = 0.0
        else:
            linear_velocity = max(-self.max_linear_velocity,
                min(self.max_linear_velocity, self.linear_velocity_gain * d_distance))
            angular_velocity = max(-self.max_angular_velocity,
                min(self.max_angular_velocity, self.angular_velocity_gain * d_yaw))

        return linear_velocity, angular_velocity
