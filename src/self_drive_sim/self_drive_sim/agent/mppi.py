class ModelPredictivePathIntegralController:
    def __init__(self,
            state_dimension: int = 3,
            control_dimension: int = 2,

            prediction_horizon: int = 20,
            control_period: float = 0.1,

            sample_num: int = 3000,
            max_control_inputs: np.ndarray = np.array([1.0, 2.0]),
            min_control_inputs: np.ndarray = np.array([0.0, -2.0]),
            non_biased_sampling_rate: float = 0.1,
            collision_cost_weight: float = 1.0,

            softmax_lambda: float = 0.3,
        ):
        self.state_dimension = state_dimension
        self.control_dimension = control_dimension
        self.prediction_horizon = prediction_horizon
        self.control_period = control_period
        self.sample_num = sample_num
        self.max_control_inputs = max_control_inputs
        self.min_control_inputs = min_control_inputs
        self.non_biased_sampling_rate = non_biased_sampling_rate
    
        self.collision_cost_weight = collision_cost_weight
        self.softmax_lambda = softmax_lambda

        # solve
        self.previous_control_sequence = np.zeros((self.prediction_horizon - 1, self.control_dimension))

    def sample_control_sequence(self,
            control_sequence_mean: np.ndarray,
            control_sequence_covariance_matrices: np.ndarray
        ):
        num_biased = int((1 - self.non_biased_sampling_rate) * self.sample_num)

        noise = np.random.multivariate_normal(
            mean=np.zeros(self.control_dimension),
            cov=control_sequence_covariance_matrices,
            size=(self.sample_num, self.prediction_horizon - 1)
        )

        noised_control_sequences = np.zeros_like(noise)
        noised_control_sequences[:num_biased] = control_sequence_mean + noise[:num_biased]
        noised_control_sequences[num_biased:] = noise[num_biased:]

        noised_control_sequences = np.clip(
            noised_control_sequences,
            self.min_control_inputs,
            self.max_control_inputs
        )

        return noised_control_sequences
    
    def predict_state_sequence(self,
            current_state: np.ndarray,
            control_sequence: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray]:
        # Declare state sequence
        global_state_sequence = np.zeros((self.prediction_horizon, 3))
        local_state_sequence = np.zeros((self.prediction_horizon, 3))

        # Set initial state
        global_state_sequence[0] = current_state
        local_state_sequence[0] = [0.0, 0.0, 0.0]

        for i in range(self.prediction_horizon - 1):
            noisy_linear_velocity, noisy_angular_velocity = control_sequence[i]

            # Update global state
            global_x, global_y, global_yaw = global_state_sequence[i]
            delta_global_x = noisy_linear_velocity * np.cos(global_yaw + noisy_angular_velocity * self.control_period / 2) * self.control_period
            delta_global_y = noisy_linear_velocity * np.sin(global_yaw + noisy_angular_velocity * self.control_period / 2) * self.control_period
            delta_global_yaw = noisy_angular_velocity * self.control_period

            global_state_sequence[i + 1] = [
                global_x + delta_global_x, 
                global_y + delta_global_y, 
                (global_yaw + delta_global_yaw + np.pi) % (2 * np.pi) - np.pi
            ]

            # Update local state
            local_x, local_y, local_yaw = local_state_sequence[i]
            delta_local_x = noisy_linear_velocity * np.cos(local_yaw + noisy_angular_velocity * self.control_period / 2) * self.control_period
            delta_local_y = noisy_linear_velocity * np.sin(local_yaw  + noisy_angular_velocity * self.control_period / 2) * self.control_period
            delta_local_yaw = noisy_angular_velocity * self.control_period

            local_state_sequence[i + 1] = [
                local_x + delta_local_x, 
                local_y + delta_local_y, 
                (local_yaw + delta_local_yaw + np.pi) % (2 * np.pi) - np.pi
            ]

        return global_state_sequence, local_state_sequence
    
    def calculate_state_sequence_cost(self,
            global_state_sequence: np.ndarray,
            local_state_sequence: np.ndarray,
            target_position: tuple[float, float],
            local_costmap: np.ndarray
        ) -> float:
        total_cost = 0

        target_x, target_y = target_position
        for i in range(self.prediction_horizon):
            global_x, global_y, global_yaw = global_state_sequence[i]
            target_yaw = np.arctan2(target_y - global_y, target_x - global_x)
            difference_yaw = target_yaw - global_yaw
            difference_yaw = (difference_yaw + np.pi) % (2 * np.pi) - np.pi

            difference_x = global_x - target_x
            difference_y = global_y - target_y

            cost = 100 * (difference_x**2 + difference_y**2) + 0.0001 * difference_yaw**2

            # test
            # 장애물과의 충돌을 확인하기 위한 코드
            state_x, state_y, _ = local_state_sequence[i]
            state_x_index = int(state_x / 0.02)  # 해상도에 맞춰 인덱싱
            state_y_index = int(state_y / 0.02)

            # 장애물 여부 확인 (로컬 costmap에서 장애물이 있는지 체크)
            if 0 <= state_x_index < local_costmap.shape[0] and 0 <= state_y_index < local_costmap.shape[1]:
                if local_costmap[state_x_index, state_y_index] == 100:  # max_cost가 장애물 값을 의미
                    # 장애물에 겹치면 추가 비용 부여
                    cost += 50  # 장애물과 겹치는 경우 비용을 크게 추가

            total_cost += cost

        return total_cost
    
    def calculate_sample_cost(self,
            control_sequence: np.ndarray,
            control_cost_matrix: np.ndarray = np.diag([1, 0.0001]),
            lambda_: float = 0.001   
        ) -> float:
        cost = lambda_ *np.sum((control_sequence @ control_cost_matrix) * control_sequence)
        
        return cost
            
    def softmax(self, 
            costs 
        ):
        min_cost = np.min(costs) 
        normalizing_constant = np.sum(np.exp(-(costs - min_cost) / self.softmax_lambda)) + 1e-10
        softmax_costs = np.exp(-(costs - min_cost) / self.softmax_lambda) / normalizing_constant

        return softmax_costs

    def solve(self, 
            current_state, 
            target_position,
            local_costmap
        ):
        # Sample control sequences
        noised_control_sequences = self.sample_control_sequence(
            control_sequence_mean=self.previous_control_sequence,
            control_sequence_covariance_matrices= 0.01 * np.identity(self.control_dimension)
        )

        # Evaluate costs for each sample
        costs = np.zeros(self.sample_num)
        control_costs = np.zeros(self.sample_num)
        for i in range(self.sample_num):
            global_state_sequence, local_state_sequence = self.predict_state_sequence(
                current_state=current_state, 
                control_sequence=noised_control_sequences[i]
            )

            costs[i] = self.calculate_state_sequence_cost(
                global_state_sequence=global_state_sequence, 
                local_state_sequence=local_state_sequence, 
                target_position=target_position, 
                local_costmap=local_costmap
            )

            control_costs[i] = self.calculate_sample_cost(
                control_sequence=noised_control_sequences[i]
            )

        softmax_weights = self.softmax(costs + control_costs)

        optimal_control_sequence = np.sum(softmax_weights[:, np.newaxis, np.newaxis] * noised_control_sequences, axis=0)

        # For biased random sampling
        self.previous_control_sequence = optimal_control_sequence

        return optimal_control_sequence