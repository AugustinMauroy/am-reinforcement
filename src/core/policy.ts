/**
 * A stochastic policy that samples actions from a distribution.
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface Policy<S, A> {
	/**
	 * Sample an action from the policy given a state.
	 *
	 * @param state - Current state
	 * @returns Sampled action
	 */
	sample(state: S): A;

	/**
	 * Optional: Get the probability distribution over actions.
	 * Used for entropy computation and policy analysis.
	 *
	 * @param state - Current state
	 * @returns Array of probabilities for each action (if applicable)
	 */
	probabilities?(state: S): number[];
}

/**
 * Estimates the expected cumulative reward from a given state.
 *
 * @template S - Type of the state representation
 */
export interface ValueFunction<S> {
	/**
	 * Predict the value (expected return) of a state.
	 *
	 * @param state - Current state
	 * @returns Predicted value
	 */
	predict(state: S): number;
}

/**
 * Estimates the expected cumulative reward for a state-action pair.
 * Also called the Q-function.
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface QFunction<S, A> {
	/**
	 * Predict the Q-value (expected return) of a state-action pair.
	 *
	 * @param state - Current state
	 * @param action - Action taken
	 * @returns Predicted Q-value
	 */
	predict(state: S, action: A): number;
}
