/**
 * Result of a single step in an environment.
 *
 * @template S - Type of the state representation
 */
export interface StepResult<S> {
	/** The new state after the step */
	state: S;

	/** The reward received for the transition */
	reward: number;

	/** Whether the episode has terminated */
	done: boolean;

	/** Optional metadata about the step */
	info?: Record<string, unknown>;
}

/**
 * An environment defines the world dynamics and how the agent interacts with it.
 *
 * Environments can be synchronous or asynchronous (return Promises).
 * This flexibility allows for:
 * - Simple tabular environments (sync)
 * - Simulators or real-world systems (async)
 * - Game engines or network-based environments
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface Environment<S, A> {
	/**
	 * Reset the environment to an initial state.
	 *
	 * @param seed - Optional random seed for reproducibility
	 * @returns The initial state
	 */
	reset(seed?: number): S | Promise<S>;

	/**
	 * Execute one step in the environment with the given action.
	 *
	 * @param action - The action to execute
	 * @returns The result of the step (new state, reward, done flag)
	 */
	step(action: A): StepResult<S> | Promise<StepResult<S>>;

	/**
	 * Optional method to render the environment state.
	 * Useful for visualization and debugging.
	 */
	render?(): void | Promise<void>;
}
