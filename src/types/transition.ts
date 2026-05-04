/**
 * Represents a single transition in an RL episode.
 * This is the atomic unit of experience that gets stored in replay buffers
 * and used for agent learning.
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface Transition<S, A> {
	/** Current state before taking the action */
	state: S;

	/** Action taken in the current state */
	action: A;

	/** Immediate reward received */
	reward: number;

	/** Next state after taking the action */
	nextState: S;

	/** Whether the episode terminated */
	done: boolean;

	/** Optional metadata about the transition */
	info?: Record<string, unknown>;
}
