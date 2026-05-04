import type { Transition } from "../types/transition.ts";

/**
 * An agent is responsible for:
 * - Deciding which action to take in a given state
 * - Learning from transitions experienced in the environment
 *
 * Agents can be synchronous or asynchronous (return Promises).
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface Agent<S, A> {
	/**
	 * Decide which action to take in the given state.
	 *
	 * @param state - Current state
	 * @returns The action to execute
	 */
	act(state: S): A | Promise<A>;

	/**
	 * Learn from a single transition.
	 *
	 * Some agents (like DQN) may batch learning internally,
	 * so this call might just store the transition for later processing.
	 *
	 * @param transition - The transition to learn from
	 */
	learn(transition: Transition<S, A>): void | Promise<void>;

	/**
	 * Optional: Save the agent's learned parameters to disk.
	 *
	 * @param path - File path to save to
	 */
	save?(path: string): void | Promise<void>;

	/**
	 * Optional: Load the agent's learned parameters from disk.
	 *
	 * @param path - File path to load from
	 */
	load?(path: string): void | Promise<void>;
}
