import type { Policy } from "../core/policy.ts";

/**
 * Epsilon-greedy policy:
 * - With probability epsilon: pick a random action
 * - With probability (1 - epsilon): pick the action with highest value
 *
 * This balances exploration and exploitation.
 *
 * @template A - Type of the action representation
 */
export class EpsilonGreedyPolicy<A> implements Policy<never, A> {
	private actions: A[];
	private getGreedyAction: () => A;
	private epsilon: number;
	private rng: () => number;

	constructor(
		actions: A[],
		getGreedyAction: () => A,
		epsilon: number,
		rng: () => number = Math.random,
	) {
		this.actions = actions;
		this.getGreedyAction = getGreedyAction;
		this.epsilon = epsilon;
		this.rng = rng;
		if (epsilon < 0 || epsilon > 1) {
			throw new Error("Epsilon must be between 0 and 1");
		}
		if (actions.length === 0) {
			throw new Error("Must provide at least one action");
		}
	}

	sample(): A {
		if (this.rng() < this.epsilon) {
			// Explore: random action
			const randomIdx = Math.floor(this.rng() * this.actions.length);
			return this.actions[randomIdx];
		}

		// Exploit: greedy action
		return this.getGreedyAction();
	}

	probabilities(): number[] {
		const uniformProb = this.epsilon / this.actions.length;
		const greedyProb = 1 - this.epsilon;

		// Assume the first action is the greedy action for probability distribution
		// In practice, we can't know which is greedy without evaluating all
		// This is a simplified version for reference purposes
		return this.actions.map((_, idx) => {
			return idx === 0 ? uniformProb + greedyProb : uniformProb;
		});
	}

	/**
	 * Decay epsilon over time (optional).
	 * Common pattern: multiply by a decay factor each episode.
	 */
	decayEpsilon(decayFactor: number): void {
		this.epsilon *= decayFactor;
	}

	getEpsilon(): number {
		return this.epsilon;
	}

	setEpsilon(epsilon: number): void {
		if (epsilon < 0 || epsilon > 1) {
			throw new Error("Epsilon must be between 0 and 1");
		}
		this.epsilon = epsilon;
	}
}
