import type { Agent } from "../core/agent.ts";
import type { Transition } from "../types/transition.ts";
import { EpsilonGreedyPolicy } from "../policies/epsilonGreedy.ts";
import { SeededRNG } from "../utils/random.ts";

/**
 * Configuration for QLearningAgent.
 */
export interface QLearningConfig {
	/** Learning rate (0 < alpha <= 1) */
	alpha: number;
	/** Discount factor (0 <= gamma <= 1) */
	gamma: number;
	/** Exploration rate (0 <= epsilon <= 1) */
	epsilon: number;
	/** Optional: RNG seed for reproducibility */
	seed?: number;
	/** Optional: Function to serialize state for Q-table key */
	stateSerializer?: (state: unknown) => string;
}

/**
 * Tabular Q-Learning Agent.
 *
 * Uses a Q-table (Map<string, number>) to store state-action values.
 * Supports any state type via optional state serializer.
 *
 * Update rule:
 * Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export class QLearningAgent<S, A> implements Agent<S, A> {
	private qTable: Map<string, Map<string, number>> = new Map();
	private policy: EpsilonGreedyPolicy<A>;
	private rng: SeededRNG;
	private stateSerializer: (state: S) => string;
	private actionSerializer: (action: A) => string;
	private actions: A[];
	private config: QLearningConfig;

	constructor(actions: A[], config: QLearningConfig) {
		this.actions = actions;
		this.config = config;
		if (config.alpha <= 0 || config.alpha > 1) {
			throw new Error("Alpha must be in (0, 1]");
		}
		if (config.gamma < 0 || config.gamma > 1) {
			throw new Error("Gamma must be in [0, 1]");
		}
		if (config.epsilon < 0 || config.epsilon > 1) {
			throw new Error("Epsilon must be in [0, 1]");
		}

		this.rng = new SeededRNG(config.seed ?? Date.now());

		// Use provided serializer or default to JSON
		this.stateSerializer = config.stateSerializer ?? ((s) => JSON.stringify(s));

		// Default action serializer
		this.actionSerializer = (a) => {
			if (typeof a === "string" || typeof a === "number") {
				return String(a);
			}
			return JSON.stringify(a);
		};

		// Initialize policy with epsilon-greedy exploration
		this.policy = new EpsilonGreedyPolicy(
			actions,
			() => this.greedyAction(this.currentState!),
			config.epsilon,
			() => this.rng.random(),
		);
	}

	private currentState: S | null = null;

	/**
	 * Initialize Q-values for a state if not already present.
	 */
	private initializeStateIfNeeded(state: S): void {
		const stateKey = this.stateSerializer(state);
		if (!this.qTable.has(stateKey)) {
			const actionValues = new Map<string, number>();
			for (const action of this.actions) {
				actionValues.set(this.actionSerializer(action), 0);
			}
			this.qTable.set(stateKey, actionValues);
		}
	}

	/**
	 * Get Q-value for state-action pair.
	 */
	private getQValue(state: S, action: A): number {
		const stateKey = this.stateSerializer(state);
		const actionKey = this.actionSerializer(action);
		this.initializeStateIfNeeded(state);

		const actionValues = this.qTable.get(stateKey)!;
		return actionValues.get(actionKey) ?? 0;
	}

	/**
	 * Set Q-value for state-action pair.
	 */
	private setQValue(state: S, action: A, value: number): void {
		const stateKey = this.stateSerializer(state);
		const actionKey = this.actionSerializer(action);
		this.initializeStateIfNeeded(state);

		const actionValues = this.qTable.get(stateKey)!;
		actionValues.set(actionKey, value);
	}

	/**
	 * Get the greedy action for a state (max Q-value).
	 */
	private greedyAction(state: S): A {
		this.initializeStateIfNeeded(state);
		const stateKey = this.stateSerializer(state);
		const actionValues = this.qTable.get(stateKey)!;

		let maxValue = Number.NEGATIVE_INFINITY;
		let bestAction = this.actions[0];

		for (const action of this.actions) {
			const actionKey = this.actionSerializer(action);
			const value = actionValues.get(actionKey) ?? 0;

			if (value > maxValue) {
				maxValue = value;
				bestAction = action;
			}
		}

		return bestAction;
	}

	/**
	 * Choose an action using epsilon-greedy policy.
	 */
	async act(state: S): Promise<A> {
		this.currentState = state;
		return this.policy.sample();
	}

	/**
	 * Learn from a transition using Q-learning update rule.
	 */
	async learn(transition: Transition<S, A>): Promise<void> {
		const { state, action, reward, nextState, done } = transition;

		const currentQ = this.getQValue(state, action);

		// Compute target: r + γ * max_a' Q(s', a')
		let target = reward;
		if (!done) {
			const maxNextQ = Math.max(
				...this.actions.map((a) => this.getQValue(nextState, a)),
			);
			target += this.config.gamma * maxNextQ;
		}

		// Q-learning update
		const newQ = currentQ + this.config.alpha * (target - currentQ);
		this.setQValue(state, action, newQ);
	}

	/**
	 * Save Q-table to JSON.
	 */
	async save(path: string): Promise<void> {
		const data = {
			config: this.config,
			qTable: Array.from(this.qTable.entries()).map(([stateKey, actions]) => [
				stateKey,
				Array.from(actions.entries()),
			]),
		};

		const fs = await import("node:fs/promises");
		await fs.writeFile(path, JSON.stringify(data, null, 2));
	}

	/**
	 * Load Q-table from JSON.
	 */
	async load(path: string): Promise<void> {
		const fs = await import("node:fs/promises");
		const content = await fs.readFile(path, "utf-8");
		const data = JSON.parse(content);

		this.qTable = new Map(
			data.qTable.map(
				([stateKey, actions]: [string, Array<[string, number]>]) => [
					stateKey,
					new Map(actions),
				],
			),
		);
	}

	/**
	 * Get current Q-table (for inspection/debugging).
	 */
	getQTable(): Record<string, Record<string, number>> {
		const result: Record<string, Record<string, number>> = {};
		for (const [stateKey, actionValues] of this.qTable.entries()) {
			result[stateKey] = Object.fromEntries(actionValues);
		}
		return result;
	}

	/**
	 * Get epsilon for inspection.
	 */
	getEpsilon(): number {
		return this.policy.getEpsilon();
	}

	/**
	 * Set epsilon (for decay schedules).
	 */
	setEpsilon(epsilon: number): void {
		this.policy.setEpsilon(epsilon);
	}
}
