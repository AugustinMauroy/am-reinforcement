import type { Agent } from "../core/agent.ts";
import type { Transition } from "../types/transition.ts";
import type { IReplayBuffer } from "../memory/replayBuffer.ts";
import { EpsilonGreedyPolicy } from "../policies/epsilonGreedy.ts";
import { SeededRNG } from "../utils/random.ts";
import { Model } from "@am/neuralnetwork";

/**
 * Wrapper interface for neural network models used by DQN.
 * Adapts @am/neuralnetwork.Model to DQN's training interface.
 */
export interface NeuralNetworkModel {
	/**
	 * Forward pass: predict Q-values for actions given states.
	 *
	 * @param states - Batch of input states
	 * @returns 2D array of Q-values (batch_size x num_actions)
	 */
	predict(states: number[][]): number[][];

	/**
	 * Train on a batch of states and targets (single epoch).
	 *
	 * @param states - Batch of input states
	 * @param targets - Batch of target Q-values
	 */
	train(states: number[][], targets: number[][]): Promise<void>;

	/**
	 * Create a deep copy of this model (for target network).
	 */
	clone(): NeuralNetworkModel;

	/**
	 * Serialize model to JSON.
	 */
	serialize(): string;

	/**
	 * Deserialize model from JSON.
	 */
	deserialize(data: string): void;
}

/**
 * Configuration for DQNAgent.
 */
export interface DQNConfig {
	/** Discount factor (0 <= gamma <= 1) */
	gamma: number;

	/** Exploration rate (0 <= epsilon <= 1) */
	epsilon: number;

	/** Batch size for training */
	batchSize: number;

	/** Frequency to update target network (in steps) */
	targetUpdateFrequency: number;

	/** Optional: RNG seed for reproducibility */
	seed?: number;

	/** Optional: State serializer for consistent input format */
	stateSerializer?: (state: unknown) => number[];

	/** Optional: Action list for epsilon-greedy policy */
	actions?: unknown[];
}

/**
 * Creates a DQN-compatible neural network wrapper from @am/neuralnetwork.Model.
 *
 * Example usage:
 * ```typescript
 * import { Model, Dense, ReLU, MeanSquaredError, Adam } from "@am/neuralnetwork";
 *
 * const model = new Model();
 * model.addLayer(new Dense(4, 128));
 * model.addLayer(new ReLU());
 * model.addLayer(new Dense(128, 2)); // 2 actions
 * model.compile(new Adam(0.001), new MeanSquaredError(), []);
 *
 * const qNetwork = createDQNModelAdapter(model);
 * ```
 */
export function createDQNModelAdapter(model: Model): NeuralNetworkModel {
	return {
		predict(states: number[][]): number[][] {
			return model.predict(states);
		},

		async train(states: number[][], targets: number[][]): Promise<void> {
			// Train for 1 epoch with batch size equal to sample size
			await model.fit(states, targets, 1, states.length);
		},

		clone(): NeuralNetworkModel {
			const newModel = Model.load(model.save());
			return createDQNModelAdapter(newModel);
		},

		serialize(): string {
			return model.save();
		},

		deserialize(data: string): void {
			const loadedModel = Model.load(data);
			// Copy state from loaded model to current model
			// Note: This requires the loaded model to replace the current one's internals
			Object.assign(model, loadedModel);
		},
	};
}

/**
 * Deep Q-Network (DQN) Agent.
 *
 * Combines:
 * - Neural network for function approximation
 * - Replay buffer for experience replay
 * - Target network for stability
 * - Epsilon-greedy exploration
 *
 * Update rule:
 * y = r + γ * max_a' Q_target(s', a')
 * loss = (Q(s,a) - y)^2
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export class DQNAgent<S, A> implements Agent<S, A> {
	private qNetwork: NeuralNetworkModel;
	private targetNetwork: NeuralNetworkModel;
	private policy: EpsilonGreedyPolicy<A>;
	private rng: SeededRNG;
	private stepCount = 0;
	private stateSerializer: (state: S) => number[];
	private replayBuffer: IReplayBuffer<S, A>;
	private actions: A[];
	private config: DQNConfig;

	constructor(
		qNetwork: NeuralNetworkModel,
		replayBuffer: IReplayBuffer<S, A>,
		actions: A[],
		config: DQNConfig,
	) {
		this.replayBuffer = replayBuffer;
		this.actions = actions;
		this.config = config;
		if (config.gamma < 0 || config.gamma > 1) {
			throw new Error("Gamma must be in [0, 1]");
		}
		if (config.epsilon < 0 || config.epsilon > 1) {
			throw new Error("Epsilon must be in [0, 1]");
		}
		if (config.batchSize <= 0) {
			throw new Error("Batch size must be positive");
		}
		if (config.targetUpdateFrequency <= 0) {
			throw new Error("Target update frequency must be positive");
		}
		if (actions.length === 0) {
			throw new Error("Must provide at least one action");
		}

		this.qNetwork = qNetwork;
		this.targetNetwork = qNetwork.clone();
		this.rng = new SeededRNG(config.seed ?? Date.now());

		// State serializer: convert state to numeric vector
		this.stateSerializer =
			config.stateSerializer ??
			((s) => {
				if (Array.isArray(s)) {
					return s as number[];
				}
				if (typeof s === "number") {
					return [s];
				}
				throw new Error("Unable to serialize state - provide stateSerializer");
			});

		// Initialize epsilon-greedy policy with sync greedy selection
		// Note: actual greedy action computation happens in act()
		this.policy = new EpsilonGreedyPolicy(
			actions,
			() => actions[0], // Placeholder - will be overridden in act()
			config.epsilon,
			() => this.rng.random(),
		);
	}

	/**
	 * Get Q-values for a single state using the online network.
	 */
	private async getQValues(state: S): Promise<number[]> {
		const stateVector = this.stateSerializer(state);
		const batch = [stateVector];
		const qValuesBatch = this.qNetwork.predict(batch);
		return qValuesBatch[0];
	}

	/**
	 * Get Q-values for a single state using the target network.
	 */
	private async getTargetQValues(state: S): Promise<number[]> {
		const stateVector = this.stateSerializer(state);
		const batch = [stateVector];
		const qValuesBatch = this.targetNetwork.predict(batch);
		return qValuesBatch[0];
	}

	/**
	 * Get the greedy action for a state.
	 */
	private async greedyAction(state: S): Promise<A> {
		const qValues = await this.getQValues(state);
		const maxIdx = qValues.indexOf(Math.max(...qValues));
		return this.actions[maxIdx];
	}

	/**
	 * Choose an action using epsilon-greedy policy.
	 */
	async act(state: S): Promise<A> {
		// Decide explore vs exploit
		if (this.rng.random() < this.config.epsilon) {
			// Explore: random action
			const randomIdx = Math.floor(this.rng.random() * this.actions.length);
			return this.actions[randomIdx];
		}

		// Exploit: greedy action based on Q-values
		return this.greedyAction(state);
	}

	/**
	 * Learn from a transition:
	 * 1. Store in replay buffer
	 * 2. Sample batch and train if buffer is ready
	 * 3. Update target network periodically
	 */
	async learn(transition: Transition<S, A>): Promise<void> {
		this.replayBuffer.add(transition);
		this.stepCount++;

		// Only train after buffer has enough samples
		if (this.replayBuffer.size() < this.config.batchSize) {
			return;
		}

		// Sample batch from replay buffer
		const batch = this.replayBuffer.sample(this.config.batchSize);

		// Prepare batch data
		const states: number[][] = [];
		const targets: number[][] = [];

		for (const t of batch) {
			const stateVector = this.stateSerializer(t.state);
			states.push(stateVector);

			// Current Q-values for this state
			const currentQValues = await this.getQValues(t.state);
			const target = [...currentQValues];

			// Find action index
			const actionIdx = this.actions.findIndex(
				(a) => JSON.stringify(a) === JSON.stringify(t.action),
			);

			if (actionIdx === -1) {
				throw new Error("Action not found in action space");
			}

			// Compute target using target network
			let targetValue = t.reward;
			if (!t.done) {
				const nextQValues = await this.getTargetQValues(t.nextState);
				const maxNextQ = Math.max(...nextQValues);
				targetValue += this.config.gamma * maxNextQ;
			}

			target[actionIdx] = targetValue;
			targets.push(target);
		}

		// Train the online network
		await this.qNetwork.train(states, targets);

		// Update target network periodically
		if (this.stepCount % this.config.targetUpdateFrequency === 0) {
			this.targetNetwork = this.qNetwork.clone();
		}
	}

	/**
	 * Save agent state.
	 */
	async save(path: string): Promise<void> {
		const fs = await import("node:fs/promises");
		const data = {
			config: this.config,
			qNetworkWeights: this.qNetwork.serialize(),
			stepCount: this.stepCount,
		};
		await fs.writeFile(path, JSON.stringify(data, null, 2));
	}

	/**
	 * Load agent state.
	 */
	async load(path: string): Promise<void> {
		const fs = await import("node:fs/promises");
		const content = await fs.readFile(path, "utf-8");
		const data = JSON.parse(content);

		this.qNetwork.deserialize(data.qNetworkWeights);
		this.targetNetwork = this.qNetwork.clone();
		this.stepCount = data.stepCount ?? 0;
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

	/**
	 * Get step count (total training steps).
	 */
	getStepCount(): number {
		return this.stepCount;
	}
}
