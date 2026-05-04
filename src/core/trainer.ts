import type { Environment, StepResult } from "./environment.ts";
import type { Agent } from "./agent.ts";
import type { Transition } from "../types/transition.ts";

/**
 * Configuration for training.
 */
export interface TrainOptions<S, A> {
	/** The environment to train in */
	env: Environment<S, A>;

	/** The agent to train */
	agent: Agent<S, A>;

	/** Number of episodes to train for */
	episodes: number;

	/** Optional: Maximum steps per episode */
	maxStepsPerEpisode?: number;

	/** Optional: Callback at the end of each episode */
	onEpisodeEnd?: (episode: number, totalReward: number, steps: number) => void;

	/** Optional: Callback at each step */
	onStep?: (
		episode: number,
		step: number,
		transition: Transition<S, A>,
	) => void;

	/** Optional: Callback when training starts */
	onTrainStart?: () => void;

	/** Optional: Callback when training completes */
	onTrainEnd?: () => void;
}

/**
 * Standard training loop for RL agents.
 *
 * This orchestrates the main RL loop:
 * 1. Reset environment
 * 2. For each step:
 *    a. Agent selects action
 *    b. Environment transitions
 *    c. Agent learns
 * 3. Repeat for N episodes
 *
 * Supports both sync and async environments/agents.
 */
export async function train<S, A>(options: TrainOptions<S, A>): Promise<void> {
	const {
		env,
		agent,
		episodes,
		maxStepsPerEpisode,
		onEpisodeEnd,
		onStep,
		onTrainStart,
		onTrainEnd,
	} = options;

	if (episodes <= 0) {
		throw new Error("Episodes must be positive");
	}

	if (maxStepsPerEpisode !== undefined && maxStepsPerEpisode <= 0) {
		throw new Error("Max steps per episode must be positive");
	}

	onTrainStart?.();

	for (let episode = 0; episode < episodes; episode++) {
		const resetResult = env.reset();
		const state0 = await Promise.resolve(resetResult);
		let state = state0 as S;
		let totalReward = 0;
		let stepCount = 0;

		let done = false;
		while (!done) {
			if (maxStepsPerEpisode !== undefined && stepCount >= maxStepsPerEpisode) {
				break;
			}

			// Agent selects action
			const action = await Promise.resolve(agent.act(state));

			// Environment transitions
			const stepResult = env.step(action);
			const result0 = await Promise.resolve(stepResult);
			const result = result0 as StepResult<S>;

			// Create transition
			const transition: Transition<S, A> = {
				state,
				action,
				reward: result.reward,
				nextState: result.state,
				done: result.done,
				info: result.info,
			};

			// Agent learns
			await Promise.resolve(agent.learn(transition));

			// Optional callbacks
			onStep?.(episode, stepCount, transition);

			// Update state
			state = result.state;
			totalReward += result.reward;
			done = result.done;
			stepCount++;
		}

		// Episode complete
		onEpisodeEnd?.(episode, totalReward, stepCount);
	}

	onTrainEnd?.();
}

/**
 * Run inference (no learning) for a single episode.
 *
 * @returns Tuple of (total_reward, num_steps)
 */
export async function runEpisode<S, A>(
	env: Environment<S, A>,
	agent: Agent<S, A>,
	maxSteps?: number,
): Promise<[number, number]> {
	const resetResult = env.reset();
	const state0 = await Promise.resolve(resetResult);
	let state = state0 as S;
	let totalReward = 0;
	let stepCount = 0;

	let done = false;
	while (!done) {
		if (maxSteps !== undefined && stepCount >= maxSteps) {
			break;
		}

		const action = await Promise.resolve(agent.act(state));
		const stepResult = env.step(action);
		const result0 = await Promise.resolve(stepResult);
		const result = result0 as StepResult<S>;

		state = result.state;
		totalReward += result.reward;
		done = result.done;
		stepCount++;
	}

	return [totalReward, stepCount];
}
