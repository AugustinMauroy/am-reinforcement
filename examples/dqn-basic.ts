/**
 * Basic DQN example: CartPole-like environment
 *
 * A simple 1D environment where an agent learns to balance a value around 0.
 * State: current position (number)
 * Actions: move left (-1), stay (0), move right (1)
 * Reward: 1 if close to 0, -1 for each step away
 */

import { DQNAgent, createDQNModelAdapter } from "../src/algorithms/dqn.ts";
import { ReplayBuffer } from "../src/memory/replayBuffer.ts";
import { train } from "../src/core/trainer.ts";
import type { Environment, StepResult } from "../src/core/environment.ts";
import { Model } from "@am/neuralnetwork";
import { Dense, ReLU } from "@am/neuralnetwork/layers";
import { MeanSquaredError } from "@am/neuralnetwork/losses";
import { Adam } from "@am/neuralnetwork/optimizes";

/**
 * Simple 1D balance environment
 */
class BalanceEnv implements Environment<number, number> {
	private position = 0;

	reset(): number {
		this.position = (Math.random() - 0.5) * 2; // Random position in [-1, 1]
		return this.position;
	}

	step(action: number): StepResult<number> {
		// Apply action
		this.position += (action - 1) * 0.1; // Actions: 0 (left), 1 (stay), 2 (right)

		// Clamp to [-2, 2]
		this.position = Math.max(-2, Math.min(2, this.position));

		// Reward: positive if close to 0, negative otherwise
		const distance = Math.abs(this.position);
		const reward = distance < 0.5 ? 1 : -distance * 0.5;

		// Episode done when the position reaches the boundary
		const done = Math.abs(this.position) >= 2;

		return {
			state: this.position,
			reward,
			done,
		};
	}
}

// Main training loop
async function main() {
	console.log("🤖 DQN Agent Training on Balance Environment\n");

	const env = new BalanceEnv();
	const model = new Model();
	model.addLayer(new Dense(1, 16));
	model.addLayer(new ReLU());
	model.addLayer(new Dense(16, 3));
	model.compile(new Adam(0.001), new MeanSquaredError(), []);

	const qNetwork = createDQNModelAdapter(model);

	const agent = new DQNAgent<number, number>(
		qNetwork,
		new ReplayBuffer<number, number>(1000),
		[0, 1, 2], // 3 actions: left, stay, right
		{
			gamma: 0.99,
			epsilon: 1,
			batchSize: 32,
			targetUpdateFrequency: 100,
			seed: 42,
		},
	);

	const rewards: number[] = [];

	await train({
		env,
		agent,
		episodes: 200,
		maxStepsPerEpisode: 200,
		onEpisodeEnd: (episode, totalReward) => {
			rewards.push(totalReward);

			if ((episode + 1) % 10 === 0) {
				const avgReward = rewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
				console.log(
					`Episode ${episode + 1}/200 | Reward: ${totalReward.toFixed(2)} | Avg (last 10): ${avgReward.toFixed(2)} | Epsilon: ${agent.getEpsilon().toFixed(3)}`,
				);
			}
		},
	});

	console.log("\n✅ Training complete!");
	console.log(
		`Average reward (last 10): ${rewards.slice(-10).reduce((a, b) => a + b, 0) / 10}`,
	);

	// Test inference
	console.log("\n🎮 Testing inference (no learning)...");
	const testEnv = new BalanceEnv();
	let state = testEnv.reset();
	let testReward = 0;
	let testSteps = 0;

	for (let i = 0; i < 50; i++) {
		const action = await agent.act(state);
		const result = testEnv.step(action);
		state = result.state;
		testReward += result.reward;
		testSteps++;

		if (result.done) break;
	}

	console.log(
		`Test episode reward: ${testReward.toFixed(2)}, steps: ${testSteps}`,
	);
}

main().catch(console.error);
