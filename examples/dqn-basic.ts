/**
 * Basic DQN example: CartPole-like environment
 *
 * A simple 1D environment where an agent learns to balance a value around 0.
 * State: current position (number)
 * Actions: move left (-1), stay (0), move right (1)
 * Reward: 1 if close to 0, -1 for each step away
 */

import { DQNAgent } from "../src/algorithms/dqn.ts";
import { ReplayBuffer } from "../src/memory/replayBuffer.ts";
import { train } from "../src/core/trainer.ts";
import type { Environment, StepResult } from "../src/core/environment.ts";
import type { NeuralNetworkModel } from "../src/algorithms/dqn.ts";

/**
 * Simple mock neural network for demonstration.
 * In practice, this would be replaced with @am/neuralnetwork.
 */
class SimpleNeuralNetwork implements NeuralNetworkModel {
	private weights: number[][][]; // [layers][neurons][input_weights]

	constructor(inputSize: number, hiddenSize: number, outputSize: number) {
		// Initialize random weights
		this.weights = [
			this.randomMatrix(hiddenSize, inputSize),
			this.randomMatrix(outputSize, hiddenSize),
		];
	}

	private randomMatrix(rows: number, cols: number): number[][] {
		const matrix: number[][] = [];
		for (let i = 0; i < rows; i++) {
			const row: number[] = [];
			for (let j = 0; j < cols; j++) {
				row.push((Math.random() - 0.5) * 2); // [-1, 1]
			}
			matrix.push(row);
		}
		return matrix;
	}

	private relu(x: number): number {
		return Math.max(0, x);
	}

	predict(state: unknown): number[] {
		const input = state as number[];

		// First layer: input -> hidden (with ReLU)
		const hidden: number[] = [];
		for (let i = 0; i < this.weights[0].length; i++) {
			let sum = 0;
			for (let j = 0; j < input.length; j++) {
				sum += this.weights[0][i][j] * input[j];
			}
			hidden.push(this.relu(sum));
		}

		// Output layer: hidden -> output (linear)
		const output: number[] = [];
		for (let i = 0; i < this.weights[1].length; i++) {
			let sum = 0;
			for (let j = 0; j < hidden.length; j++) {
				sum += this.weights[1][i][j] * hidden[j];
			}
			output.push(sum);
		}

		return output;
	}

	train(states: unknown[], targets: number[][]): void {
		// Simplified gradient descent update
		// In practice, this would use proper backpropagation
		const learningRate = 0.01;

		for (let s = 0; s < states.length; s++) {
			const input = states[s] as number[];
			const target = targets[s];
			const prediction = this.predict(input);

			// Compute output layer gradients
			const outputGradients: number[] = [];
			for (let i = 0; i < prediction.length; i++) {
				outputGradients.push((prediction[i] - target[i]) / states.length);
			}

			// Simple weight update (not real backprop, just demo)
			for (let i = 0; i < this.weights[1].length; i++) {
				for (let j = 0; j < this.weights[1][i].length; j++) {
					this.weights[1][i][j] -= learningRate * outputGradients[i];
				}
			}
		}
	}

	clone(): SimpleNeuralNetwork {
		const copy = new SimpleNeuralNetwork(
			this.weights[0][0].length,
			this.weights[0].length,
			this.weights[1].length,
		);
		copy.weights = this.weights.map((layer) =>
			layer.map((neuron) => [...neuron]),
		);
		return copy;
	}

	serialize(): string {
		return JSON.stringify(this.weights);
	}

	deserialize(data: string): void {
		this.weights = JSON.parse(data);
	}
}

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

		// Episode done if too far from center
		const done = Math.abs(this.position) > 2;

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
	const model = new SimpleNeuralNetwork(
		1, // input: 1D position
		16, // hidden layer: 16 neurons
		3, // output: 3 actions
	);

	const agent = new DQNAgent(
		model,
		new ReplayBuffer(1000),
		[0, 1, 2], // 3 actions: left, stay, right
		{
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 32,
			targetUpdateFrequency: 100,
		},
	);

	const rewards: number[] = [];

	await train({
		env,
		agent,
		episodes: 100,
		maxStepsPerEpisode: 100,
		onEpisodeEnd: (episode, totalReward) => {
			rewards.push(totalReward);

			if ((episode + 1) % 10 === 0) {
				const avgReward = rewards.slice(-10).reduce((a, b) => a + b, 0) / 10;
				console.log(
					`Episode ${episode + 1}/100 | Reward: ${totalReward.toFixed(2)} | Avg (last 10): ${avgReward.toFixed(2)} | Epsilon: ${agent.getEpsilon().toFixed(3)}`,
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
