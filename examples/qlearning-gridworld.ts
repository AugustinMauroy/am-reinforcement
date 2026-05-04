/**
 * Q-Learning example: Simple GridWorld
 *
 * A 2D grid where an agent learns to reach a goal.
 * State: (x, y) coordinates
 * Actions: up, down, left, right
 * Reward: +10 for reaching goal, -1 for each step
 */

import { QLearningAgent } from "../src/algorithms/qlearning.ts";
import { train, runEpisode } from "../src/core/trainer.ts";
import type { Environment, StepResult } from "../src/core/environment.ts";

interface Position {
	x: number;
	y: number;
}

type GridState = Position;

type GridAction = "up" | "down" | "left" | "right";

/**
 * Simple gridworld environment
 */
class GridWorldEnv implements Environment<GridState, GridAction> {
	private agent: Position;
	private readonly goal: Position = { x: 4, y: 4 };
	private readonly gridSize = 5;

	reset(): GridState {
		this.agent = { x: 0, y: 0 };
		return { ...this.agent };
	}

	step(action: GridAction): StepResult<GridState> {
		// Apply action
		switch (action) {
			case "up":
				this.agent.y = Math.max(0, this.agent.y - 1);
				break;
			case "down":
				this.agent.y = Math.min(this.gridSize - 1, this.agent.y + 1);
				break;
			case "left":
				this.agent.x = Math.max(0, this.agent.x - 1);
				break;
			case "right":
				this.agent.x = Math.min(this.gridSize - 1, this.agent.x + 1);
				break;
		}

		// Check if reached goal
		const reachedGoal =
			this.agent.x === this.goal.x && this.agent.y === this.goal.y;

		const reward = reachedGoal ? 10 : -0.1; // Small penalty per step to encourage efficiency

		return {
			state: { ...this.agent },
			reward,
			done: reachedGoal,
		};
	}
}

async function main() {
	console.log("🎮 Q-Learning on GridWorld\n");

	const env = new GridWorldEnv();

	const agent = new QLearningAgent<GridState, GridAction>(
		["up", "down", "left", "right"],
		{
			alpha: 0.1, // Learning rate
			gamma: 0.99, // Discount factor
			epsilon: 0.2, // Exploration rate
			seed: 42,
			// State serializer: convert {x, y} to string for Q-table key
			stateSerializer: (state) => `${state.x},${state.y}`,
		},
	);

	const episodeRewards: number[] = [];

	// Training phase
	console.log("📚 Training...");
	await train({
		env,
		agent,
		episodes: 200,
		maxStepsPerEpisode: 50,
		onEpisodeEnd: (episode, totalReward) => {
			episodeRewards.push(totalReward);

			if ((episode + 1) % 20 === 0) {
				const avgReward =
					episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
				console.log(
					`Episode ${episode + 1}/200 | Reward: ${totalReward.toFixed(2)} | Avg (last 20): ${avgReward.toFixed(2)} | Epsilon: ${agent.getEpsilon().toFixed(3)}`,
				);
			}
		},
	});

	console.log("\n✅ Training complete!");
	console.log(
		`Average reward (last 20): ${episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20}`,
	);

	// Decay epsilon for testing (greedy policy)
	agent.setEpsilon(0.01);

	// Test phase
	console.log("\n🧪 Testing (greedy policy)...");
	const testRewards: number[] = [];

	for (let i = 0; i < 10; i++) {
		const [reward, steps] = await runEpisode(env, agent, 50);
		testRewards.push(reward);
		console.log(
			`Test episode ${i + 1}: reward = ${reward.toFixed(2)}, steps = ${steps}`,
		);
	}

	const avgTestReward =
		testRewards.reduce((a, b) => a + b, 0) / testRewards.length;
	console.log(`\nAverage test reward: ${avgTestReward.toFixed(2)}`);

	// Inspect Q-table (sample)
	console.log("\n📊 Sample Q-values from Q-table:");
	const qTable = agent.getQTable();
	let count = 0;
	for (const [state, actions] of Object.entries(qTable)) {
		if (count < 5) {
			console.log(`  State ${state}:`, actions);
			count++;
		}
	}
}

main().catch(console.error);
