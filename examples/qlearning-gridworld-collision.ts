/**
 * Q-Learning example: Simple GridWorld
 *
 * A 2D grid where an agent learns to reach a goal while avoiding obstacles.
 * State: (x, y) coordinates
 * Actions: up, down, left, right
 * Reward: +10 for reaching goal, -1 for each step, -1 for hitting obstacle
 * This example demonstrates how to implement a Q-Learning agent in a gridworld environment with obstacles.
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


class GridWorldWithObstacles implements Environment<GridState, GridAction> {
	private agent: Position;

	private readonly gridSize = 10;

	private readonly goal: Position = { x: 9, y: 9 };

	// Define obstacles
	private readonly obstacles: Set<string> = new Set([
		"3,3",
		"3,4",
		"3,5",
		"4,5",
		"5,5",
		"6,5",
		"7,5",
		"7,4",
		"7,3",
	]); // simple wall with a gap

	private serialize(pos: Position): string {
		return `${pos.x},${pos.y}`;
	}

	private isObstacle(pos: Position): boolean {
		return this.obstacles.has(this.serialize(pos));
	}

	reset(): GridState {
		this.agent = { x: 0, y: 0 };
		return { ...this.agent };
	}

	step(action: GridAction): StepResult<GridState> {
		const next = { ...this.agent };

		// Attempt move
		switch (action) {
			case "up":
				next.y -= 1;
				break;
			case "down":
				next.y += 1;
				break;
			case "left":
				next.x -= 1;
				break;
			case "right":
				next.x += 1;
				break;
		}

		// Clamp to grid
		next.x = Math.max(0, Math.min(this.gridSize - 1, next.x));
		next.y = Math.max(0, Math.min(this.gridSize - 1, next.y));

		let reward = -0.1; // base step cost

		// Handle obstacle collision
		if (this.isObstacle(next)) {
			// Stay in place
			next.x = this.agent.x;
			next.y = this.agent.y;

			reward -= 1; // penalty for hitting obstacle
		}

		// Move agent
		this.agent = next;

		// Check goal
		const reachedGoal =
			this.agent.x === this.goal.x && this.agent.y === this.goal.y;

		// Distance shaping
		const distanceToGoal =
			Math.abs(this.agent.x - this.goal.x) +
			Math.abs(this.agent.y - this.goal.y);

		if (reachedGoal) {
			reward = 10;
		} else {
			reward -= 0.01 * distanceToGoal;
		}

		return {
			state: { ...this.agent },
			reward,
			done: reachedGoal,
		};
	}
}


async function main() {
	console.log("🎮 Q-Learning on GridWorld\n");

	const env = new GridWorldWithObstacles();

	const agent = new QLearningAgent<GridState, GridAction>(
		["up", "down", "left", "right"],
		{
			alpha: 0.1, // Learning rate
			gamma: 0.99, // Discount factor
			epsilon: 1.0, // Start with full exploration
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
		episodes: 500, // Increased episodes for better learning
		maxStepsPerEpisode: 200, // Increased steps per episode
		onEpisodeEnd: (episode, totalReward) => {
			episodeRewards.push(totalReward);

			// Decay epsilon
			agent.setEpsilon(Math.max(0.01, agent.getEpsilon() * 0.995));

			if ((episode + 1) % 20 === 0) {
				const avgReward =
					episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
				console.log(
					`Episode ${episode + 1}/500 | Reward: ${totalReward.toFixed(2)} | Avg (last 20): ${avgReward.toFixed(2)} | Epsilon: ${agent.getEpsilon().toFixed(3)}`,
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
		const [reward, steps] = await runEpisode(env, agent, 200);
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
