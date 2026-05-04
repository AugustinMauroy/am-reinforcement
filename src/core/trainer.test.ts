import { describe, it } from "node:test";
import assert from "node:assert";
import { train, runEpisode } from "./trainer.ts";
import type { Environment, StepResult } from "./environment.ts";
import type { Agent } from "./agent.ts";
import type { Transition } from "../types/transition.ts";

/**
 * Simple test environment
 */
class TestEnv implements Environment<number, number> {
	private position = 0;
	private maxSteps = 10;

	reset(): number {
		this.position = 0;
		return this.position;
	}

	step(action: number): StepResult<number> {
		this.position += action;
		const done = Math.abs(this.position) > this.maxSteps || this.position === 5;

		const reward = this.position === 5 ? 10 : -0.1;

		return {
			state: this.position,
			reward,
			done,
		};
	}
}

/**
 * Simple test agent
 */
class TestAgent implements Agent<number, number> {
	private qValues = new Map<number, number>();

	async act(state: number): Promise<number> {
		// Simple: go towards 5
		if (state < 5) return 1;
		if (state > 5) return -1;
		return 0;
	}

	async learn(transition: Transition<number, number>): Promise<void> {
		this.qValues.set(transition.state, transition.reward);
	}
}

describe("Trainer", () => {
	it("should throw on invalid episodes", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		assert.rejects(
			() =>
				train({
					env,
					agent,
					episodes: 0,
				}),
			(err: Error) => err.message.includes("positive"),
		);

		assert.rejects(
			() =>
				train({
					env,
					agent,
					episodes: -5,
				}),
			(err: Error) => err.message.includes("positive"),
		);
	});

	it("should throw on invalid maxStepsPerEpisode", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		assert.rejects(
			() =>
				train({
					env,
					agent,
					episodes: 10,
					maxStepsPerEpisode: 0,
				}),
			(err: Error) => err.message.includes("positive"),
		);

		assert.rejects(
			() =>
				train({
					env,
					agent,
					episodes: 10,
					maxStepsPerEpisode: -5,
				}),
			(err: Error) => err.message.includes("positive"),
		);
	});

	it("should complete training", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		let completed = false;

		await train({
			env,
			agent,
			episodes: 5,
			onTrainEnd: () => {
				completed = true;
			},
		});

		assert(completed);
	});

	it("should call onTrainStart", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		let called = false;

		await train({
			env,
			agent,
			episodes: 1,
			onTrainStart: () => {
				called = true;
			},
		});

		assert(called);
	});

	it("should call onEpisodeEnd with correct parameters", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const episodes: number[] = [];
		const rewards: number[] = [];
		const steps: number[] = [];

		await train({
			env,
			agent,
			episodes: 3,
			onEpisodeEnd: (ep, reward, step) => {
				episodes.push(ep);
				rewards.push(reward);
				steps.push(step);
			},
		});

		assert.strictEqual(episodes.length, 3);
		assert.deepStrictEqual(episodes, [0, 1, 2]);
		assert(rewards.length === 3);
		assert(steps.every((s) => s > 0));
	});

	it("should respect maxStepsPerEpisode", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const stepCounts: number[] = [];

		await train({
			env,
			agent,
			episodes: 5,
			maxStepsPerEpisode: 3,
			onEpisodeEnd: (_ep, _reward, steps) => {
				stepCounts.push(steps);
			},
		});

		// All episodes should have <= 3 steps
		assert(stepCounts.every((s) => s <= 3));
	});

	it("should call onStep callback", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const transitions: Transition<number, number>[] = [];

		await train({
			env,
			agent,
			episodes: 2,
			maxStepsPerEpisode: 5,
			onStep: (_ep, _step, transition) => {
				transitions.push(transition);
			},
		});

		assert(transitions.length > 0);
		// Each transition should have required fields
		transitions.forEach((t) => {
			assert(typeof t.state === "number");
			assert(typeof t.action === "number");
			assert(typeof t.reward === "number");
			assert(typeof t.nextState === "number");
			assert(typeof t.done === "boolean");
		});
	});

	it("should handle episode termination", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const doneFlags: boolean[] = [];

		await train({
			env,
			agent,
			episodes: 5,
			onStep: (_ep, _step, transition) => {
				doneFlags.push(transition.done);
			},
		});

		// Should have at least one done transition
		assert(doneFlags.some((d) => d === true));
	});

	it("should handle max steps stopping", async () => {
		const env = new TestEnv();

		class NonTerminalAgent implements Agent<number, number> {
			async act(): Promise<number> {
				return 0; // No progress
			}

			async learn(): Promise<void> {}
		}

		const agent = new NonTerminalAgent();
		let stepCount = 0;

		await train({
			env,
			agent,
			episodes: 1,
			maxStepsPerEpisode: 5,
			onStep: () => {
				stepCount++;
			},
		});

		assert.strictEqual(stepCount, 5);
	});

	it("should accumulate rewards correctly", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const episodeRewards: number[] = [];

		await train({
			env,
			agent,
			episodes: 3,
			maxStepsPerEpisode: 20,
			onEpisodeEnd: (_ep, reward) => {
				episodeRewards.push(reward);
			},
		});

		assert.strictEqual(episodeRewards.length, 3);
		// Each episode should have accumulated rewards
		episodeRewards.forEach((reward) => {
			assert(typeof reward === "number");
		});
	});

	it("should work with async environment", async () => {
		class AsyncTestEnv implements Environment<number, number> {
			private position = 0;

			async reset(): Promise<number> {
				this.position = 0;
				return this.position;
			}

			async step(action: number): Promise<StepResult<number>> {
				this.position += action;
				const done = this.position === 5;

				return {
					state: this.position,
					reward: done ? 10 : -0.1,
					done,
				};
			}
		}

		const env = new AsyncTestEnv();
		const agent = new TestAgent();

		let completed = false;

		await train({
			env,
			agent,
			episodes: 2,
			maxStepsPerEpisode: 20,
			onTrainEnd: () => {
				completed = true;
			},
		});

		assert(completed);
	});

	it("should work with async agent", async () => {
		const env = new TestEnv();

		class AsyncTestAgent implements Agent<number, number> {
			async act(state: number): Promise<number> {
				return state < 5 ? 1 : -1;
			}

			async learn(): Promise<void> {
				// Async learning
			}
		}

		const agent = new AsyncTestAgent();

		let completed = false;

		await train({
			env,
			agent,
			episodes: 2,
			maxStepsPerEpisode: 20,
			onTrainEnd: () => {
				completed = true;
			},
		});

		assert(completed);
	});
});

describe("runEpisode", () => {
	it("should return tuple of [reward, steps]", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const [reward, steps] = await runEpisode(env, agent);

		assert(typeof reward === "number");
		assert(typeof steps === "number");
		assert(steps > 0);
	});

	it("should respect maxSteps", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const [, steps] = await runEpisode(env, agent, 5);

		assert(steps <= 5);
	});

	it("should stop exactly at maxSteps when the episode never ends", async () => {
		class EndlessEnv implements Environment<number, number> {
			reset(): number {
				return 0;
			}

			step(): StepResult<number> {
				return {
					state: 0,
					reward: 1,
					done: false,
				};
			}
		}

		const env = new EndlessEnv();
		const agent = new TestAgent();

		const [reward, steps] = await runEpisode(env, agent, 3);

		assert.strictEqual(steps, 3);
		assert.strictEqual(reward, 3);
	});

	it("should terminate early if done", async () => {
		const env = new TestEnv();
		const agent = new TestAgent();

		const [, steps] = await runEpisode(env, agent, 100);

		// Should terminate early because agent reaches state 5
		assert(steps < 100);
	});

	it("should not learn during episode", async () => {
		const env = new TestEnv();

		class TrackingAgent implements Agent<number, number> {
			learnCalls = 0;

			async act(state: number): Promise<number> {
				return state < 5 ? 1 : -1;
			}

			async learn(): Promise<void> {
				this.learnCalls++;
			}
		}

		const agent = new TrackingAgent();

		await runEpisode(env, agent, 20);

		// Should not call learn
		assert.strictEqual(agent.learnCalls, 0);
	});

	it("should work with async environment", async () => {
		class AsyncTestEnv implements Environment<number, number> {
			private position = 0;

			async reset(): Promise<number> {
				this.position = 0;
				return this.position;
			}

			async step(action: number): Promise<StepResult<number>> {
				this.position += action;
				return {
					state: this.position,
					reward: this.position === 5 ? 10 : -0.1,
					done: this.position === 5,
				};
			}
		}

		const env = new AsyncTestEnv();
		const agent = new TestAgent();

		const [reward, steps] = await runEpisode(env, agent);

		assert(typeof reward === "number");
		assert(steps > 0);
	});
});
