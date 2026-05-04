import { describe, it } from "node:test";
import assert from "node:assert";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DQNAgent, type NeuralNetworkModel } from "./dqn.ts";
import { ReplayBuffer } from "../memory/replayBuffer.ts";
import type { Transition } from "../types/transition.ts";

class MockNN implements NeuralNetworkModel {
	private weights: number[] = [0.1, 0.2, 0.3];

	getWeights(): number[] {
		return [...this.weights];
	}

	predict(): number[] {
		return [1.0, 2.0, 3.0];
	}

	async train(): Promise<void> {
		// Mock training
	}

	clone(): MockNN {
		const copy = new MockNN();
		copy.weights = [...this.weights];
		return copy;
	}

	serialize(): string {
		return JSON.stringify(this.weights);
	}

	deserialize(data: string): void {
		this.weights = JSON.parse(data);
	}
}

describe("DQNAgent", () => {
	it("should throw on invalid config", () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		assert.throws(() => {
			new DQNAgent(nn, buffer, actions, {
				gamma: -0.1,
				epsilon: 0.1,
				batchSize: 32,
				targetUpdateFrequency: 100,
			});
		});

		assert.throws(() => {
			new DQNAgent(nn, buffer, actions, {
				gamma: 0.99,
				epsilon: 1.1,
				batchSize: 32,
				targetUpdateFrequency: 100,
			});
		});

		assert.throws(() => {
			new DQNAgent(nn, buffer, actions, {
				gamma: 0.99,
				epsilon: 0.1,
				batchSize: 0,
				targetUpdateFrequency: 100,
			});
		});

		assert.throws(() => {
			new DQNAgent(nn, buffer, actions, {
				gamma: 0.99,
				epsilon: 0.1,
				batchSize: 32,
				targetUpdateFrequency: 0,
			});
		});

		assert.throws(() => {
			new DQNAgent(nn, buffer, [], {
				gamma: 0.99,
				epsilon: 0.1,
				batchSize: 32,
				targetUpdateFrequency: 100,
			});
		});
	});

	it("should initialize with config values", () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1, 2];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 32,
			targetUpdateFrequency: 100,
		});

		assert.strictEqual(agent.getEpsilon(), 0.1);
		assert.strictEqual(agent.getStepCount(), 0);
	});

	it("should act and return valid action", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1, 2];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.5,
			batchSize: 32,
			targetUpdateFrequency: 100,
			seed: 42,
		});

		for (let i = 0; i < 10; i++) {
			const action = await agent.act(1);
			assert(actions.includes(action));
		}
	});

	it("should add transition to replay buffer on learn", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 32,
			targetUpdateFrequency: 100,
		});

		const transition: Transition<number, number> = {
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: false,
		};

		assert.strictEqual(buffer.size(), 0);

		await agent.learn(transition);

		assert.strictEqual(buffer.size(), 1);
	});

	it("should not train before buffer is ready", async () => {
		const trainedStates: unknown[][] = [];

		const nn: NeuralNetworkModel = {
			predict: () => [1, 2],
			train: (states: unknown[]) => {
				trainedStates.push(...(states as Array<unknown[]>));
			},
			clone: function () {
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 32,
			targetUpdateFrequency: 100,
		});

		// Learn one transition
		await agent.learn({
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: false,
		});

		// Should not have trained yet (buffer < batch size)
		assert.strictEqual(trainedStates.length, 0);
	});

	it("should train when buffer has enough samples", async () => {
		let trainCallCount = 0;

		const nn: NeuralNetworkModel = {
			predict: () => [1, 2],
			train: () => {
				trainCallCount++;
			},
			clone: function () {
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<number, number>(100);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
		});

		// Learn enough transitions
		for (let i = 0; i < 5; i++) {
			await agent.learn({
				state: i,
				action: i % 2,
				reward: i,
				nextState: i + 1,
				done: i === 4,
			});
		}

		// Should have trained
		assert(trainCallCount > 0);
	});

	it("should get and set epsilon", () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.5,
			batchSize: 32,
			targetUpdateFrequency: 100,
		});

		assert.strictEqual(agent.getEpsilon(), 0.5);

		agent.setEpsilon(0.2);
		assert.strictEqual(agent.getEpsilon(), 0.2);

		agent.setEpsilon(0.8);
		assert.strictEqual(agent.getEpsilon(), 0.8);
	});

	it("should track step count", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
		});

		assert.strictEqual(agent.getStepCount(), 0);

		// Add transitions
		for (let i = 0; i < 5; i++) {
			await agent.learn({
				state: i,
				action: 0,
				reward: 1,
				nextState: i + 1,
				done: false,
			});
		}

		assert.strictEqual(agent.getStepCount(), 5);
	});

	it("should work with custom state serializer", async () => {
		interface ComplexState {
			x: number;
			y: number;
		}

		const nn: NeuralNetworkModel = {
			predict: () => [1, 2],
			train: () => {},
			clone: function () {
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<ComplexState, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
			stateSerializer: (s: unknown) => {
				const state = s as ComplexState;
				return [state.x, state.y];
			},
		});

		const state: ComplexState = { x: 1, y: 2 };
		const action = await agent.act(state);

		assert(actions.includes(action));
	});

	it("should serialize array states and update the target network on schedule", async () => {
		let cloneCalls = 0;

		const nn: NeuralNetworkModel = {
			predict: (stateVector: number[]) => {
				if (stateVector[0] === 1) {
					return [4, 1];
				}

				return [2, 6];
			},
			train: async () => {},
			clone: function () {
				cloneCalls++;
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<number[], number>(10);
		const agent = new DQNAgent(nn, buffer, [0, 1], {
			gamma: 0.5,
			epsilon: 0,
			batchSize: 1,
			targetUpdateFrequency: 1,
		});

		const action = await agent.act([1, 2]);
		assert.strictEqual(action, 0);

		await agent.learn({
			state: [1, 2],
			action: 0,
			reward: 3,
			nextState: [2, 3],
			done: false,
		});

		assert.strictEqual(cloneCalls, 2);
	});

	it("should exercise the internal policy callbacks", () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);

		const agent = new DQNAgent(nn, buffer, [0, 1], {
			gamma: 0.99,
			epsilon: 0,
			batchSize: 4,
			targetUpdateFrequency: 100,
			seed: 42,
		});

		const policy = (agent as unknown as { policy: { sample: () => number } })
			.policy;
		const action = policy.sample();

		assert.strictEqual(action, 0);
	});

	it("should throw when state cannot be serialized", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<Record<string, number>, number>(10);
		const agent = new DQNAgent(nn, buffer, [0, 1], {
			gamma: 0.99,
			epsilon: 0,
			batchSize: 32,
			targetUpdateFrequency: 100,
		});

		await assert.rejects(
			() => agent.act({ x: 1 }),
			(err: Error) => err.message.includes("Unable to serialize state"),
		);
	});

	it("should throw when action is not in the action space", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const agent = new DQNAgent(nn, buffer, [0, 1], {
			gamma: 0.99,
			epsilon: 0,
			batchSize: 1,
			targetUpdateFrequency: 100,
		});

		await assert.rejects(
			() =>
				agent.learn({
					state: 1,
					action: 99,
					reward: 1,
					nextState: 2,
					done: true,
				}),
			(err: Error) => err.message.includes("Action not found"),
		);
	});

	it("should save and load agent state", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
			seed: 42,
		});

		await agent.learn({
			state: 1,
			action: 0,
			reward: 2,
			nextState: 2,
			done: true,
		});

		const tempDir = await mkdtemp(join(tmpdir(), "am-reinforcement-dqn-"));
		const filePath = join(tempDir, "agent.json");

		await agent.save(filePath);

		const restoredNetwork = new MockNN();
		const restoredBuffer = new ReplayBuffer<number, number>(10);
		const restored = new DQNAgent(restoredNetwork, restoredBuffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
			seed: 42,
		});

		await restored.load(filePath);

		assert.strictEqual(restored.getStepCount(), 1);
		assert.deepStrictEqual(restoredNetwork.getWeights(), nn.getWeights());

		await rm(tempDir, { recursive: true, force: true });
	});

	it("should default step count to zero when loading data without stepCount", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(10);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
			seed: 42,
		});

		const tempDir = await mkdtemp(join(tmpdir(), "am-reinforcement-dqn-"));
		const filePath = join(tempDir, "agent.json");

		await writeFile(
			filePath,
			JSON.stringify(
				{
					config: {
						gamma: 0.99,
						epsilon: 0.1,
						batchSize: 4,
						targetUpdateFrequency: 100,
						seed: 42,
					},
					qNetworkWeights: nn.serialize(),
				},
				null,
				2,
			),
		);

		await agent.load(filePath);

		assert.strictEqual(agent.getStepCount(), 0);

		await rm(tempDir, { recursive: true, force: true });
	});

	it("should have reproducible behavior with seed", async () => {
		const createAgent = () => {
			const nn = new MockNN();
			const buffer = new ReplayBuffer<number, number>(10);
			return new DQNAgent(nn, buffer, [0, 1, 2], {
				gamma: 0.99,
				epsilon: 0.5,
				batchSize: 4,
				targetUpdateFrequency: 100,
				seed: 42,
			});
		};

		const agent1 = createAgent();
		const agent2 = createAgent();

		const actions1: number[] = [];
		const actions2: number[] = [];

		for (let i = 0; i < 20; i++) {
			actions1.push(await agent1.act(1));
			actions2.push(await agent2.act(1));
		}

		assert.deepStrictEqual(actions1, actions2);
	});

	it("should handle terminal states", async () => {
		let targetValue: number | null = null;

		const nn: NeuralNetworkModel = {
			predict: () => [0, 0],
			train: (_states: unknown[], targets: number[][]) => {
				targetValue = targets[0][0];
			},
			clone: function () {
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<number, number>(100);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 1,
			targetUpdateFrequency: 100,
		});

		// Terminal transition with reward 100
		await agent.learn({
			state: 1,
			action: 0,
			reward: 100,
			nextState: 2,
			done: true,
		});

		// Target should be reward only (no future value for terminal)
		assert.strictEqual(targetValue, 100);
	});

	it("should handle non-terminal states", async () => {
		let targetValue: number | null = null;

		const nn: NeuralNetworkModel = {
			predict: () => [5, 10],
			train: (_states: unknown[], targets: number[][]) => {
				targetValue = targets[0][0];
			},
			clone: function () {
				return this;
			},
			serialize: () => "",
			deserialize: () => {},
		};

		const buffer = new ReplayBuffer<number, number>(100);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 1,
			targetUpdateFrequency: 100,
		});

		// Non-terminal: reward + gamma * max(Q(s', a'))
		await agent.learn({
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: false,
		});

		// Target = 5 + 0.99 * max(5, 10) = 5 + 0.99 * 10 = 14.9
		assert.strictEqual(targetValue, 5 + 0.99 * 10);
	});

	it("should not train if batch size exceeds buffer", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(100);
		const actions = [0, 1];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 100, // Large batch size
			targetUpdateFrequency: 100,
		});

		// Add only a few transitions
		for (let i = 0; i < 5; i++) {
			await agent.learn({
				state: i,
				action: 0,
				reward: 1,
				nextState: i + 1,
				done: false,
			});
		}

		// Should not throw even though batch size > buffer size
		assert.strictEqual(agent.getStepCount(), 5);
	});

	it("should handle multiple actions", async () => {
		const nn = new MockNN();
		const buffer = new ReplayBuffer<number, number>(100);
		const actions = [10, 20, 30, 40, 50];

		const agent = new DQNAgent(nn, buffer, actions, {
			gamma: 0.99,
			epsilon: 0.1,
			batchSize: 4,
			targetUpdateFrequency: 100,
		});

		// Learn with different actions
		for (let i = 0; i < 10; i++) {
			await agent.learn({
				state: i,
				action: actions[i % actions.length],
				reward: i,
				nextState: i + 1,
				done: i === 9,
			});
		}

		assert.strictEqual(agent.getStepCount(), 10);
	});
});
