import { describe, it } from "node:test";
import assert from "node:assert";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { QLearningAgent } from "./qlearning.ts";
import type { Transition } from "../types/transition.ts";

describe("QLearningAgent", () => {
	it("should throw on invalid config", () => {
		const actions = ["a", "b"];

		assert.throws(
			() => {
				new QLearningAgent(actions, {
					alpha: 0,
					gamma: 0.99,
					epsilon: 0.1,
				});
			},
			(err: Error) => err.message.includes("Alpha"),
		);

		assert.throws(
			() => {
				new QLearningAgent(actions, {
					alpha: 0.1,
					gamma: -0.1,
					epsilon: 0.1,
				});
			},
			(err: Error) => err.message.includes("Gamma"),
		);

		assert.throws(
			() => {
				new QLearningAgent(actions, {
					alpha: 0.1,
					gamma: 0.99,
					epsilon: 1.1,
				});
			},
			(err: Error) => err.message.includes("Epsilon"),
		);
	});

	it("should initialize with default values", () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.1,
		});

		assert.strictEqual(agent.getEpsilon(), 0.1);
		assert.deepStrictEqual(agent.getQTable(), {});
	});

	it("should act and return valid action", async () => {
		const agent = new QLearningAgent([0, 1, 2], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.5,
			seed: 42,
		});

		for (let i = 0; i < 10; i++) {
			const action = await agent.act(1);
			assert([0, 1, 2].includes(action));
		}
	});

	it("should update Q-values through learning", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.5,
			gamma: 0.9,
			epsilon: 0.1,
			seed: 42,
		});

		const transition: Transition<number, number> = {
			state: 1,
			action: 0,
			reward: 10,
			nextState: 2,
			done: false,
		};

		await agent.learn(transition);

		const qTable = agent.getQTable();
		assert(qTable["1"] !== undefined);
		assert(qTable["1"]["0"] !== 0, "Q-value should be updated");
	});

	it("should apply Q-learning update rule correctly", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 1.0, // Full update for easy verification
			gamma: 0.0, // No discount for simple calculation
			epsilon: 0.0, // Greedy only
		});

		// Initial learn
		const t1: Transition<number, number> = {
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: true,
		};

		await agent.learn(t1);

		// Q(1, 0) = 0 + 1.0 * (5 + 0 * max(Q(2, *)) - 0) = 5
		const qTable1 = agent.getQTable();
		assert.strictEqual(qTable1["1"]["0"], 5);

		// Learn from state 2
		const t2: Transition<number, number> = {
			state: 2,
			action: 0,
			reward: 3,
			nextState: 3,
			done: false,
		};

		await agent.learn(t2);

		// Q(2, 0) = 0 + 1.0 * (3 + 0 * max(Q(3, *)) - 0) = 3
		const qTable2 = agent.getQTable();
		assert.strictEqual(qTable2["2"]["0"], 3);

		// Learn from state 1 again with knowledge of state 2
		const t3: Transition<number, number> = {
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: false,
		};

		await agent.learn(t3);

		// Q(1, 0) = 5 + 1.0 * (5 + 0 * max(3) - 5) = 5
		// (since gamma = 0, the discount makes future values 0)
		const qTable3 = agent.getQTable();
		assert.strictEqual(qTable3["1"]["0"], 5);
	});

	it("should handle terminal states", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 1.0,
			gamma: 0.9,
			epsilon: 0.0,
		});

		const t1: Transition<number, number> = {
			state: 1,
			action: 0,
			reward: 100,
			nextState: 2,
			done: true, // Terminal
		};

		await agent.learn(t1);

		const qTable = agent.getQTable();
		// Q(1, 0) = 0 + 1.0 * (100 + 0 * 0 - 0) = 100
		assert.strictEqual(qTable["1"]["0"], 100);
	});

	it("should use custom state serializer", async () => {
		interface State {
			x: number;
			y: number;
		}

		const agent = new QLearningAgent<State, number>([0, 1], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.1,
			stateSerializer: (s: unknown) => {
				const state = s as State;
				return `${state.x}_${state.y}`;
			},
		});

		const transition: Transition<State, number> = {
			state: { x: 1, y: 2 },
			action: 0,
			reward: 5,
			nextState: { x: 2, y: 3 },
			done: false,
		};

		await agent.learn(transition);

		const qTable = agent.getQTable();
		assert(qTable["1_2"] !== undefined);
	});

	it("should get and set epsilon", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.5,
		});

		assert.strictEqual(agent.getEpsilon(), 0.5);

		agent.setEpsilon(0.2);
		assert.strictEqual(agent.getEpsilon(), 0.2);

		agent.setEpsilon(0.9);
		assert.strictEqual(agent.getEpsilon(), 0.9);
	});

	it("should initialize Q-values for all actions", async () => {
		const agent = new QLearningAgent([0, 1, 2], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.1,
		});

		// Take action to initialize state
		await agent.act(1);
		await agent.learn({
			state: 1,
			action: 0,
			reward: 1,
			nextState: 2,
			done: false,
		});

		const qTable = agent.getQTable();
		assert(qTable["1"] !== undefined);
		assert(qTable["1"]["0"] !== undefined);
		assert(qTable["1"]["1"] !== undefined);
		assert(qTable["1"]["2"] !== undefined);
	});

	it("should handle multiple states", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.0,
		});

		// Learn from multiple states
		await agent.learn({
			state: 1,
			action: 0,
			reward: 5,
			nextState: 2,
			done: false,
		});

		await agent.learn({
			state: 2,
			action: 1,
			reward: 3,
			nextState: 3,
			done: false,
		});

		await agent.learn({
			state: 3,
			action: 0,
			reward: 10,
			nextState: 4,
			done: true,
		});

		const qTable = agent.getQTable();
		assert(qTable["1"] !== undefined);
		assert(qTable["2"] !== undefined);
		assert(qTable["3"] !== undefined);
	});

	it("should have reproducible behavior with seed", async () => {
		const createAgent = () =>
			new QLearningAgent([0, 1, 2], {
				alpha: 0.1,
				gamma: 0.99,
				epsilon: 0.5,
				seed: 42,
			});

		const agent1 = createAgent();
		const agent2 = createAgent();

		// Record actions
		const actions1: number[] = [];
		const actions2: number[] = [];

		for (let i = 0; i < 20; i++) {
			actions1.push(await agent1.act(1));
			actions2.push(await agent2.act(1));
		}

		assert.deepStrictEqual(actions1, actions2);
	});

	it("should initialize Q-values to 0", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.1,
			gamma: 0.99,
			epsilon: 0.0,
		});

		// First transition
		await agent.learn({
			state: 1,
			action: 0,
			reward: 0,
			nextState: 2,
			done: true,
		});

		const qTable = agent.getQTable();
		// Q(1, 0) should be updated from 0
		assert(qTable["1"]["0"] !== undefined);
	});

	it("should handle complex state objects", async () => {
		interface GameState {
			health: number;
			position: number;
			inventory: string[];
		}

		const agent = new QLearningAgent<GameState, string>(
			["attack", "defend", "heal"],
			{
				alpha: 0.1,
				gamma: 0.99,
				epsilon: 0.1,
				stateSerializer: (s: unknown) => {
					const state = s as GameState;
					return JSON.stringify({
						health: state.health,
						position: state.position,
					});
				},
			},
		);

		const state1: GameState = {
			health: 100,
			position: 0,
			inventory: ["sword"],
		};
		const state2: GameState = { health: 90, position: 1, inventory: ["sword"] };

		const transition: Transition<GameState, string> = {
			state: state1,
			action: "attack",
			reward: 10,
			nextState: state2,
			done: false,
		};

		await agent.learn(transition);

		const qTable = agent.getQTable();
		assert(Object.keys(qTable).length > 0);
	});

	it("should serialize object actions and restore from disk", async () => {
		interface State {
			id: number;
		}

		const actions = [{ kind: "left" }, { kind: "right" }];
		const config = {
			alpha: 1,
			gamma: 0,
			epsilon: 0,
			stateSerializer: (s: unknown) => {
				const state = s as State;
				return `state-${state.id}`;
			},
		};

		const agent = new QLearningAgent<State, (typeof actions)[number]>(
			actions,
			config,
		);

		await agent.learn({
			state: { id: 1 },
			action: actions[1],
			reward: 7,
			nextState: { id: 2 },
			done: true,
		});

		const tempDir = await mkdtemp(
			join(tmpdir(), "am-reinforcement-qlearning-"),
		);
		const filePath = join(tempDir, "agent.json");

		await agent.save(filePath);

		const restored = new QLearningAgent<State, (typeof actions)[number]>(
			actions,
			config,
		);
		await restored.load(filePath);

		assert.deepStrictEqual(restored.getQTable(), agent.getQTable());

		await rm(tempDir, { recursive: true, force: true });
	});

	it("should fall back to zero for missing actions after loading malformed q-table data", async () => {
		interface State {
			id: number;
		}

		interface Action {
			kind: string;
		}

		const actions: Action[] = [{ kind: "left" }, { kind: "right" }];
		const config = {
			alpha: 1,
			gamma: 0,
			epsilon: 0,
			stateSerializer: (s: unknown) => {
				const state = s as State;
				return `state-${state.id}`;
			},
		};

		const tempDir = await mkdtemp(
			join(tmpdir(), "am-reinforcement-qlearning-"),
		);
		const filePath = join(tempDir, "malformed.json");

		await writeFile(
			filePath,
			JSON.stringify(
				{
					config,
					qTable: [["state-1", [[JSON.stringify(actions[0]), 5]]]],
				},
				null,
				2,
			),
		);

		const agent = new QLearningAgent<State, Action>(actions, config);
		await agent.load(filePath);

		const greedyAction = await agent.act({ id: 1 });
		assert.deepStrictEqual(greedyAction, actions[0]);

		await agent.learn({
			state: { id: 1 },
			action: actions[1],
			reward: 2,
			nextState: { id: 2 },
			done: true,
		});

		const qTable = agent.getQTable();
		assert.strictEqual(qTable["state-1"][JSON.stringify(actions[1])], 2);

		await rm(tempDir, { recursive: true, force: true });
	});

	it("should converge Q-values to optimal with consistent rewards", async () => {
		const agent = new QLearningAgent([0, 1], {
			alpha: 0.5,
			gamma: 0.9,
			epsilon: 0.0, // Greedy only
		});

		// Repeatedly learn the same transition
		for (let i = 0; i < 100; i++) {
			await agent.learn({
				state: 1,
				action: 0,
				reward: 10,
				nextState: 2,
				done: true,
			});
		}

		const qTable = agent.getQTable();
		// Should converge to the reward value for terminal transitions
		assert(
			qTable["1"]["0"] > 9.9,
			`Q-value is ${qTable["1"]["0"]}, expected near 10`,
		);
	});
});
