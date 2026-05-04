import { describe, it } from "node:test";
import assert from "node:assert";
import { EpsilonGreedyPolicy } from "./epsilonGreedy.ts";

describe("EpsilonGreedyPolicy", () => {
	it("should throw on invalid epsilon", () => {
		assert.throws(() => {
			new EpsilonGreedyPolicy(["a", "b"], () => "a", -0.1);
		});

		assert.throws(() => {
			new EpsilonGreedyPolicy(["a", "b"], () => "a", 1.1);
		});
	});

	it("should throw on empty actions", () => {
		assert.throws(() => {
			new EpsilonGreedyPolicy([], () => "a", 0.1);
		});
	});

	it("should return greedy action when epsilon is 0", () => {
		const actions = ["a", "b", "c"];
		let greedyCallCount = 0;

		const policy = new EpsilonGreedyPolicy(
			actions,
			() => {
				greedyCallCount++;
				return "a";
			},
			0,
		);

		// With epsilon=0, should always call greedy
		for (let i = 0; i < 10; i++) {
			const action = policy.sample();
			assert.strictEqual(action, "a");
		}

		assert.strictEqual(greedyCallCount, 10);
	});

	it("should explore with epsilon=1", () => {
		const actions = [0, 1, 2, 3, 4];
		let greedyCallCount = 0;

		const policy = new EpsilonGreedyPolicy(
			actions,
			() => {
				greedyCallCount++;
				return 0;
			},
			1,
		);

		// With epsilon=1, should never call greedy
		for (let i = 0; i < 20; i++) {
			policy.sample();
		}

		assert.strictEqual(greedyCallCount, 0);
	});

	it("should mix exploration and exploitation with mid epsilon", () => {
		const actions = ["a", "b"];
		let greedyCallCount = 0;
		const rng = () => {
			// Predictable sequence: 0.1, 0.6, 0.2, 0.7, ...
			const vals = [0.1, 0.6, 0.2, 0.7];
			return vals[greedyCallCount % 4];
		};

		const policy = new EpsilonGreedyPolicy(
			actions,
			() => {
				greedyCallCount++;
				return "a";
			},
			0.5,
			rng,
		);

		// First call: 0.1 < 0.5 → explore (random action)
		let action = policy.sample();
		assert(action === "a" || action === "b");

		// Second call: 0.6 > 0.5 → exploit (greedy)
		action = policy.sample();
		assert.strictEqual(action, "a");
	});

	it("should have initial epsilon", () => {
		const policy = new EpsilonGreedyPolicy(["a", "b"], () => "a", 0.3);
		assert.strictEqual(policy.getEpsilon(), 0.3);
	});

	it("should update epsilon", () => {
		const policy = new EpsilonGreedyPolicy(["a", "b"], () => "a", 0.5);

		policy.setEpsilon(0.1);
		assert.strictEqual(policy.getEpsilon(), 0.1);

		policy.setEpsilon(0.9);
		assert.strictEqual(policy.getEpsilon(), 0.9);
	});

	it("should throw on invalid epsilon update", () => {
		const policy = new EpsilonGreedyPolicy(["a", "b"], () => "a", 0.5);

		assert.throws(() => policy.setEpsilon(-0.1));
		assert.throws(() => policy.setEpsilon(1.1));
	});

	it("should decay epsilon", () => {
		const policy = new EpsilonGreedyPolicy(["a", "b"], () => "a", 1.0);

		assert.strictEqual(policy.getEpsilon(), 1.0);

		policy.decayEpsilon(0.99);
		assert(Math.abs(policy.getEpsilon() - 0.99) < 1e-9);

		policy.decayEpsilon(0.99);
		assert(Math.abs(policy.getEpsilon() - 0.9801) < 1e-9);
	});

	it("probabilities should sum to 1", () => {
		const actions = [1, 2, 3];
		const policy = new EpsilonGreedyPolicy(actions, () => 1, 0.1);

		const probs = policy.probabilities();

		const sum = probs.reduce((a, b) => a + b, 0);
		assert(Math.abs(sum - 1.0) < 1e-9, `Probabilities sum to ${sum}, not 1`);
	});

	it("should return correct number of probabilities", () => {
		const actions = ["a", "b", "c", "d"];
		const policy = new EpsilonGreedyPolicy(actions, () => "a", 0.2);

		const probs = policy.probabilities();
		assert.strictEqual(probs.length, 4);
	});

	it("should have higher probability for greedy action", () => {
		const actions = ["a", "b", "c"];
		const policy = new EpsilonGreedyPolicy(actions, () => "a", 0.3);

		const probs = policy.probabilities();

		assert(probs[0] > probs[1]);
		assert(probs[0] > probs[2]);
	});

	it("should explore all actions", () => {
		const actions = ["a", "b", "c"];
		const policy = new EpsilonGreedyPolicy(actions, () => "a", 1.0);

		const sampled = new Set<string>();
		for (let i = 0; i < 100; i++) {
			sampled.add(policy.sample());
		}

		// With epsilon=1, should explore all actions
		assert(sampled.has("a"));
		assert(sampled.has("b"));
		assert(sampled.has("c"));
	});

	it("should work with numeric actions", () => {
		const actions = [0, 1, 2, 3];
		const policy = new EpsilonGreedyPolicy(actions, () => 0, 0.5);

		for (let i = 0; i < 20; i++) {
			const action = policy.sample();
			assert(actions.includes(action));
		}
	});

	it("should work with complex action objects", () => {
		interface Action {
			type: string;
			value: number;
		}

		const actions: Action[] = [
			{ type: "move", value: 1 },
			{ type: "jump", value: 2 },
		];

		const policy = new EpsilonGreedyPolicy(actions, () => actions[0], 0.5);

		for (let i = 0; i < 10; i++) {
			const action = policy.sample();
			assert(
				(action.type === "move" && action.value === 1) ||
					(action.type === "jump" && action.value === 2),
			);
		}
	});
});
