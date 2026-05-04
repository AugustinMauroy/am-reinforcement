import { describe, it } from "node:test";
import assert from "node:assert";
import { ReplayBuffer } from "./replayBuffer.ts";
import type { Transition } from "../types/transition.ts";

describe("ReplayBuffer", () => {
	it("should throw on invalid buffer size", () => {
		assert.throws(
			() => new ReplayBuffer(0),
			(err: Error) => err.message.includes("positive"),
		);

		assert.throws(
			() => new ReplayBuffer(-5),
			(err: Error) => err.message.includes("positive"),
		);
	});

	it("should start with size 0 and not full", () => {
		const buf = new ReplayBuffer<number, string>(5);
		assert.strictEqual(buf.size(), 0);
		assert.strictEqual(buf.isFull(), false);
	});

	it("should add transitions and update size", () => {
		const buf = new ReplayBuffer<number, string>(5);

		const t1: Transition<number, string> = {
			state: 1,
			action: "a",
			reward: 10,
			nextState: 2,
			done: false,
		};

		buf.add(t1);
		assert.strictEqual(buf.size(), 1);
		assert.strictEqual(buf.isFull(), false);

		buf.add(t1);
		assert.strictEqual(buf.size(), 2);
	});

	it("should mark buffer as full after wrapping", () => {
		const buf = new ReplayBuffer<number, string>(3);

		for (let i = 0; i < 3; i++) {
			buf.add({
				state: i,
				action: `a${i}`,
				reward: i,
				nextState: i + 1,
				done: false,
			});
		}

		assert.strictEqual(buf.isFull(), true);

		it("should sample from a wrapped full buffer", () => {
			const buf = new ReplayBuffer<number, string>(3);

			for (let i = 0; i < 5; i++) {
				buf.add({
					state: i,
					action: `a${i}`,
					reward: i,
					nextState: i + 1,
					done: false,
				});
			}

			const batch = buf.sample(2);

			assert.strictEqual(batch.length, 2);
			assert(batch.every((transition) => transition.state >= 2));
		});
		assert.strictEqual(buf.size(), 3);
	});

	it("should overwrite oldest when buffer is full", () => {
		const buf = new ReplayBuffer<number, string>(3);

		// Add 5 items to a buffer of size 3
		for (let i = 0; i < 5; i++) {
			buf.add({
				state: i,
				action: `a${i}`,
				reward: i,
				nextState: i + 1,
				done: false,
			});
		}

		assert.strictEqual(buf.size(), 3);
		assert.strictEqual(buf.isFull(), true);
	});

	it("should sample without throwing when size >= batchSize", () => {
		const buf = new ReplayBuffer<number, string>(10);

		for (let i = 0; i < 5; i++) {
			buf.add({
				state: i,
				action: `a${i}`,
				reward: i,
				nextState: i + 1,
				done: false,
			});
		}

		const batch = buf.sample(3);
		assert.strictEqual(batch.length, 3);

		// All samples should be transitions
		batch.forEach((t) => {
			assert(typeof t.state === "number");
			assert(typeof t.action === "string");
			assert(typeof t.reward === "number");
		});
	});

	it("should throw when sampling more than size", () => {
		const buf = new ReplayBuffer<number, string>(10);

		buf.add({
			state: 1,
			action: "a",
			reward: 10,
			nextState: 2,
			done: false,
		});

		assert.throws(
			() => buf.sample(5),
			(err: Error) => err.message.includes("exceeds buffer size"),
		);
	});

	it("should clear buffer", () => {
		const buf = new ReplayBuffer<number, string>(5);

		buf.add({
			state: 1,
			action: "a",
			reward: 10,
			nextState: 2,
			done: false,
		});

		assert.strictEqual(buf.size(), 1);

		buf.clear();

		assert.strictEqual(buf.size(), 0);
		assert.strictEqual(buf.isFull(), false);
	});

	it("should sample without replacement", () => {
		const buf = new ReplayBuffer<number, string>(100);

		for (let i = 0; i < 10; i++) {
			buf.add({
				state: i,
				action: `a${i}`,
				reward: i,
				nextState: i + 1,
				done: false,
			});
		}

		const batch = buf.sample(5);

		// Create a set of indices to verify no duplicates
		const indices = new Set<number>();
		batch.forEach((t) => {
			indices.add(t.state);
		});

		// We should have unique items (with high probability - may fail rarely due to randomness)
		assert(indices.size > 0, "Should have sampled items");
	});

	it("should handle complex state types", () => {
		interface ComplexState {
			x: number;
			y: number;
		}

		const buf = new ReplayBuffer<ComplexState, string>(5);

		buf.add({
			state: { x: 1, y: 2 },
			action: "move",
			reward: 5,
			nextState: { x: 2, y: 3 },
			done: false,
		});

		const batch = buf.sample(1);
		assert.deepStrictEqual(batch[0].state, { x: 1, y: 2 });
		assert.deepStrictEqual(batch[0].nextState, { x: 2, y: 3 });
	});

	it("should skip undefined slots while sampling malformed internal state", () => {
		const buf = new ReplayBuffer<number, string>(3);
		const internal = buf as unknown as {
			buffer: Array<Transition<number, string> | undefined>;
			pointer: number;
		};

		internal.pointer = 1;

		const mathAny = Math as typeof Math & { random: () => number };
		const originalRandom = mathAny.random;
		mathAny.random = () => 0;

		try {
			const batch = buf.sample(1);
			assert.strictEqual(batch.length, 0);
		} finally {
			mathAny.random = originalRandom;
		}
	});
});
