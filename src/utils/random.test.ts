import { describe, it } from "node:test";
import assert from "node:assert";
import { SeededRNG } from "./random.ts";

describe("SeededRNG", () => {
	it("should produce consistent results with same seed", () => {
		const rng1 = new SeededRNG(42);
		const rng2 = new SeededRNG(42);

		const values1 = Array.from({ length: 10 }, () => rng1.random());
		const values2 = Array.from({ length: 10 }, () => rng2.random());

		assert.deepStrictEqual(values1, values2);
	});

	it("should produce different results with different seeds", () => {
		const rng1 = new SeededRNG(42);
		const rng2 = new SeededRNG(43);

		const values1 = Array.from({ length: 10 }, () => rng1.random());
		const values2 = Array.from({ length: 10 }, () => rng2.random());

		assert.notDeepStrictEqual(values1, values2);
	});

	it("should return values in [0, 1)", () => {
		const rng = new SeededRNG(123);

		for (let i = 0; i < 1000; i++) {
			const value = rng.random();
			assert(value >= 0 && value < 1, `Value ${value} out of range`);
		}
	});

	it("should handle seed of 0", () => {
		const rng = new SeededRNG(0);
		const value = rng.random();
		assert(typeof value === "number");
		assert(value >= 0 && value < 1);
	});

	it("should handle negative seeds", () => {
		const rng = new SeededRNG(-42);
		const value = rng.random();
		assert(typeof value === "number");
		assert(value >= 0 && value < 1);
	});

	it("randint should return integers in [min, max)", () => {
		const rng = new SeededRNG(42);

		for (let i = 0; i < 100; i++) {
			const value = rng.randint(0, 10);
			assert(Number.isInteger(value), `${value} is not integer`);
			assert(value >= 0 && value < 10, `${value} out of range [0, 10)`);
		}
	});

	it("randint should throw on invalid range", () => {
		const rng = new SeededRNG(42);

		assert.throws(
			() => rng.randint(10, 5),
			(err: Error) => err.message.includes("min must be less than max"),
		);

		assert.throws(
			() => rng.randint(5, 5),
			(err: Error) => err.message.includes("min must be less than max"),
		);
	});

	it("choice should return element from array", () => {
		const rng = new SeededRNG(42);
		const arr = ["a", "b", "c", "d", "e"];

		for (let i = 0; i < 50; i++) {
			const choice = rng.choice(arr);
			assert(arr.includes(choice), `${choice} not in array`);
		}
	});

	it("choice should throw on empty array", () => {
		const rng = new SeededRNG(42);

		assert.throws(
			() => rng.choice([]),
			(err: Error) => err.message.includes("empty array"),
		);
	});

	it("choice should return diverse elements", () => {
		const rng = new SeededRNG(42);
		const arr = [1, 2, 3, 4, 5];
		const choices = new Set<number>();

		for (let i = 0; i < 100; i++) {
			choices.add(rng.choice(arr));
		}

		// Should have picked at least 3 different elements in 100 tries
		assert(choices.size >= 3, `Only picked ${choices.size} unique elements`);
	});

	it("shuffle should return array same length", () => {
		const rng = new SeededRNG(42);
		const arr = [1, 2, 3, 4, 5];

		const shuffled = rng.shuffle(arr);

		assert.strictEqual(shuffled.length, arr.length);
	});

	it("shuffle should not modify original array", () => {
		const rng = new SeededRNG(42);
		const arr = [1, 2, 3, 4, 5];
		const original = [...arr];

		rng.shuffle(arr);

		assert.deepStrictEqual(arr, original);
	});

	it("shuffle should contain same elements", () => {
		const rng = new SeededRNG(42);
		const arr = [1, 2, 3, 4, 5];

		const shuffled = rng.shuffle(arr);

		assert.deepStrictEqual(
			shuffled.sort((a, b) => a - b),
			arr.sort((a, b) => a - b),
		);
	});

	it("shuffle with same seed should produce same order", () => {
		const arr = [1, 2, 3, 4, 5];

		const rng1 = new SeededRNG(42);
		const shuffled1 = rng1.shuffle(arr);

		const rng2 = new SeededRNG(42);
		const shuffled2 = rng2.shuffle(arr);

		assert.deepStrictEqual(shuffled1, shuffled2);
	});

	it("shuffle should produce different orders for different seeds", () => {
		const arr = [1, 2, 3, 4, 5];

		const rng1 = new SeededRNG(42);
		const shuffled1 = rng1.shuffle(arr);

		const rng2 = new SeededRNG(43);
		const shuffled2 = rng2.shuffle(arr);

		// Very unlikely to be same after multiple shuffles
		let same = true;
		for (let i = 0; i < 5; i++) {
			if (shuffled1[i] !== shuffled2[i]) {
				same = false;
				break;
			}
		}

		assert(!same, "Shuffles should differ for different seeds");
	});
});
