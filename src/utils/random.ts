/**
 * Seeded random number generator for reproducibility.
 * Uses a simple linear congruential generator (LCG).
 *
 * This allows RL experiments to be reproducible across runs,
 * which is critical for debugging and publication.
 */
export class SeededRNG {
	private state: number;

	constructor(seed: number) {
		// Ensure seed is properly initialized
		this.state = Math.abs(seed) | 0; // Convert to 32-bit integer
		if (this.state === 0) {
			this.state = 1;
		}
	}

	/**
	 * Generate a random number in [0, 1).
	 */
	random(): number {
		// LCG parameters (Park and Miller)
		const a = 1103515245;
		const c = 12345;
		const m = 2147483647; // 2^31 - 1

		this.state = (a * this.state + c) % m;
		return (this.state >>> 0) / m;
	}

	/**
	 * Generate a random integer in [min, max) (max exclusive).
	 */
	randint(min: number, max: number): number {
		if (min >= max) {
			throw new Error("min must be less than max");
		}
		return Math.floor(min + this.random() * (max - min));
	}

	/**
	 * Generate a random element from an array.
	 */
	choice<T>(array: T[]): T {
		if (array.length === 0) {
			throw new Error("Cannot choose from empty array");
		}
		const idx = this.randint(0, array.length);
		return array[idx];
	}

	/**
	 * Shuffle an array in-place using Fisher-Yates.
	 */
	shuffle<T>(array: T[]): T[] {
		const copy = [...array];
		for (let i = copy.length - 1; i > 0; i--) {
			const j = this.randint(0, i + 1);
			[copy[i], copy[j]] = [copy[j], copy[i]];
		}
		return copy;
	}
}
