import type { Transition } from "../types/transition.ts";

/**
 * Interface for replay buffer implementations.
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export interface IReplayBuffer<S, A> {
	/**
	 * Add a transition to the buffer.
	 * When the buffer is full, the oldest transition is overwritten (circular).
	 */
	add(transition: Transition<S, A>): void;

	/**
	 * Sample a batch of transitions uniformly at random.
	 *
	 * @param batchSize - Number of transitions to sample
	 * @returns Array of sampled transitions
	 * @throws If batch size exceeds buffer size
	 */
	sample(batchSize: number): Transition<S, A>[];

	/**
	 * Get the current number of transitions in the buffer.
	 */
	size(): number;

	/**
	 * Clear all transitions from the buffer.
	 */
	clear(): void;

	/**
	 * Get whether the buffer has been filled at least once.
	 * Useful for knowing when to start training.
	 */
	isFull(): boolean;
}

/**
 * Circular replay buffer for storing transitions.
 *
 * This implementation:
 * - Uses a fixed-size circular buffer (no array resizing)
 * - Overwrites oldest transitions when full
 * - Supports efficient uniform sampling
 * - Has O(1) add and O(batch_size) sample operations
 *
 * @template S - Type of the state representation
 * @template A - Type of the action representation
 */
export class ReplayBuffer<S, A> implements IReplayBuffer<S, A> {
	private buffer: (Transition<S, A> | undefined)[];
	private pointer = 0;
	private filled = false;
	private maxSize: number;

	constructor(maxSize: number) {
		if (maxSize <= 0) {
			throw new Error("Buffer size must be positive");
		}
		this.maxSize = maxSize;
		this.buffer = new Array(maxSize);
	}

	add(transition: Transition<S, A>): void {
		this.buffer[this.pointer] = transition;
		this.pointer = (this.pointer + 1) % this.maxSize;

		// Mark buffer as filled once we wrap around
		if (this.pointer === 0) {
			this.filled = true;
		}
	}

	sample(batchSize: number): Transition<S, A>[] {
		const currentSize = this.size();

		if (batchSize > currentSize) {
			throw new Error(
				`Batch size (${batchSize}) exceeds buffer size (${currentSize})`,
			);
		}

		const batch: Transition<S, A>[] = [];
		const indices = new Set<number>();

		// Sample without replacement using Fisher-Yates approach
		while (indices.size < batchSize) {
			const randomIdx = Math.floor(Math.random() * currentSize);
			indices.add(randomIdx);
		}

		// Convert indices to actual buffer positions and collect transitions
		for (const idx of indices) {
			const bufferIdx = this.filled ? (this.pointer + idx) % this.maxSize : idx;

			const transition = this.buffer[bufferIdx];
			if (transition !== undefined) {
				batch.push(transition);
			}
		}

		return batch;
	}

	size(): number {
		return this.filled ? this.maxSize : this.pointer;
	}

	isFull(): boolean {
		return this.filled;
	}

	clear(): void {
		this.buffer = new Array(this.maxSize);
		this.pointer = 0;
		this.filled = false;
	}
}
