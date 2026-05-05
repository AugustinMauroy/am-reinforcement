# Reinforcement Learning

[![JSR](https://jsr.io/badges/@am/reinforcement)](https://jsr.io/@am/reinforcement)
[![codecov](https://codecov.io/gh/AugustinMauroy/am-reinforcement/graph/badge.svg?token=YMPW5PAI29)](https://codecov.io/gh/AugustinMauroy/am-reinforcement)

`@am/reinforcement` is a TypeScript reinforcement learning library for building agents that learn by interacting with environments. It focuses on strong typing, composable abstractions, and a clean separation between environments, agents, replay buffers, policies, and training loops.

The library is designed to be cross-runtime compatible, allowing usage in [Node.js](https://nodejs.org), [Deno](https://deno.com), [Bun](https://bun.sh), and modern browsers.

<!--
need beta test first

> **✅ Feature Complete**: This package is considered feature complete for v1.x. Core functionality including environments, Q-learning, DQN, replay buffers, epsilon-greedy policies, reproducible randomness, and training orchestration are implemented and production-ready.
-->

## Key Features

* **Environment-Driven Training**: Train agents by interacting with `Environment<S, A>` implementations rather than fixed datasets.
* **Tabular Q-Learning**: Includes `QLearningAgent` for discrete action spaces with configurable learning rate, discount factor, and exploration rate.
* **Deep Q-Networks**: Includes `DQNAgent` for function approximation with replay buffer integration and target network updates.
* **Experience Replay**: Provides a circular `ReplayBuffer` with O(1) insertion and random sampling.
* **Exploration Control**: Uses `EpsilonGreedyPolicy` for epsilon-greedy action selection and exploration scheduling.
* **Reproducibility**: Includes `SeededRNG` for deterministic experiments and testable training runs.
* **Serialization**: Save and load agent state with `save()` and `load()` methods where supported.
* **Async-Friendly APIs**: Works with sync or async environments, agents, and neural network backends.
* **Strict Typing**: Uses generics throughout so state and action types remain explicit from end to end.

## How to construct and use a Reinforcement Learning agent

Using `@am/reinforcement` involves these main steps:

1. **Import the necessary types and classes**:

	```typescript
	import { QLearningAgent, DQNAgent } from "@am/reinforcement/algorithms";
	import { ReplayBuffer } from "@am/reinforcement/memory";
	import { EpsilonGreedyPolicy } from "@am/reinforcement/policies";
	import { SeededRNG } from "@am/reinforcement/utils";
	import type { Environment, StepResult } from "@am/reinforcement/core";
	```

2. **Define your environment**:
    * `state`: the observation returned by the environment.
    * `action`: the decision chosen by the agent.
    * `reward`: the scalar feedback signal.
    * `done`: whether the episode has ended.

	```typescript
	import type { Environment, StepResult } from "@am/reinforcement/core";

	class GridWorld implements Environment<number, "left" | "right"> {
	  reset(): number {
	    return 0;
	  }

	  step(action: "left" | "right"): StepResult<number> {
	    const nextState = action === "right" ? 1 : 0;
	    const done = nextState === 1;

	    return {
	      state: nextState,
	      reward: done ? 1 : 0,
	      done,
	    };
	  }
	}
	```

3. **Instantiate the agent**: Choose `QLearningAgent` for tabular problems or `DQNAgent` for function approximation.

	```typescript
	import { QLearningAgent, DQNAgent } from "@am/reinforcement/algorithms";
	import { ReplayBuffer } from "@am/reinforcement/memory";

	const qAgent = new QLearningAgent<number, "left" | "right">(["left", "right"], {
	  alpha: 0.1,
	  gamma: 0.99,
	  epsilon: 0.2,
	  seed: 42,
	});

	const dqnAgent = new DQNAgent<number, "left" | "right">(network, replayBuffer, ["left", "right"], {
	  gamma: 0.99,
	  epsilon: 0.2,
	  batchSize: 32,
	  targetUpdateFrequency: 100,
	  seed: 42,
	});
	```

4. **Train the agent**: Use `train()` to run full episodes and update the agent.

	```typescript
	import { train } from "@am/reinforcement/core";

	await train({
	  env: new GridWorld(),
	  agent: qAgent,
	  episodes: 100,
	  maxStepsPerEpisode: 50,
	});
	```

5. **Run an episode without learning**: Use `runEpisode()` to evaluate the trained agent.

	```typescript
	import { runEpisode } from "@am/reinforcement/core";

	const [reward, steps] = await runEpisode(new GridWorld(), qAgent, 50);
	console.log({ reward, steps });
	```

6. **Inspect or persist model state**:

	```typescript
	console.log(qAgent.getQTable());
	console.log(qAgent.getEpsilon());

	await qAgent.save("./q-table.json");
	await qAgent.load("./q-table.json");
	```

## Simple Q-Learning Example

Here is a minimal example showing how to train a `QLearningAgent` on a small environment:

```typescript
import { QLearningAgent } from "@am/reinforcement/algorithms";
import { train } from "@am/reinforcement/core";
import type { Environment, StepResult } from "@am/reinforcement/core";

class OneDimensionalWalk implements Environment<number, "left" | "right"> {
  private position = 0;

  reset(): number {
    this.position = 0;
    return this.position;
  }

  step(action: "left" | "right"): StepResult<number> {
    this.position += action === "right" ? 1 : -1;

    const done = this.position >= 3;
    const reward = done ? 10 : -0.1;

    return {
      state: this.position,
      reward,
      done,
    };
  }
}

const agent = new QLearningAgent<number, "left" | "right">(["left", "right"], {
  alpha: 0.1,
  gamma: 0.95,
  epsilon: 0.2,
  seed: 123,
});

await train({
  env: new OneDimensionalWalk(),
  agent,
  episodes: 50,
  maxStepsPerEpisode: 20,
});

console.log(agent.getQTable());
```

## Simple DQN Example

For neural-network based agents, implement the `NeuralNetworkModel` interface and pair it with a replay buffer:

```typescript } from "@am/reinforcement/algorithms";
import { ReplayBuffer } from "@am/reinforcement/memory";
import type { NeuralNetworkModel } from "@am/reinforcement/algorithms;
import type { NeuralNetworkModel } from "@am/reinforcement";

const network: NeuralNetworkModel = {
  predict(state) {
    return [0.1, 0.9];
  },
  train(states, targets) {
    // Update the network weights here.
  },
  clone() {
    return this;
  },
  serialize() {
    return JSON.stringify({});
  },
  deserialize() {
    // Restore weights here.
  },
};

const replayBuffer = new ReplayBuffer<number[], number>(1000);

const agent = new DQNAgent(network, replayBuffer, [0, 1], {
  gamma: 0.99,
  epsilon: 0.1,
  batchSize: 32,
  targetUpdateFrequency: 100,
});
```

For more detailed examples, including a grid world and a DQN walkthrough, please check out the [examples/](examples/) folder in the repository.
