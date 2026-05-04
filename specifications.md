# `@am/reinforcement` — Technical Specification

---

# 1. Design Principles

### 1.1 Core Philosophy

* **Composable over monolithic**
  The library composes with `@am/neuralnetwork` and `@am/decisiontree`, never reimplements them.

* **Strict RL abstraction boundaries**
  No leakage of supervised learning APIs into RL (e.g., no `.fit()` on agents).

* **Environment-first design**
  All training flows are driven by interaction with environments, not datasets.

* **Deterministic + reproducible by default**
  Seedable RNG across agents, buffers, and trainers.

---

### 1.2 API Design Rules

* Strong typing via generics:

  ```ts
  Agent<State, Action>
  Environment<State, Action>
  ```

* No implicit shape assumptions:

  * State can be vector, object, or tensor-like
  * Action can be discrete or continuous

* Explicit lifecycle:

  * `reset → act → step → learn`

---

### 1.3 Separation of Concerns

| Component    | Responsibility         |
| ------------ | ---------------------- |
| Environment  | World dynamics         |
| Agent        | Decision + learning    |
| Policy/Value | Function approximation |
| ReplayBuffer | Experience storage     |
| Trainer      | Orchestration          |

---

### 1.4 Extensibility Strategy

* Interface-first design
* Algorithms built as pluggable modules
* Minimal base classes, mostly composition
* Hooks for:

  * logging
  * exploration strategies
  * schedulers

---

### 1.5 Performance Considerations

* Batch-first APIs where possible
* Async-compatible environments
* Zero-copy transitions when possible
* Optional typed array usage for numeric workloads

---

# 2. Architecture Overview

```
+-------------------+
|   Environment     |
+--------+----------+
         |
         v
+--------+----------+
|      Agent        |
|  +-------------+  |
|  | Policy/Value|  |
|  +-------------+  |
+--------+----------+
         |
         v
+--------+----------+
|  Replay Buffer    |
+--------+----------+
         |
         v
+--------+----------+
|    Trainer        |
+-------------------+
```

---

### Data Flow

```
state → agent.act() → action
action → env.step() → (nextState, reward, done)
transition → buffer
buffer → agent.learn()
```

---

# 3. Core Interfaces (TypeScript)

## 3.1 Environment

```ts
export interface StepResult<S> {
  state: S
  reward: number
  done: boolean
  info?: Record<string, unknown>
}

export interface Environment<S, A> {
  reset(seed?: number): Promise<S> | S
  step(action: A): Promise<StepResult<S>> | StepResult<S>
  render?(): void | Promise<void>
}
```

---

## 3.2 Agent

```ts
export interface Agent<S, A> {
  act(state: S): A | Promise<A>

  learn(transition: Transition<S, A>): void | Promise<void>

  save?(path: string): Promise<void>
  load?(path: string): Promise<void>
}
```

---

## 3.3 Transition

```ts
export interface Transition<S, A> {
  state: S
  action: A
  reward: number
  nextState: S
  done: boolean
}
```

---

## 3.4 Policy / Value Function

```ts
export interface Policy<S, A> {
  sample(state: S): A
  probabilities?(state: S): number[]
}

export interface ValueFunction<S> {
  predict(state: S): number
}

export interface QFunction<S, A> {
  predict(state: S, action: A): number
}
```

---

## 3.5 Replay Buffer

```ts
export interface ReplayBuffer<S, A> {
  add(transition: Transition<S, A>): void
  sample(batchSize: number): Transition<S, A>[]
  size(): number
  clear(): void
}
```

---

## 3.6 Trainer

```ts
export interface TrainOptions<S, A> {
  env: Environment<S, A>
  agent: Agent<S, A>
  episodes: number
  maxStepsPerEpisode?: number
  onEpisodeEnd?: (episode: number, reward: number) => void
}

export interface Trainer {
  train<S, A>(options: TrainOptions<S, A>): Promise<void>
}
```

---

# 4. 🤖 Algorithms (MVP)

---

## 4.1 Tabular Q-Learning

### Update Rule

```
Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') - Q(s,a)]
```

---

### Implementation Structure

```ts
class QLearningAgent<S, A> implements Agent<S, A> {
  private qTable: Map<string, number>

  constructor(
    private actions: A[],
    private alpha: number,
    private gamma: number,
    private epsilon: number
  ) {}
}
```

---

### Pseudocode

```
if random < ε:
  choose random action
else:
  choose argmax Q(s,a)

update:
  target = r + γ * max Q(s', a')
  Q(s,a) += α * (target - Q(s,a))
```

---

## 4.2 Deep Q-Network (DQN)

### Components

* Q-network (from `@am/neuralnetwork`)
* Target network
* Replay buffer
* Epsilon-greedy policy

---

### Update Rule

```
y = r + γ max_a' Q_target(s', a')
loss = (Q(s,a) - y)^2
```

---

### Data Flow

```
sample batch → forward pass → compute targets → backprop
```

---

### Pseudocode

```
batch = replay.sample()

for each (s,a,r,s',done):
  target = r
  if not done:
    target += γ * max(Q_target(s'))

train Q network on (s,a) → target
```

---

## 4.3 Policy Gradient (Optional)

### REINFORCE

```
θ ← θ + α ∇ log π(a|s) * G
```

---

# 5. Integration

---

## 5.1 With `@am/neuralnetwork`

### Usage

```ts
import { NeuralNetwork } from "@am/neuralnetwork"
```

* Used as function approximator
* Agent controls training loop (NOT `.fit()`)

---

### Key Difference

| Supervised     | RL             |
| -------------- | -------------- |
| `.fit(X, y)`   | online updates |
| static dataset | dynamic buffer |

---

### Required Adapter

```ts
interface NNAdapter<S, A> {
  predict(state: S): number[]
  train(states: S[], targets: number[][]): void
}
```

---

## 5.2 With `@am/decisiontree`

Use cases:

* Policy distillation
* Offline RL analysis
* Interpretable policy extraction

Example:

```ts
tree.fit(states, actions)
```

---

# 6. Training System

---

## Core API

```ts
await train({
  env,
  agent,
  episodes: 1000,
  maxStepsPerEpisode: 200
})
```

---

## Training Loop

```ts
for episode:
  state = env.reset()

  for step:
    action = agent.act(state)
    result = env.step(action)

    agent.learn({
      state,
      action,
      reward,
      nextState,
      done
    })

    state = nextState

    if done: break
```

---

## Hooks

```ts
onEpisodeEnd(episode, totalReward)
onStep?(step, transition)
```

---

# 7. Module Structure

```
@am/reinforcement
│
├── core/
│   ├── agent.ts
│   ├── environment.ts
│   ├── trainer.ts
│
├── algorithms/
│   ├── qlearning.ts
│   ├── dqn.ts
│   ├── reinforce.ts
│
├── memory/
│   ├── replayBuffer.ts
│
├── policies/
│   ├── epsilonGreedy.ts
│
├── utils/
│   ├── random.ts
│   ├── math.ts
│
├── types/
│   ├── transition.ts
│
└── index.ts
```

---

# 8. Developer Experience

---

## Defaults

```ts
createDQNAgent({
  epsilon: 0.1,
  gamma: 0.99,
  batchSize: 32
})
```

---

## Debugging

* Step tracing
* Q-value inspection
* Replay buffer introspection

---

## Logging

```ts
train({
  ...,
  onEpisodeEnd: (ep, reward) => {
    console.log(ep, reward)
  }
})
```

---

# 9. Mineflayer Use Case

---

## Environment Wrapper

```ts
class MineflayerEnv implements Environment<State, Action> {
  async reset() {
    // spawn/reset bot
  }

  async step(action: Action) {
    // execute bot action
    // compute reward
  }
}
```

---

## State Representation

```ts
type State = {
  position: [number, number, number]
  health: number
  nearbyBlocks: number[]
}
```

---

## Action Space

```ts
type Action =
  | "move_forward"
  | "turn_left"
  | "mine"
  | "jump"
```

---

## Rewards

* +1 for mining valuable block
* -1 for damage
* +10 for goal completion

---

# 10. Serialization

---

## Agent Save/Load

```ts
await agent.save("model.json")
await agent.load("model.json")
```

---

## Strategy

* Q-table → JSON
* Neural networks → delegate to `@am/neuralnetwork`
* Replay buffer → optional persistence

---

# 11. Performance

---

## Batching

* DQN trains on batches
* Avoid per-step training

---

## Async Environments

```ts
await env.step(action)
```

---

## Memory

* Circular replay buffer
* Configurable size

---

## Limits

* JS runtime constraints
* No GPU acceleration (yet)

---

# 12. Future Extensions

---

### Algorithms

* PPO
* Actor-Critic
* SAC

---

### Systems

* Multi-agent training
* Distributed rollouts
* Parallel environments

---

### Tooling

* Visualization dashboards
* Hyperparameter tuning

---

# 13. ✅ End-to-End Example

```ts
import {
  DQNAgent,
  ReplayBuffer,
  train
} from "@am/reinforcement"

import { NeuralNetwork } from "@am/neuralnetwork"

// --- Environment

class SimpleEnv {
  state = 0

  reset() {
    this.state = 0
    return this.state
  }

  step(action: number) {
    this.state += action === 0 ? -1 : 1

    return {
      state: this.state,
      reward: this.state === 10 ? 10 : -0.1,
      done: Math.abs(this.state) >= 10
    }
  }
}

// --- Setup

const env = new SimpleEnv()

const model = new NeuralNetwork({
  input: 1,
  hidden: [16, 16],
  output: 2
})

const agent = new DQNAgent({
  model,
  actions: [0, 1],
  gamma: 0.99,
  epsilon: 0.1,
  buffer: new ReplayBuffer(10000)
})

// --- Train

await train({
  env,
  agent,
  episodes: 500,
  maxStepsPerEpisode: 100,
  onEpisodeEnd: (ep, reward) => {
    console.log(`Episode ${ep}: ${reward}`)
  }
})

// --- Inference

let state = env.reset()

while (true) {
  const action = await agent.act(state)
  const result = env.step(action)

  state = result.state

  if (result.done) break
}
```

---

# Final Notes

This design:

* Preserves **strict modularity**
* Enables **real-world RL workloads**
* Integrates cleanly with existing libraries
* Scales toward **advanced RL systems**

It is intentionally minimal at the core, but structurally capable of long-term evolution into a full RL ecosystem.
