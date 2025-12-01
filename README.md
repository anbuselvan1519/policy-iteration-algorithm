# POLICY ITERATION ALGORITHM

## AIM
The aim of this experiment is to implement the Policy Iteration Algorithm in Reinforcement Learning to determine the optimal policy and corresponding value function for a given environment. Policy Iteration combines iterative policy evaluation and policy improvement steps to achieve convergence towards an optimal policy.

## PROBLEM STATEMENT
In Reinforcement Learning, the agent interacts with an environment modeled as a Markov Decision Process (MDP).
The challenge is to find an optimal policy that maximizes the long-term cumulative reward.
Policy Iteration addresses this by:

Evaluating the value of a given policy (Policy Evaluation).
Improving the policy based on the evaluated value function (Policy Improvement).
Repeating these steps until the policy converges to the optimal policy.

## POLICY ITERATION ALGORITHM
# STEP 1:
Initialization

Initialize an arbitrary policy π and value function V(s).
A
# STEP 2:
Policy Evaluation

For the current policy π, compute the value function V(s) for all states until convergence.

# STEP 3:
Policy Improvement

Update the policy by choosing actions that maximize the expected return using the current value function.
# STEP 4:
Check for Convergence

If the policy does not change (π′ = π), then the policy is optimal and the algorithm terminates.
Otherwise, repeat steps 2 and 3.

## POLICY IMPROVEMENT FUNCTION

```
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)

    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

    new_pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return new_pi

```
## POLICY ITERATION FUNCTION
```
def policy_iteration(P,gamma=1.0,theta=1e-10):
  random_actions=np.random.choice(tuple(P[0].keys()),len(P))
  pi=lambda s: {s:a for s, a in enumerate(random_actions)}[s]
  while True:
    old_pi={s: pi(s) for s in range(len(P))}
    V=policy_evaluation(pi,P,gamma,theta)
    pi=policy_improvement(V,P,gamma)
    if old_pi=={s:pi(s) for s in range(len(P))}:
      break
  return V,pi

```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="575" height="184" alt="Screenshot 2025-10-14 102739" src="https://github.com/user-attachments/assets/d8233ea0-f1bf-4ab4-b468-98bfcad41735" />
<img width="612" height="171" alt="Screenshot 2025-10-14 102817" src="https://github.com/user-attachments/assets/c750aeb2-a922-4116-86c8-a3e6b77875c2" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="618" height="172" alt="Screenshot 2025-10-14 102911" src="https://github.com/user-attachments/assets/42855b9a-e6d0-40dd-b8ba-1c382489e98d" />
<img width="589" height="170" alt="Screenshot 2025-10-14 102948" src="https://github.com/user-attachments/assets/6c745af0-57a8-4537-b717-dcf866b88353" />


### 3. Policy, Value function and success rate after policy iteration
<img width="598" height="198" alt="Screenshot 2025-10-14 103019" src="https://github.com/user-attachments/assets/d876ca22-2282-4ed9-a623-ebaeeaa334b1" />
<img width="606" height="177" alt="Screenshot 2025-10-14 103041" src="https://github.com/user-attachments/assets/4d40ad46-97bc-43b8-9d6d-8699b14c6de6" />




## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
