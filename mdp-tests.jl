include("mdp.jl")
using .MDP
using Test

@testset "Action Tests" begin

@test MDP.opposit_direction(up) === down
@test MDP.opposit_direction(left) === right
@test MDP.opposit_direction(down) === up
@test MDP.opposit_direction(right) === left

end

@testset "State Tests" begin

s1 = State(1, 2)
s2 = State(1, 2)

@test s1 === s2

end

@testset "Environment Tests" begin

g1 = [0 0 0 1
      0 9 0 -1
      0 0 0 0]

e1 = Environment(g1)

@test e1.agent_state == State(3,1)

@test e1.states[1] == State(1,1)
@test e1.states[2] == State(2,1)
@test e1.states[3] == State(3,1)
@test e1.states[4] == State(1,2)
@test e1.states[5] == State(3,2)
@test e1.states[6] == State(1,3)
@test e1.states[7] == State(2,3)
@test e1.states[8] == State(3,3)
@test e1.states[9] == State(1,4)
@test e1.states[10] == State(2,4)
@test e1.states[11] == State(3,4)

@test MDP.can_action_at(e1, State(1,1))
@test !MDP.can_action_at(e1, State(2,2))
@test !MDP.can_action_at(e1, State(1,4))

@test MDP.out_of_the_grid(e1, State(0,0))
@test MDP.out_of_the_grid(e1, State(4,1))
@test MDP.out_of_the_grid(e1, State(2,5))
@test !MDP.out_of_the_grid(e1, State(3, 4))

@test MDP.move(e1, State(1,1), up) === State(1,1)
@test MDP.move(e1, State(1,1), down) === State(2,1)
@test MDP.move(e1, State(1,1), right) === State(1,2)
@test MDP.move(e1, State(2,1), right) === State(2,1)
@test MDP.move(e1, State(2,3), left) === State(2,3)
@test MDP.move(e1, State(2,3), down) === State(3,3)

probs = MDP.transition_probs(e1, State(1,1), down)
@test probs[State(2,1)] ≈ 0.8
@test probs[State(1,2)] ≈ 0.1
@test probs[State(1,1)] ≈ 0.1

probs = MDP.transition_probs(e1, State(2,3), up)
@test probs[State(1,3)] ≈ 0.8
@test probs[State(2,4)] ≈ 0.1
@test probs[State(2,3)] ≈ 0.1
@test probs[State(3,3)] ≈ 0.0
@test length(probs) === 4

@test MDP.get_reward(e1, State(1,1)) === (MDP.DEFAULT_REWARD, false)
@test MDP.get_reward(e1, State(1,4)) === (1.0, true)
@test MDP.get_reward(e1, State(2,4)) === (-1.0, true)

env2 = MDP.step(e1, up)


end