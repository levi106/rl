include("mdp.jl")
using .MDP
using StatsBase

function main()
    grid = [0 0 0 1
            0 9 0 -1
            0 0 0 0]
    env = Environment(grid)

    for i = 1:10
        env = MDP.reset(env)
        state = env.agent_state
        total_reward = 0.0
        done = false
        steps = 1
        println("($(state.row),$(state.col))=>")
        while !done
            action = Action(sample([1, -1, 2, -2]))
            new_env, next_state, reward, done = MDP.step(env, action)
            total_reward += reward
            state = next_state
            env = new_env
            print("($(state.row),$(state.col))")
            if !done
                print("=>")
                if (steps % 5) == 0
                    println("")
                end
            else
                println("")
            end
            steps += 1
        end
        println("Episode $i: Steps=$(steps) Reward=$(total_reward)\n")
    end
end

main()