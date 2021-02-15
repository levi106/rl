module MDP

using Match
using DataStructures
using StatsBase

export up, down, left, right
export step, reset
export Action, State, Environment

@enum Action up=1 down=-1 left=2 right=-2

const DEFAULT_REWARD = -0.04

struct State
    row::Int
    col::Int
end

struct Environment
    grid::Array{Int64,2}
    move_prob::Float64
    agent_state::State
    states::Array{State,1}

    Environment(grid,move_prob,agent_state,states) = new(grid,move_prob,agent_state,states)
    Environment(grid) = Environment(grid,0.8)

    function Environment(grid::Array{Int64,2}, move_prob)
        states = [State(getindex(ij,1), getindex(ij,2)) for ij in CartesianIndices(grid) if grid[ij] != 9]
        agent_state = State(size(grid)[1],1)
        new(grid, move_prob, agent_state, states)
    end
end

@inline function reset(env::Environment)
    agent_state = State(size(env.grid)[1],1)
    return Environment(env.grid, env.move_prob, agent_state, env.states)
end

@inline function can_action_at(env::Environment, state::State)
    return env.grid[state.row,state.col] === 0
end

@inline function opposit_direction(action::Action)
    return Action(Int(action) * -1)
end

@inline function out_of_the_grid(env::Environment, state::State)
    if (1 <= state.row <= size(env.grid)[1]) && (1 <= state.col <= size(env.grid)[2])
        return false
    else
        return true
    end
end

function move(env::Environment, state::State, action::Action)
    if !can_action_at(env, state)
        error("Can't move from here")
    end

    next_state = @match Int(action) begin
                    1  => State(state.row - 1, state.col)
                    -1 => State(state.row + 1, state.col)
                    2  => State(state.row, state.col - 1)
                    -2 => State(state.row, state.col + 1)
                 end

    if !(1 <= next_state.row <= size(env.grid)[1])
        next_state = state
    end
    if !(1 <= next_state.col <= size(env.grid)[2])
        next_state = state
    end
    if env.grid[next_state.row, next_state.col] === 9
        next_state = state
    end
    return next_state
end

function transition_probs(env::Environment, state::State, action::Action)
    transition_probs = DefaultDict{State,Float64}(0.0)
    if !can_action_at(env, state)
        return transition_probs
    end

    opposit = opposit_direction(action)
    for a in instances(Action)
        if a === action
            prob = env.move_prob
        elseif a !== opposit
            prob = (1 - env.move_prob) / 2.0
        else
            prob = 0.0
        end

        next_state = move(env, state, a)
        transition_probs[next_state] += prob
    end

    return transition_probs
end

function get_reward(env::Environment, state::State)
    @match env.grid[state.row, state.col] begin
        1  => (1.0, true)
        -1 => (-1.0, true)
        _  => (DEFAULT_REWARD, false)
    end
end

function step(env::Environment, action::Action)
    next_state, reward, done = transit(env, env.agent_state, action)
    if next_state === nothing
        next_state = env.agent_state
    end
    env = Environment(env.grid, env.move_prob, next_state, env.states)
    return env, next_state, reward, done
end

function transit(env::Environment, state::State, action::Action)
    probs = transition_probs(env, state, action)
    if length(probs) === 0
        return nothing, nothing, true
    end

    next_state::State = sample(collect(keys(probs)), Weights(collect(values(probs))))
    reward, done = get_reward(env, next_state)
    next_state, reward, done
end

end