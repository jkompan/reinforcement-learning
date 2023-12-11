using ReinforcementLearningBase, GridWorlds
using PyPlot

world = GridWorlds.GridRoomsDirectedModule.GridRoomsDirected();
env = GridWorlds.RLBaseEnv(world)

mutable struct Agent
    env::AbstractEnv
    algo::Symbol
    ϵ::Float64 #współczynnik eksploracji
    ϵ_decay::Float64
    ϵ_min::Float64
    β::Float64 #dyskonto
    α::Float64 #stopa uczenia się
    Q::Dict
    score::Int #wynik - ile razy agent dotarl do mety
    steps_per_episode::Vector{Float64} #ile trwal przecietny epizod
    use_UCB::Bool
    N::Dict # licznik odwiedzin
end

function Agent(env, algo; ϵ = 1.0, ϵ_decay = 0.9975, ϵ_min = 0.005,
        β = 0.99, α = 0.1, use_UCB = false) 
    if algo != :SARSA && algo != :Qlearning
        @error "unknown algorithm"
    end
    Agent(env, algo, ϵ, ϵ_decay, ϵ_min, β, α, 
        Dict(), 0, [0.0,], use_UCB, Dict())
end

# SARSA
function learn!(agent, S, A, r, S′,A′)
    if !haskey(agent.Q, S)  # if state has not been visited yet
        agent.Q[S] = zeros(length(action_space(agent.env)))     # initialize Q values with 0
        agent.Q[S][A] = r   # save immediate reward corresponding to performed action
    else
        Q_S′ = 0.0
        haskey(agent.Q, S′) && (Q_S′ += agent.Q[S′][A′])    # find Q value corresponding to action selected in the next state (unless the state has not been visited)
        agent.Q[S][A] += agent.α * (r + agent.β*Q_S′ - agent.Q[S][A])   # update Q value for state-action pair (S,A)
    end
end

# Q learning
function learn!(agent, S, A, r, S′)
    if !haskey(agent.Q, S)  # just as above
        agent.Q[S] = zeros(length(action_space(agent.env)))
        agent.Q[S][A] = r
    else
        Q_S′ = 0.0
        haskey(agent.Q, S′) && ( Q_S′ += maximum(agent.Q[S′]))  # find Q value corresponding to best action in the next state
        agent.Q[S][A] += agent.α * (r + agent.β*Q_S′ - agent.Q[S][A])   # update Q value
    end
end

# auxilliary function: Upper Confidence Bound
calculate_UCB(Q,episode,N) = Q .+ sqrt.(2*log(episode)./max.(1,N))


# function to perform learning
function run_learning!(agent, steps; burning = true, 
    animated = nothing) 
step = 1.0
steps_per_episode = 1.0
episode = 1.0

if !isnothing(animated)
    global str = ""
    global str = str * "FRAME_START_DELIMITER"
    global str = str * "step: $(step)\n"
    global str = str * "episode: $(episode)\n"
    global str = str * repr(MIME"text/plain"(), env)
    global str = str * "\ntotal_reward: 0"
end
while step <= steps
    S = deepcopy(state(agent.env))

    # version using upper confidence bound (UCB)
    if agent.use_UCB

        # select action
        if (burning && step < 0.1*steps) || !haskey(agent.Q, S)
            A = rand(1:length(action_space(agent.env)))  # randomly if state S has not been visited yet
        else    
            UCB = calculate_UCB(agent.Q[S], episode, agent.N[S])
            A = argmax(UCB)     # or maximizing upper confidence bound
        end
        # initialize action counts for a newly visited state
        if !haskey(agent.Q, S)
            agent.N[S] = zeros(length(action_space(agent.env)))
        end
        agent.N[S][A] += 1  # update state-action counts
        
        agent.env(action_space(agent.env)[A])   # perform selected action
        r = reward(agent.env)
        S′ = deepcopy(state(agent.env))

        # learn
        if agent.algo == :SARSA
            # select action in the next state
            if (burning && step < 0.1*steps) || !haskey(agent.Q, S′)
                A′ = rand(1:length(action_space(agent.env))) # randomly if state S′ has not been visited yet
            else    
                UCB = calculate_UCB(agent.Q[S′], episode, agent.N[S′])
                A′ = argmax(UCB)    # or maximizing upper confidence bound
            end
            learn!(agent, S, A, r, S′,A′)
        else
            learn!(agent, S, A, r, S′)
        end
    
    # non-UCB version (epsilon greedy)
    else
        if (burning && step < 0.1*steps) || rand() < agent.ϵ || !haskey(agent.Q, state(agent.env))
            A = rand(1:length(action_space(agent.env)))
        else 
            A = argmax(agent.Q[state(agent.env)])
        end
        agent.env(action_space(agent.env)[A])
        r = reward(agent.env)
        S′ = deepcopy(state(agent.env))
        if agent.algo == :SARSA
            if (burning && step < 0.1 * steps) || rand() < agent.ϵ || !haskey(agent.Q, state(agent.env))
                A′ = rand(1:length(action_space(agent.env)))
            else 
                A′ = argmax(agent.Q[state(agent.env)])
            end
            learn!(agent, S, A, r, S′,A′)
        else
            learn!(agent, S, A, r, S′)
        end
    end

    if !isnothing(animated) 
        global str = str * "FRAME_START_DELIMITER"
        global str = str * "step: $(step)\n"
        global str = str * "episode: $(episode)\n"
        global str = str * repr(MIME"text/plain"(), env)
        global str = str * "\ntotal_reward: $(agent.score)"
    end

    if is_terminated(agent.env)
        if !agent.use_UCB
            eps = agent.ϵ * agent.ϵ_decay
            agent.ϵ = max(agent.ϵ_min, eps)
        end
        agent.score += 1.0
        push!(agent.steps_per_episode, 
            agent.steps_per_episode[end] + (steps_per_episode - agent.steps_per_episode[end])/episode)
        episode += 1.0
        steps_per_episode = 0
        reset!(agent.env)
    end

    step += 1.0 
    steps_per_episode += 1.0
end

if !isnothing(animated) 
    write(animated * ".txt", str)
end
end


# tests
using DataFrames
R = 20;
metrics = Dict();
metrics[:score] = DataFrame();
metrics[:steps_per_ep] = DataFrame();
for _=1:R
    agent_Q = Agent(env,:Qlearning);
    run_learning!(agent_Q, 250_000)
    #println("agent_Q: $(agent_Q.score)")

    agent_Q_UCB = Agent(env,:Qlearning; use_UCB = true);
    run_learning!(agent_Q_UCB, 250_000)
    #println("agent_Q_UCB: $(agent_Q_UCB.score)")
    
    agent_SARSA = Agent(env,:SARSA);
    run_learning!(agent_SARSA, 250_000)
    #println("agent_SARSA: $(agent_SARSA.score)")

    agent_SARSA_UCB = Agent(env,:SARSA, use_UCB = true);
    run_learning!(agent_SARSA_UCB, 250_000)
    #println("agent_SARSA_UCB: $(agent_SARSA_UCB.score)")

    # save metrics
    push!(metrics[:score],(Q=agent_Q.score,Q_UCB=agent_Q_UCB.score,SARSA=agent_SARSA.score,SARSA_UCB=agent_SARSA_UCB.score))
    push!(metrics[:steps_per_ep],(Q=agent_Q.steps_per_episode[end],Q_UCB=agent_Q_UCB.steps_per_episode[end],SARSA=agent_SARSA.steps_per_episode[end],SARSA_UCB=agent_SARSA_UCB.steps_per_episode[end]))
end