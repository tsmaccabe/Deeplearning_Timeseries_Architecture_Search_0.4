using DifferentialEquations, JLD2, Random

# Define the Lorenz system
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Function to simulate and save data
function save_lorenz(path, total_iterations, chunk_size=500000, dt=0.05; u0 = rand(3) .- 0.5, params = [10f0, 28f0, 8f0/3f0])
    tspan = (0.0, total_iterations*dt)  # Total time span
    prob = ODEProblem(lorenz!, u0, tspan, params)
    sol = solve(prob, Tsit5(), saveat=dt)

    keys = String[]
    datasets = Dict[]
    for i in 1:chunk_size:total_iterations
        end_i = min(i + chunk_size - 1, total_iterations)
        key = "lorenz_raw_$(i)_to_$(end_i)"
        push!(keys, key)
        u_array = hcat(sol.u[i:end_i]...)
        data = Dict("u" => u_array, "t" => sol.t[i:end_i])
        push!(datasets, data)
        register_data(key, path)
    end
    save_datasets(datasets, keys)
end

# Function to load the first N chunks and unpack the data
function load_lorenz(n_chunks, chunk_size)
    keys = String[]
    for i in 1:n_chunks
        start_i = (i - 1) * chunk_size + 1
        end_i = i * chunk_size
        key = "lorenz_raw_$(start_i)_to_$(end_i)"
        push!(keys, key)
    end

    return load_datasets(keys)
end