using DifferentialEquations, LaTeXStrings, Plots
theme(:wong, lw = 2,
     size = (600,400),
     fontfamily = "Computer Modern")

include("model.jl")

# Output
S₁ = Float64[]
I₁ = Float64[]
v₁ = Float64[]
S₂ = Float64[]
I₂ = Float64[]
v₂ = Float64[]

function sweep(tag,vals)
    # 1% initially infected
    u₀ = [0.4; 0.1; 0.01; 0.5; 0.0; 0.01]
    # Initialize storage
    S₁ = Float64[]
    I₁ = Float64[]
    v₁ = Float64[]
    S₂ = Float64[]
    I₂ = Float64[]
    v₂ = Float64[]
    for val ∈ vals
        q = merge(p,NamedTuple{(Symbol(tag),)}([val]))
        prob = ODEProblem(sipv!, u₀, tspan, q)
        sol = solve(prob,saveat=0.1)
        push!(S₁,sum(sol[1,1:end])/(10*T))
        push!(I₁,sum(sol[2,1:end])/(10*T))
        push!(v₁,sum(sol[3,1:end])/(10*T))
        push!(S₂,sum(sol[4,1:end])/(10*T))
        push!(I₂,sum(sol[5,1:end])/(10*T))
        push!(v₂,sum(sol[6,1:end])/(10*T))
    end
    return [S₁,I₁,v₁,S₂,I₂,v₂]
end

theta1_vals = 0.0:0.00001:0.1
theta1_sweep = sweep("θ₁",theta1_vals)
sweep_theta1_p1 = plot(theta1_vals,[theta1_sweep[1],theta1_sweep[4]],label=[L"S_1" L"S_2"],xlabel=L"\theta_1",legend=:outerright)
sweep_theta1_p2 = plot(theta1_vals,[theta1_sweep[2],theta1_sweep[5]],label=[L"I_1" L"I_2"],xlabel=L"\theta_1",legend=:outerright)
sweep_theta1_p3 = plot(theta1_vals,[theta1_sweep[3],theta1_sweep[6]],label=[L"v_1" L"v_2"],xlabel=L"\theta_1",legend=:outerright)

plot(sweep_theta1_p1,sweep_theta1_p2,sweep_theta1_p3, layout = grid(3, 1), size = (350,600))

savefig("sweep_theta1_plot_omega01.pdf")

rho1_vals = 0.0:0.00001:0.02
rho1_sweep = sweep("ρ₁",rho1_vals)
sweep_rho1_p1 = plot(rho1_vals,[rho1_sweep[1],rho1_sweep[4]],label=[L"S_1" L"S_2"],xlabel=L"\rho_1",legend=:outerright)
sweep_rho1_p2 = plot(rho1_vals,[rho1_sweep[2],rho1_sweep[5]],label=[L"I_1" L"I_2"],xlabel=L"\rho_1",legend=:outerright)
sweep_rho1_p3 = plot(rho1_vals,[rho1_sweep[3],rho1_sweep[6]],label=[L"v_1" L"v_2"],xlabel=L"\rho_1",legend=:outerright)

plot(sweep_rho1_p1,sweep_rho1_p2,sweep_rho1_p3, layout = grid(3, 1), size = (350,600))

savefig("sweep_rho1_plot_omega01.pdf")

phi1_vals = 0.0:0.001:1.0
phi1_sweep = sweep("ρ₁",phi1_vals)
sweep_phi1_p1 = plot(phi1_vals,[phi1_sweep[1],phi1_sweep[4]],label=[L"S_1" L"S_2"],xlabel=L"\phi",legend=:outerright)
sweep_phi1_p2 = plot(phi1_vals,[phi1_sweep[2],phi1_sweep[5]],label=[L"I_1" L"I_2"],xlabel=L"\phi",legend=:outerright)
sweep_phi1_p3 = plot(phi1_vals,[phi1_sweep[3],phi1_sweep[6]],label=[L"v_1" L"v_2"],xlabel=L"\phi",legend=:outerright)

plot(sweep_phi1_p1,sweep_phi1_p2,sweep_phi1_p3, layout = grid(3, 1), size = (350,600))

savefig("sweep_phi1_plot_omega01.pdf")