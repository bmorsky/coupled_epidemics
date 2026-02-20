using DifferentialEquations, LaTeXStrings, Plots

T0 = 0.0
T = 1000.0
ω = 0

function sipv!(du, u, p, t)
    α, β, γ, ϵ, θ1, θ2, κ, ρ1, ρ2, ϕ, ψ = p
    S1, I1, v1, S2, I2, v2 = u
    du[1] = α*(ψ-S1-I1) - β*S1*(I1+ϕ*I2) - ϵ*v1*S1
    du[2] = β*S1*(I1+ϕ*I2) - γ*I1
    du[3] = 10000*(1/(1+exp(-κ*(β*(I1+ϕ*I2) - ρ1 - θ1*(1-2*(v1+ω*v2)/(1+ω))))) - v1)
    du[4] = α*(1-ψ-S2-I2) - β*S2*(ϕ*I1+I2) - ϵ*v2*S2
    du[5] = β*S2*(ϕ*I1+I2) - γ*I2
    du[6] = 10000*(1/(1+exp(-κ*(β*(ϕ*I1+I2) - ρ2 - θ2*(1-2*(ω*v1+v2)/(1+ω))))) - v2)
end

u0 = [0.4; 0.1; 0.01; 0.5; 0.0; 0.01]
p = [1/100,0.5,1/7,7000,0.01,0.01,1000000,0.01,0.01,0.5,0.5]
tspan = (0.0, T)

prob = ODEProblem(sipv!, u0, tspan, p)
sol = solve(prob, saveat=1)

cpal = palette(:tab10)

p1=plot(sol,idxs = [(0, 1),(0, 4)],thickness_scaling = 1,linewidth=1,
legend=false,ylabel=L"S",xlabel="",xaxis=nothing,ylims=(-0.01,1),
xlims=(T0,T),grid=false,
yticks = ([0,0.5,1],[L"0", L"0.5", L"1"]))

p2=plot(sol,idxs = [(0, 2),(0, 5)],thickness_scaling = 1,linewidth=1,
legend=false,ylabel=L"I",xlabel="",xaxis=nothing,ylims=(-0.001,0.05),
xlims=(T0,T),grid=false,
yticks = ([0,0.025,0.05],[L"0", L"0.025", L"0.05"]))

p3=plot(sol,idxs = [(0, 3),(0, 6)],thickness_scaling = 1,linewidth=1,
legend=false,ylabel=L"v",xlabel=L"t",ylims=(-0.01,1),
xlims=(T0,T),grid=false,
yticks = ([0,0.5,1],[L"0", L"0.5", L"1"]))

plot(p1,p2,p3, layout = grid(3, 1), size = (300,300))

savefig("cycles_theta1_009.pdf")