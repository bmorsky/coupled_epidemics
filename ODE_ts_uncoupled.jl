using DifferentialEquations, LaTeXStrings, Plots, Sundials, LSODA

T0 = 0.0
T = 200.0
ω = 0

function sipv!(du, u, p, t)
    α, β, γ, η, θ1, θ2, κ, ρ1, ρ2, ϵ, ψ = p
    S1, I1, v1, S2, I2, v2 = u
    du[1] = α*(ψ-S1-I1) - β*S1*I1/ψ - η*v1*S1
    du[2] = β*S1*I1/ψ - γ*I1
    du[3] = 1/(1+exp(-κ*(β*I1/ψ - ρ1 - θ1*(1-2*((1-ϵ)*v1+ϵ*v2))))) - v1
    du[4] = α*(1-ψ-S2-I2) - β*S2*I2/(1-ψ) - η*v2*S2
    du[5] = β*S2*I2/(1-ψ) - γ*I2
    du[6] = 1/(1+exp(-κ*(β*I2/(1-ψ) - ρ2 - θ2*(1-2*(ϵ*v1+(1-ϵ)*v2))))) - v2
end

u0 = [0.1428, 0.0212/2, 0.09562, 0.1428, 0.0212/2, 0.09562]
p = [0.01,0.5,1/7,1/7,0.01,0.01,300,0.01,0.01,0.0,0.5]
tspan = (0.0, T)

prob = ODEProblem(sipv!, u0, tspan, p)
sol = solve(prob,QNDF(),abstol=1e-10,reltol=1e-10)
plot(sol,ylims=(0.285/2,0.286/2))


# u0 = [0.4; 0.1; 0.01; 0.4; 0.1; 0.01]
# cpal = palette(:tab10)

# p1=plot(sol,idxs = [(0, 1),(0, 4)],thickness_scaling = 1,linewidth=1,
# legend=false,ylabel=L"S",xlabel="",xaxis=nothing,ylims=(-0.01,1),
# xlims=(T0,T),grid=false,
# yticks = ([0,0.5,1],[L"0", L"0.5", L"1"]))

# p2=plot(sol,idxs = [(0, 2),(0, 5)],thickness_scaling = 1,linewidth=1,
# legend=false,ylabel=L"I",xlabel="",xaxis=nothing,ylims=(-0.001,0.05),
# xlims=(T0,T),grid=false,
# yticks = ([0,0.025,0.05],[L"0", L"0.025", L"0.05"]))

# p3=plot(sol,idxs = [(0, 3),(0, 6)],thickness_scaling = 1,linewidth=1,
# legend=false,ylabel=L"v",xlabel=L"t",ylims=(-0.01,1),
# xlims=(T0,T),grid=false,
# yticks = ([0,0.5,1],[L"0", L"0.5", L"1"]))

# plot(p1,p2,p3, layout = grid(3, 1), size = (300,300))

# savefig("cycles_theta1_009.pdf")