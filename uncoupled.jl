using DifferentialEquations, LaTeXStrings, Plots, Sundials, LSODA

T0 = 0.0
T = 2000.0

function sipv!(du, u, p, t)
    α, β, γ, η, θ, κ, ρ = p
    S, I, v = u
    du[1] = -(α*(1-S-I) - β*S*I - η*v*S)
    du[2] = -(β*S*I - γ*I)
    du[3] = -(1/(1+exp(-κ*(β*I - ρ - θ*(1-2*v)))) - v)
end

init = rand(3)

S0=init[1]/sum(init)
I0=init[2]/sum(init)
v0 = rand()

u0 = [2/7, 0.0212, 0.09562]
# u0 = [S0, I0, v0]
p = [0.01,0.5,1/7,1/7,0.01,300,0.01]
tspan = (0.0, T)

prob = ODEProblem(sipv!, u0, tspan, p)
sol = solve(prob,QNDF(),abstol=1e-10,reltol=1e-10)
plot(sol)