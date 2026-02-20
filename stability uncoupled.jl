using Roots, LinearAlgebra

# parameters
α = 0.01
β = 0.5
γ = 1/7
η = 1/7
θ = 0.01
κ = 300
ρ = 0.01

BR(I,v) = 1 ./ (1 .+ exp.(-κ*(β*I .- ρ .- θ*(1 .- 2*v))))

dBRI(I,v) = κ*β*exp.(-κ*(β*I .- ρ .- θ*(1 .- 2*v))).*(BR(I,v).^2)
dBRv(I,v) = κ*θ*2*exp.(-κ*(β*I .- ρ .- θ*(1 .- 2*v))).*(BR(I,v).^2)


# endemic equilibria
v̄ₑ = find_zeros(v -> BR((α*(β-γ)-γ*η*v)/(β*(α+γ)),v) - v, (0,1))
Īₑ = (α*(β-γ) .- γ*η*v̄ₑ)/(β*(α+γ))

c₂ = 1 .- dBRv(Īₑ,v̄ₑ) .+ α + β*Īₑ .+ η*v̄ₑ
c₁ = (1 .- dBRv(Īₑ,v̄ₑ)).*(α .+ β*Īₑ .+ η*v̄ₑ) .+ β*(α+γ)*Īₑ
c₀ = (β*(α+γ)*(1 .- dBRv(Īₑ,v̄ₑ)) + γ*η*dBRI(Īₑ,v̄ₑ)).*Īₑ

c₂.*c₁ .- c₀

J = [-α-β*Īₑ[1]-η*v̄ₑ[1] -α-γ -η*γ/β;
    β*Īₑ[1] 0 0;
    0 (dBRI(Īₑ,v̄ₑ)[1]) (dBRv(Īₑ,v̄ₑ)[1]-1)]

eigvals(J)

# disease-free equilibria
v̄₀ = find_zeros(v -> BR(0,v) - v, (0,1))
S̄₀ = α./(α .+ v̄₀)

# eigs = find_zeros(λ -> λ^3 + c₂[1]*λ^2 + c₁[1]*λ + c₀[1], -1000,1000)

# using IntervalArithmetic, IntervalRootFinding
# root(λ -> λ^3 + c₂[1]*λ^2 + c₁[1]*λ + c₀[1], -1000,1000)