using DataFrames, CSV, Statistics
function data_setup(
	data::DataFrame, S_col::Union{String, Symbol}, 
	T_col::Union{String, Symbol}, D_col::Union{String, Symbol}
	)

  select!(groupby(data, S_col), :, D_col => maximum => :tunit)
  data.ty = @. ifelse(data[:, D_col] == 0, nothing, data[:, T_col])
  select!(groupby(data, S_col), :, :ty => minnothing => :tyear)
  sort!(data, [T_col, :tunit, S_col])

  return data
end

function projected(data, Y_col, S_col, T_col, covariates)

  k = size(covariates, 1)
  X = Matrix(data[:, covariates])
  y = data[:, Y_col]

  # Pick non-treated
  df_c = data[isnothing.(data.tyear), :]

  # One-hot encoding for T_col and S_col
  select!(df_c, :, [S_col => ByRow(isequal(v)) => Symbol(v) for v in unique(df_c[:, S_col])[2:end]])
  select!(df_c, :, [T_col => ByRow(isequal(v)) => Symbol(v) for v in unique(df_c[:, T_col])[2:end]])
  o_h_cov = Symbol.([covariates; unique(df_c[:, S_col])[2:end]; unique(df_c[:, T_col])[2:end]])

  # Create X_c Matrix with covariates, one-hot encoding for T_col and S_col. Create Y_c vector
  y_c = df_c[:, Y_col]
  X_c = Matrix(df_c[:, o_h_cov])

  # OLS of Y_c on X_c, get β
  XX = [X_c ones(size(X_c, 1))]' * [X_c ones(size(X_c, 1))]
  Xy = [X_c ones(size(X_c, 1))]' * y_c
  all_beta = inv(XX) * Xy
  beta = all_beta[1:k]

  # Calculate adjusted Y
  Y_adj = y - X * beta
  
  # Output projected data
  data[:, Y_col] = Y_adj
  return data
end

function minnothing(x)
  x = x[.!isnothing.(x)]
  if length(x) == 0
    return nothing
  end
  return minimum(x)
end

function find_treat(W)
  N1 = 0
  for row in eachrow(W)
    if 1 in row
      N1 += 1
    end
  end
  return N1
end


function collapse_form(Y::Matrix, N0::Int64, T0::Int64)
  N, T = size(Y)
  Y_T0N0 = Y[1:N0, 1:T0]
  Y_T1N0 = mean(Y[1:N0, T0 + 1:end], dims = 2)
  Y_T0N1 = mean(Y[N0 + 1:end, 1:T0], dims = 1)
  Y_T1N1 = mean(Y[N0 + 1:end, T0 + 1:end])

  return [Y_T0N0 Y_T1N0; Y_T0N1 Y_T1N1]
end

# function pairwise_sum_decreasing(x::Vector{Number}, y::Vector{Number})
function pairwise_sum_decreasing(x::Vector, y::Vector)
  na_x = isnan.(x)
  na_y = isnan.(y)
  x[na_x] .= minimum(x[.!na_x])
  y[na_y] .= minimum(y[.!na_y])
  pairwise_sum = x .+ y
  pairwise_sum[na_x.&na_y] .= NaN
  return pairwise_sum
end

mutable struct random_walk
  Y::Matrix
  n0::Number
  t0::Number
  L::Matrix
end

function random_low_rank()
  n0 = 100
  n1 = 10
  t0 = 120
  t1 = 20
  n = n0 + n1
  t = t0 + t1
  tau = 1
  sigma = 0.5
  rank = 2
  rho = 0.7
  var = [rho^(abs(x - y)) for x in 1:t, y in 1:t]
  W = Int.(1:n .> n0) * transpose(Int.(1:t .> t0))

  # U = rand(Poisson(sqrt.(1:n) ./ sqrt(n)), n, rank)
  pU = Poisson(sqrt(sample(1:n)) ./ sqrt(n))
  pV = Poisson(sqrt(sample(1:t)) ./ sqrt(t))
  U = rand(pU, n, rank)
  V = rand(pV, t, rank)

  # sample.(1:n)

  alpha = reshape(repeat(10 * (1:n) ./ n, outer=(t, 1)), n, t)
  beta = reshape(repeat(10 * (1:t) ./ t, outer=(n, 1)), n, t)
  mu = U * V' + alpha + beta
  error = rand(pV, size(mu))
  Y = mu .+ tau .* W .+ sigma .* error
  random_data = random_walk(Y, n0, t0, mu)
  return random_data
end

function fw_step(
	A::Matrix, b::Vector, x::Vector; eta::Number, 
	alpha::Union{Nothing,Float64} = nothing
	)::Vector{Float64}
  Ax = A * x
  half_grad = (Ax .- b)' * A + eta * x'
  i = findmin(half_grad)[2][2]
  if !isnothing(alpha)
    x *= (1 - alpha)
    x[i] += alpha
    return x
  else
    d_x = -x
    d_x[i] = 1 - x[i]
    if all(d_x .== 0)
      return x
    end
    d_err = A[:, i] - Ax
    step_upper = -half_grad * d_x
    step_bot = sum(d_err .^ 2) + eta * sum(d_x .^ 2)
    step = step_upper[1] / step_bot
    constrained_step = min(1, max(0, step))
    return x + constrained_step * d_x
  end
end

function sc_weight_fw(
	A::Matrix, b::Vector, x::Union{Vector, Nothing} = nothing; 
	intercept::Bool = true, zeta::Number,
	min_decrease::Number = 1e-3, max_iter::Int64 = 1000
	)
  
  k = size(A, 2)
  n = size(A, 1)
  if isnothing(x)
    x = fill(1 / k, k)
  end
  if intercept
    A = A .- mean(A, dims = 1)
    b = b .- mean(b, dims = 1)
  end

  t = 0
  vals = zeros(max_iter)
  eta = n * real(zeta ^ 2)
  while (t < max_iter) && (t < 2 || vals[t-1] - vals[t] > min_decrease ^ 2)
    t += 1
    x_p = fw_step(A, b, x, eta = eta)
    x = x_p
    err = A * x - b
    vals[t] = real(zeta^2) * sum(x .^ 2) + sum(err .^ 2) / n
  end
  Dict("params" => x, "vals" => vals)
end;


function sparsify_function(v::Vector)
  v[v .<= maximum(v) / 4] .= 0
  return v ./ sum(v)
end


function california_prop99()
    url = "https://github.com/d2cml-ai/Synthdid.jl/raw/stag_treat/data/california_prop99.csv"
    return CSV.read(download(url), delim=";", DataFrame)
end

function quota()
    url = "https://github.com/d2cml-ai/Synthdid.jl/raw/stag_treat/data/quota.csv"
	return CSV.read(download(url), DataFrame)
end

cal = california_prop99()
quot = quota()

Y_col, S_col, T_col, D_col = "PacksPerCapita", "State", "Year", "treated"

Y_col = Symbol(Y_col)
S_col = Symbol(S_col)
T_col = Symbol(T_col)
D_col = Symbol(D_col)



tdf = data_setup(cal, S_col, T_col, D_col)

tyears = sort(unique(tdf.tyear)[.!isnothing.(unique(tdf.tyear))])
T_total = 0


year = tyears[1]

df_y = tdf[in.(tdf.tyear, Ref([year, nothing])), [Y_col, S_col, T_col, :tunit]]

N1 = size(unique(df_y[df_y.tunit .== 1, S_col]), 1)
T1 = maximum(tdf[:, T_col]) - year + 1
T_total += N1 * T1
T_post = N1 * T1

# create Y matrix and collapse it
Y = Matrix(unstack(df_y, S_col, T_col, Y_col)[:, 2:end])
N = size(Y, 1)
T = size(Y, 2)
N0 = N - N1
T0 = T - T1
Yc = collapse_form(Y, N0, T0)

# calculate penalty parameters
prediff = diff(Y[1:N0, 1:T0], dims = 2)
noise_level = std(diff(Y[1:N0, 1:T0], dims = 2)) # gotta fix this, probably its own function
eta_omega = ((size(Y, 1) - N0) * (size(Y, 2) - T0))^(1 / 4)
eta_lambda = 1e-6
zeta_omega = eta_omega * noise_level
zeta_lambda = eta_lambda * noise_level
min_decrease = 1e-5 * noise_level




function varianza(x)
    n = length(x)
    media = sum(x) / n
    return sum([(xi - media) ^ 2 for xi in x]) / (n - 1)
end


sqrt(varianza([1, 2, -4, 5, 9]))

varianza(prediff)