### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ ab80fef0-d25e-11ed-1985-f94fc7748511
using DataFrames, CSV, Statistics

# ╔═╡ f6417b5e-cdae-448c-a471-aa137013138b
begin
	function data_setup(
	  data::DataFrame, S_col::Union{String,Symbol},
	  T_col::Union{String,Symbol}, D_col::Union{String,Symbol}
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
	  Y_T1N0 = mean(Y[1:N0, T0+1:end], dims=2)
	  Y_T0N1 = mean(Y[N0+1:end, 1:T0], dims=1)
	  Y_T1N1 = mean(Y[N0+1:end, T0+1:end])
	
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
end

# ╔═╡ 29670cc2-cb14-44dd-a2f8-04b8fca792b8
begin
	function sparsify_function(v::Vector)
	  v[v.<=maximum(v)/4] .= 0
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
end

# ╔═╡ 7a50b044-9789-4bf2-ac55-fbcb54453cf9
begin
	tdf = data_setup(quota(), :country, :year, :quota)
	Y_col, S_col, T_col, D_col = :womparl, :country, :year, :quota
	covariates = [:lngdp]
end

# ╔═╡ 439f3ae0-be72-4032-910e-a6bce3163458
begin
	  tyears = sort(unique(tdf.tyear)[.!isnothing.(unique(tdf.tyear))])
	  T_total = sum(Matrix(unstack(tdf, S_col, T_col, D_col)[:, 2:end]))
	  units = unique(tdf[:, S_col])
	  N_out = size(units, 1)
	
	  tdf_ori = copy(tdf)
	  info_names = ["year", "tau", "weighted_tau", "N0", "T0", "N1", "T1"]
	  year_params = DataFrame([[] for i in info_names], info_names)
	  T_out = size(unique(tdf[:, T_col]), 1)
	  info_beta = []
end

# ╔═╡ aa9c80d9-59b4-415c-bc19-849bc45583cd
for year in tyears
  info = []
  df_y = tdf[in.(tdf.tyear, Ref([year, nothing])), [[Y_col, S_col, T_col, :tunit]; covariates]]
  N1 = size(unique(df_y[df_y.tunit .== 1, S_col]), 1)
  T1 = maximum(tdf[:, T_col]) - year + 1
  T_post = N1 * T1

  # create Y matrix
  Y = Matrix(unstack(df_y, S_col, T_col, Y_col)[:, 2:end])
  N = size(Y, 1)
  T = size(Y, 2)
  global N0 = N - N1
  global T0 = T - T1
  global Yc = collapse_form(Y, N0, T0)

	# create penalty parameters
  noise_level = std(diff(Y[1:N0, 1:T0], dims = 2)) # this needs fixed, maybe in its own function
  eta_omega = ((size(Y, 1) - N0) * (size(Y, 2) - T0))^(1 / 4)
  eta_lambda = 1e-6
  global zeta_omega = eta_omega * noise_level
  global zeta_lambda = eta_lambda * noise_level
  global min_decrease = 1e-5 * noise_level

  # create X vector of matrices
  X = []
  for covar in covariates
	global X_temp = Matrix(unstack(df_y, S_col, T_col, covar)[:, 2:end])
	push!(X, X_temp)
  end
end


# ╔═╡ eb98e437-9cab-479d-a830-32157042ebf3
begin
	lambda_intercept, max_iter_pre_sparsify, sparsify = true, 100, sparsify_function
	omega_intercept = true
	max_iter = 1000
	
end

# ╔═╡ 46735155-a58a-465e-be61-dff16d7b54b8
X = collapse_form(X_temp, N0, T0)

# ╔═╡ fd4a3c7f-6d1f-46c8-8114-07a328f62453
begin
	Y = Yc
	N01, T01 = size(Y) .- 1
end

# ╔═╡ bb049657-5912-4bdc-82a6-8ba9331b40ef
X_temp

# ╔═╡ b1c49fdf-0279-4195-9af2-ed4d8a010f82
begin
	lambda, omega, beta = nothing, nothing, nothing
	  if isnothing(lambda)
	    lambda = repeat([1 / T0], T0)
	  end
	  if isnothing(omega)
	    omega = repeat([1 / N0], N0)
	  end
	  if isnothing(beta)
 		beta = zeros(size(X, 1))
	  end
end

# ╔═╡ ee9d9b87-8e05-48b3-affc-21650ccd8f7b
begin
	update_lambda, update_omega = true, true
	
end

# ╔═╡ 64eebd0e-2fe7-4be5-a631-7025afcbe519


# ╔═╡ ff4edbf2-6d1a-4212-a6bd-fb9460720eb1


# ╔═╡ 8be586aa-313f-4ac0-bebe-e85cb8869ff9


# ╔═╡ ec3cf44a-0b9b-45c2-94a3-a751bd0a48b9


# ╔═╡ 1e9ab538-7730-41d5-8ebd-30a5dcbb6eae
begin
	X1 = [X, X]
	length(X1)
end

# ╔═╡ 60bf50ca-1eed-49de-b933-2b1c98135158
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

# ╔═╡ 9d4f2d87-4a03-4122-ba71-1461a08c9a6d
function update_weights(Y, lambda, omega)

Y_lambda = if lambda_intercept Y[1:N0, :] .- mean(Y[1:N0, :], dims = 1) else Y[1:N0, :] end
if update_lambda lambda = fw_step(Y_lambda[:, 1:T0], Y_lambda[:, T0 + 1], lambda, eta = N0 * real(zeta_lambda ^ 2)) end
err_lambda = Y_lambda * [lambda; -1]

Y_omega = if omega_intercept Y'[1:T0, :] .- mean(Y'[1:T0, :], dims = 1) else Y[:, 1:T0]' end
if update_omega omega = fw_step(Y_omega[:, 1:N0], Y_omega[:, N0 + 1], omega, eta = T0 * real(zeta_omega ^ 2)) end
err_omega = Y_omega * [omega; -1]

val = real(zeta_omega ^ 2) * sum(omega .^ 2) + real(zeta_lambda ^ 2) * sum(lambda .^ 2) + sum(err_omega .^ 2) / T0 + sum(err_lambda .^ 2) / N0

return Dict("val" => val, "lambda" => lambda, "omega" => omega, "err_lambda" => err_lambda, "err_omega" => err_omega)
end


# ╔═╡ ee34841e-2709-416f-94f1-c75c1d0a871e
weights = update_weights(Yc, lambda, omega)

# ╔═╡ d39aa8d6-1f89-43ad-b342-caf1ed45da77
weights["val"] - 21.654189296597696

# ╔═╡ 80820705-195b-4107-a5ec-0aad2dd1c0e7
length(weights["err_lambda"]), length(weights["err_omega"]), size(X1[1])

# ╔═╡ 52ca4744-7852-4b1f-8111-31fdf4a347a1
begin
	beta1 = [1, 2]
	sum(beta1 .* X1)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
CSV = "~0.10.9"
DataFrames = "~1.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.8.2"
manifest_format = "2.0"
project_hash = "5a15f52aa21619e2b75b00b28bf1fd13bab24b22"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "SentinelArrays", "SnoopPrecompile", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "c700cce799b51c9045473de751e9319bdd1c6e94"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "9c209fb7536406834aa938fb149964b985de6c83"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.1"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "7a60c856b9fa189eb34f5f8a6f6b5529b7942957"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.1"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "0.5.2+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "e27c4ebe80e8699540f2d6c805cc12203b614f12"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.20"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f2355693d6778a178ade15952b7ac47a4ff97996"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.20+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "d321bf2de576bf25ec4d3e4360faca399afca282"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.0"

[[deps.Parsers]]
deps = ["Dates", "SnoopPrecompile"]
git-tree-sha1 = "478ac6c952fddd4399e71d4779797c538d0ff2bf"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.5.8"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "a6062fe4063cdafe78f4a0a81cfffb89721b30e7"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.2"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.PrettyTables]]
deps = ["Crayons", "Formatting", "LaTeXStrings", "Markdown", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "548793c7859e28ef026dba514752275ee871169f"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.2.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "77d3c4726515dca71f6d80fbb5e251088defe305"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.3.18"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StringManipulation]]
git-tree-sha1 = "46da2434b41f41ac3594ee9816ce5541c6096123"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits", "Test"]
git-tree-sha1 = "1544b926975372da01227b382066ab70e574a3ec"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.10.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
deps = ["Random", "Test"]
git-tree-sha1 = "94f38103c984f89cf77c402f2a68dbd870f8165f"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.9.11"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.12+3"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.1.1+0"
"""

# ╔═╡ Cell order:
# ╠═ab80fef0-d25e-11ed-1985-f94fc7748511
# ╟─f6417b5e-cdae-448c-a471-aa137013138b
# ╟─29670cc2-cb14-44dd-a2f8-04b8fca792b8
# ╠═7a50b044-9789-4bf2-ac55-fbcb54453cf9
# ╠═439f3ae0-be72-4032-910e-a6bce3163458
# ╠═aa9c80d9-59b4-415c-bc19-849bc45583cd
# ╠═eb98e437-9cab-479d-a830-32157042ebf3
# ╠═46735155-a58a-465e-be61-dff16d7b54b8
# ╠═fd4a3c7f-6d1f-46c8-8114-07a328f62453
# ╠═bb049657-5912-4bdc-82a6-8ba9331b40ef
# ╠═b1c49fdf-0279-4195-9af2-ed4d8a010f82
# ╠═9d4f2d87-4a03-4122-ba71-1461a08c9a6d
# ╟─ee9d9b87-8e05-48b3-affc-21650ccd8f7b
# ╠═ee34841e-2709-416f-94f1-c75c1d0a871e
# ╠═d39aa8d6-1f89-43ad-b342-caf1ed45da77
# ╠═64eebd0e-2fe7-4be5-a631-7025afcbe519
# ╠═ff4edbf2-6d1a-4212-a6bd-fb9460720eb1
# ╠═8be586aa-313f-4ac0-bebe-e85cb8869ff9
# ╠═ec3cf44a-0b9b-45c2-94a3-a751bd0a48b9
# ╠═1e9ab538-7730-41d5-8ebd-30a5dcbb6eae
# ╟─60bf50ca-1eed-49de-b933-2b1c98135158
# ╠═80820705-195b-4107-a5ec-0aad2dd1c0e7
# ╠═52ca4744-7852-4b1f-8111-31fdf4a347a1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
