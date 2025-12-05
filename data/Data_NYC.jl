using CSV
using DataFrames
using Statistics
using Plots
using Distributions
using NPZ

I = 20 
T_horizon = 72
T_start = 72
days = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27, 28]
days_march = [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 25, 26, 27, 28, 29]

D = length(days) 
D_march = length(days_march) 

df = CSV.read("df_20.csv", DataFrame; delim = ',', header = true)
df_march = CSV.read("df_20_march.csv", DataFrame; delim = ',', header = true)

rename!(df,:"five minutes" => :five_minutes)
rename!(df_march,:"five minutes" => :five_minutes)


#Arrival rates

arrival_rates_20loc = zeros(I, I, T, D)

l = 0
sumation=0
for d in days
    println(d)
    l += 1
    w = df[(df.day  .== d),:] 
    for i in 1:I
        x = w[(w.PULocationID  .== i-1),:] 
        for j in 1:I
            y = x[(x.DOLocationID  .== j-1),:]
            for t in 1:T_horizon
                z = y[(y.five_minutes .== T_start-1+t),:]
                s = size(z)[1]
                arrival_rates_20loc[i,j,t,l]=s      
                sumation += s     
            end
        end
    end
end

arrival_rates_20loc_march = zeros(I, I, T, D_march)

l = 0
sumation=0
for d in days_march
    println(d)
    l += 1
    w = df_march[(df_march.day  .== d),:] 
    for i in 1:I
        x = w[(w.PULocationID  .== i-1),:] 
        for j in 1:I
            y = x[(x.DOLocationID  .== j-1),:]
            for t in 1:T_horizon
                z = y[(y.five_minutes .== T_start-1+t),:]
                s = size(z)[1]
                arrival_rates_20loc_march[i,j,t,l]=s      
                sumation += s     
            end
        end
    end
end

#Travel time 

average_time_duration_20loc = zeros(I, I)

for i in 1:I
    for j in 1:I
        x = df[(df.PULocationID  .== i-1) .& (df.DOLocationID  .== j-1),:trip_time] 
        y = mean(x)
        average_time_duration_20loc[i,j] = y/300 
    end
end

#Transit distance 

average_distance_20loc = zeros(I, I)

for i in 1:I
    for j in 1:I
        x = df[(df.PULocationID  .== i-1) .& (df.DOLocationID  .== j-1),:trip_miles] 
        y = mean(x)
        average_distance_20loc[i,j] = y
    end
end

#Entry rate 

function normal_dist(x, µ, σ)
    2 * exp.(-(x .- µ).^2 / (2 * σ^2))
end

μ = T/2 # can be used to adjust the peak time
σ = 10  # can be used to flatten the curve
x = vcat(range(1, stop=Int(T_horizon/2), length=Int(T_horizon/2)), range(Int((T_horizon/2)+1), stop=T_horizon, length=Int(T_horizon/2)))
y = normal_dist(x, µ, σ) * 10
entry_points = [1,4,11,13,15,19,20]
nr_entry = length(entry_points)
entry_rates_20loc = zeros(I, T)
for i in 1:I
    for t in 1:T
        if i in entry_points
            entry_rates_20loc[i, t] = y[t]/nr_entry
        end
    end
end

npzwrite("entry_rates_20loc.npy", entry_rates_20loc)
npzwrite("average_distances_20loc.npy", average_distance_20loc)
npzwrite("average_time_durations_20loc.npy", average_time_duration_20loc)
npzwrite("arrival_rates_20loc.npy", arrival_rates_20loc)
npzwrite("arrival_rates_20loc_march.npy", arrival_rates_20loc_march)

##Plots

z = df[:,:trip_miles]
histogram(z, normalize=:pdf, xlim=(0,25))
# Define the Erlang-2 distribution
λ = 2.0/(mean(z))
# Rate parameter
erlang_dist = Gamma(2, 1/λ)  # Gamma(shape = 2, scale = 1/λ)

# Define the range of x values and compute the PDF
x = 0:0.1:25
y = pdf.(erlang_dist, x)

# Create the plot
plot!(x, y, label="Erlang-2 PDF (λ = $λ)", lw=2, color=:purple, xlabel="x", ylabel="Density", title="Erlang-2 Distribution")


plot(x[1:72],y[1:72], yticks=[0,5,10,15,20], xticks=([1, 13, 25, 37, 49, 61, 72], [0, 12, 24, 36, 48, 60, 71]), xlabel="Decision Epochs", ylabel="Driver Entry Rate", label=false, color=:blue, lw=2.0,bottom_margin=7Plots.mm)
savefig("arrival_rate_drivers.pdf")