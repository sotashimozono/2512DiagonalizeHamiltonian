
const p_init = plot(;
    aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=:bottomright
)

function plot_setup(lat::Lattice; title="", ms_scale=200.0)
    ms = find_marker_size(lat; ms_scale=ms_scale)
    p = plot(;
        aspect_ratio=:equal, grid=false, axis=false, ticks=false, legend=false, title=title
    )
    visualize_bonds(p, lat)
    return p, ms
end
function find_marker_size(lat::Lattice; ms_scale=220.0)
    min_dist = 0.0
    if isempty(lat.bonds)
        min_dist = 1.0
    else
        min_dist = minimum([
            norm(lat.positions[b.src] - lat.positions[b.dst]) for b in lat.bonds
        ])
    end
    xs = [p[1] for p in lat.positions]
    ys = [p[2] for p in lat.positions]
    area = [(maximum(xs) - minimum(xs)), (maximum(ys) - minimum(ys))]
    scaling = min_dist / norm(area)
    marker_size = ms_scale * scaling
    return marker_size
end
function visualize_bonds(p, lat::Lattice)
    threhold = 1.5 * maximum(norm.(lat.unit_cell.basis))
    seg_x, seg_y = Float64[], Float64[]
    for bond in lat.bonds
        src_pos = lat.positions[bond.src]
        dst_pos = lat.positions[bond.dst]
        if norm(dst_pos - src_pos) < threhold
            push!(seg_x, src_pos[1], dst_pos[1], NaN)
            push!(seg_y, src_pos[2], dst_pos[2], NaN)
        end
    end
    return plot!(p, seg_x, seg_y; color=:black, lw=1.0, label="")
end
function get_wf_colors(amplitudes::AbstractVector)
    norm_amps = amplitudes ./ maximum(amplitudes)
    return [cgrad(:viridis)[a] for a in norm_amps]
end

function plot_eigenstate!(p, lat, vecs, n; ms=10.0)
    probs = abs2.(vecs[:, n])
    colors = get_wf_colors(probs)

    xs = [pos[1] for pos in lat.positions]
    ys = [pos[2] for pos in lat.positions]
    return scatter!(p, xs, ys; ms=ms, mc=colors, markerstrokewidth=0, label="")
end

Ly = 40
Lx = Ly

# parameters
t = 1.0
W_list = [w for w in 0.5:0.5:8.0]
W = 4.0
plot_list = []

# 1. 関数定義
p_poisson(r) = 2 / (1 + r)^2
p_gue(r) = (81 * sqrt(3) / (4 * pi)) * ((r + r^2)^2 / (1 + r + r^2)^4)

r_range = 0:0.01:1.0

for lattice_type in AVAILABLE_LATTICES
    lat = build_lattice(lattice_type, Lx, Ly)

    H = zeros(ComplexF64, lat.N, lat.N)
    for i in 1:(lat.N)
        H[i, i] = W * (rand() - 0.5)
        for j in lat.nearest_neighbors[i]
            H[i, j] = -t
        end
    end
    H = Hermitian(H)
    values = eigvals(H)
    spacings = diff(sort(real.(values)))
    spacing_min = [min(spacings[i], spacings[i + 1]) for i in 1:(length(spacings) - 1)]
    spacing_max = [max(spacings[i], spacings[i + 1]) for i in 1:(length(spacings) - 1)]
    rn = spacing_min ./ spacing_max

    p = histogram(
        rn;
        bins=100,
        normalize=:pdf,
        title="$(lattice_type)Lattice-N=$(lat.N)",
        xlabel="Energy",
        ylabel="Density of State",
        legend=false,
    )
    plot!(p, r_range, p_poisson.(r_range); label="Poisson", color=:red, lw=2)
    plot!(p, r_range, p_gue.(r_range); label="GUE", color=:blue, lw=2, linestyle=:dash)
    push!(plot_list, p)
end
final_plot = plot(
    plot_list...;
    layout=(2, 4),
    size=(1600, 1000),
    plot_title="PBC-Lx_$(Lx)Ly_$(Ly)",
    margin=5mm,
)
display(final_plot)
