my_free_truss_node_family = truss_node_family_element(default_truss_node_family(), support=false)
free_node(pt) = truss_node(pt, family=my_free_truss_node_family)
fixed_node(pt) = truss_node(pt, family=fixed_truss_node_family)

space_frame(ptss) =
    let ais = ptss[1],
        bis = ptss[2],
        cis = ptss[3]

        fixed_node(ais[1])
        free_node.(ais[2:end-1])
        fixed_node(ais[end])
        free_node.(bis)
        truss_bar.(ais, cis)
        truss_bar.(bis, ais[1:end-1])
        truss_bar.(bis, cis[1:end-1])
        truss_bar.(bis, ais[2:end])
        truss_bar.(bis, cis[2:end])
        truss_bar.(ais[2:end], ais[1:end-1])
        truss_bar.(bis[2:end], bis[1:end-1])
        if ptss[4:end] == []
            fixed_node(cis[1])
            free_node.(cis[2:end-1])
            fixed_node(cis[end])
            truss_bar.(cis[2:end], cis[1:end-1])
        else
            truss_bar.(bis, ptss[4])
            space_frame(ptss[3:end])
        end
    end

parametric_truss(x11, y11, z11, x12, y12, z12, x13, y13, z13, x21, y21, z21, x22, y22, z22, x31, y31, z31, x32, y32, z32, x41, y41, z41, x42, y42, z42, x51, y51, z51, x52, y52, z52, x53, y53, z53) =
    let p11 = xyz(x11, y11, z11),
        p12 = xyz(x12, y12, z12),
        p13 = xyz(x13, y13, z13),
        p21 = xyz(x21, y21, z21),
        p22 = xyz(x22, y22, z22),
        p31 = xyz(x31, y31, z31),
        p32 = xyz(x32, y32, z32),
        p33 = xyz(x33, y33, z33),
        p41 = xyz(x41, y41, z41),
        p42 = xyz(x42, y42, z42),
        p51 = xyz(x51, y51, z51),
        p52 = xyz(x52, y52, z52),
        p53 = xyz(x53, y53, z53)

        space_frame([[p11, p12, p13],
            [p21, p22],
            [p31, p32, p33],
            [p41, p42],
            [p51, p52, p53]])
    end

fixed_parametric_truss(
    x12, y12, z12,
    x21, y21, z21,
    x22, y22, z22,
    x32, y32, z32,
    x41, y41, z41,
    x42, y42, z42,
    x52, y52, z52) =
    begin
        delete_all_shapes()
        parametric_truss(
            0, 0, 0, x12, y12, z12, 20, 0, 0,
            x21, y21, z21, x22, y22, z22,
            0, 10, 0, x32, y32, z32, 20, 10, 0,
            x41, y41, z41, x42, y42, z42,
            0, 20, 0, x52, y52, z52, 20, 20, 0)
    end

# Helpers

const step_size = 0.01
int2float(x, min, step = step_size) = min + step * x
bounds_coordinates(v, r=0.3) = (v - r, v + r) .* 10

# Materials

const materials_e = [
    1.6409e11,
    1.86e11,
    2.e11,
    2.0684e11,
    2.047e11,
    1.93e11,
]

const materials_cost = [
    460.0,
    1480.0,
    860.0,
    950.0,
    2750.0,
    1825.0,
]

# Variable bounds

r = 0.4
x12_interval = bounds_coordinates(1, r)
y12_interval = bounds_coordinates(0, r)
z12_interval = bounds_coordinates(0, r)
x21_interval = bounds_coordinates(0.5, r)
y21_interval = bounds_coordinates(0.5, r)
z21_interval = bounds_coordinates(1, r)
x22_interval = bounds_coordinates(1.5, r)
y22_interval = bounds_coordinates(0.5, r)
z22_interval = bounds_coordinates(1, r)
x32_interval = bounds_coordinates(1, r)
y32_interval = bounds_coordinates(1, r)
z32_interval = bounds_coordinates(0, r)
x41_interval = bounds_coordinates(0.5, r)
y41_interval = bounds_coordinates(1.5, r)
z41_interval = bounds_coordinates(1, r)
x42_interval = bounds_coordinates(1.5, r)
y42_interval = bounds_coordinates(1.5, r)
z42_interval = bounds_coordinates(1, r)
x52_interval = bounds_coordinates(1, r)
y52_interval = bounds_coordinates(2, r)
z52_interval = bounds_coordinates(0, r)

const n_objs = 2

# Objectives

cost(truss_volume, material) = truss_volume * materials_cost[Int(material)]

objectives(
    material, bar_radius,
    x12, y12, z12, x21, y21, z21, x22, y22, z22,
    x32, y32, z32, x41, y41, z41, x42, y42, z42,
    x52, y52, z52) =
    let b_radius = int2float(bar_radius, 0.035, 0.005),
        load = vz(-3500.0) * 20 * 20
        x12 = int2float(x12, x12_interval[1])
        y12 = int2float(y12, y12_interval[1])
        z12 = int2float(z12, z12_interval[1])
        x21 = int2float(x21, x21_interval[1])
        y21 = int2float(y21, y21_interval[1])
        z21 = int2float(z21, z21_interval[1])
        x22 = int2float(x22, x22_interval[1])
        y22 = int2float(y22, y22_interval[1])
        z22 = int2float(z22, z22_interval[1])
        x32 = int2float(x32, x32_interval[1])
        y32 = int2float(y32, y32_interval[1])
        z32 = int2float(z32, z32_interval[1])
        x41 = int2float(x41, x41_interval[1])
        y41 = int2float(y41, y41_interval[1])
        z41 = int2float(z41, z41_interval[1])
        x42 = int2float(x42, x42_interval[1])
        y42 = int2float(y42, y42_interval[1])
        z42 = int2float(z42, z42_interval[1])
        x52 = int2float(x52, x52_interval[1])
        y52 = int2float(y52, y52_interval[1])
        z52 = int2float(z52, z52_interval[1])
        set_backend_family(
            default_truss_bar_family(), frame3dd,
            frame3dd_truss_bar_family(E=materials_e[Int(material)], G=7.95e10, p=0.0, d=7850.0))
        with_truss_node_family(radius=b_radius * 2.4) do
            with_truss_bar_family(radius=b_radius, inner_radius=b_radius - 0.02) do
                fixed_parametric_truss(x12, y12, z12, x21, y21, z21, x22, y22, z22, x32, y32, z32, x41, y41, z41, x42, y42, z42, x52, y52, z52)
                free_ns = length(filter(!KhepriBase.truss_node_is_supported, frame3dd.truss_nodes))
                truss_volume = truss_bars_volume()
                analysis = truss_analysis(load / free_ns)
                max_disp = max_displacement(analysis)
                [max_disp, cost(truss_volume, material)]
        end
    end
end

function problem(x)
    try
        return (objectives(
            x[1], x[2], x[3], x[4], x[5],
            x[6], x[7], x[8], x[9], x[10],
            x[11], x[12], x[13], x[14], x[15],
            x[16], x[17], x[18], x[19], x[20],
            x[21], x[22], x[23]),
        [0.0], [0.0])
    catch
        return ([Inf, Inf], [0.0], [0.0])
    end
end

# Search space

const n_vars = 23
material_idx = 1:6
bar_radius = 0:8

upper_bound(interval, step_size = step_size) = Int((interval[end] - interval[1]) / step_size)

x12_upper_bound = upper_bound(x12_interval)
y12_upper_bound = upper_bound(y12_interval)
z12_upper_bound = upper_bound(z12_interval)
x21_upper_bound = upper_bound(x21_interval)
y21_upper_bound = upper_bound(y21_interval)
z21_upper_bound = upper_bound(z21_interval)
x22_upper_bound = upper_bound(x22_interval)
y22_upper_bound = upper_bound(y22_interval)
z22_upper_bound = upper_bound(z22_interval)
x32_upper_bound = upper_bound(x32_interval)
y32_upper_bound = upper_bound(y32_interval)
z32_upper_bound = upper_bound(z32_interval)
x41_upper_bound = upper_bound(x41_interval)
y41_upper_bound = upper_bound(y41_interval)
z41_upper_bound = upper_bound(z41_interval)
x42_upper_bound = upper_bound(x42_interval)
y42_upper_bound = upper_bound(y42_interval)
z42_upper_bound = upper_bound(z42_interval)
x52_upper_bound = upper_bound(x52_interval)
y52_upper_bound = upper_bound(y52_interval)
z52_upper_bound = upper_bound(z52_interval)

lb_points = [material_idx[1], bar_radius[1], fill(0, 21)...]
ub_points = [material_idx[end], bar_radius[end],
        x12_upper_bound, y12_upper_bound, z12_upper_bound,
        x21_upper_bound, y21_upper_bound, z21_upper_bound,
        x22_upper_bound, y22_upper_bound, z22_upper_bound,
        x32_upper_bound, y32_upper_bound, z32_upper_bound,
        x41_upper_bound, y41_upper_bound, z41_upper_bound,
        x42_upper_bound, y42_upper_bound, z42_upper_bound,
        x52_upper_bound, y52_upper_bound, z52_upper_bound]

integer_space = BoxConstrainedSpace(lb_points, ub_points)
