using Random

function split_data(x, y, train_perc)
    n = size(x, 2)
    idx = 1:n
    train_idx = view(idx, 1:floor(Int, train_perc * n))
    test_idx = view(idx, (floor(Int, train_perc * n) + 1):n)
    x_train = x[:, train_idx]
    y_train = y[:, train_idx]
    x_test = x[:, test_idx]
    y_test = y[:, test_idx]
    return x_train, y_train, x_test, y_test
end

function data_windows(u, n, in_out_nodes, stride = 1)
    c = size(u, 1)
    timesteps = div(in_out_nodes, c)
    x = zeros(Float32, in_out_nodes, n)
    for i in 1:n
        t = (i - 1) * stride + 1
        x[:, i] = vec(u[:, t:t + timesteps - 1])
    end
    return x
end

function pair_data(u, predict_length, n_samples, in_out_nodes, stride = 1)
    x = data_windows(u[:, 1:end-predict_length], n_samples, in_out_nodes, stride)
    x = (x .- minimum(x))/(maximum(x) - minimum(x))
    y = data_windows(u[:, 1+predict_length:end], n_samples, in_out_nodes, stride)
    y = (y .- minimum(y))/(maximum(y) - minimum(y))
    return Dict("x" => x, "y" => y)
end

function augment_data(x, y, noise_level=0.01f0, scale_range=(0.9f0, 1.1f0))
    n_samples = size(x, 2)

    # Noise Injection
    noise = randn(Float32, size(x)) .* noise_level
    x_noisy = x .+ noise
    y_noisy = y .+ noise

    # Scaling
    scales = rand(scale_range[1]:0.01f0:scale_range[2], n_samples)'
    x_scaled = x .* scales
    y_scaled = y .* scales

    return x_noisy, y_noisy, x_scaled, y_scaled
end