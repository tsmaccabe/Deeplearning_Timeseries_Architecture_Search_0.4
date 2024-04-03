### Helper aliases
const SampleShape = Tuple{Vararg{Int}} # Shape of each sample datum
const ShapeVector = Vector{SampleShape}

### ReshapeLayer for reshaping data sample-wise mid-chain
struct ReshapeLayer
    output_shape::SampleShape
end
(Fl::ReshapeLayer)(x) = reshape(x, Fl.output_shape..., size(x, ndims(x)))

### Chain builder funcions
# Empty (Identity) Chain
function build_empty(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    return Chain(x -> x)
end
function build_empty() :: Chain
    return build_empty(SampleShape(), SampleShape(), ParamTuple{0}())
end

# Encoder Chain
function build_encoder(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    (h_layers, activation_fn) = params
    in_nodes = prod(in_shape)
    out_nodes = prod(out_shape)
    linear_interpolate(start_int, end_int, n) = Int.(round.(LinRange(start_int, end_int, n + 2)))

    encoder_layers = []
    encoder_dims = linear_interpolate(in_nodes, out_nodes, h_layers)
    for i in 1:length(encoder_dims)-1
        push!(encoder_layers, Dense(encoder_dims[i], encoder_dims[i+1]))
        if i != length(encoder_dims)-1
            push!(encoder_layers, BatchNorm(encoder_dims[i+1]))
        end
        push!(encoder_layers, activation_fn)
        if i < length(encoder_dims)-2
            push!(encoder_layers, Dropout(drop_rate))
        end
    end

    return Chain(Flux.flatten, encoder_layers..., ReshapeLayer(out_shape))
end

# Decoder Chain
function build_decoder(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    (h_layers, activation_fn) = params
    in_nodes = prod(in_shape)
    out_nodes = prod(out_shape)
    linear_interpolate(start_int, end_int, n) = Int.(round.(LinRange(start_int, end_int, n + 2)))

    decoder_layers = []
    decoder_dims = linear_interpolate(in_nodes, out_nodes, h_layers)
    for i in 1:length(decoder_dims)-1
        push!(decoder_layers, Dense(decoder_dims[i], decoder_dims[i+1]))
        if i != length(decoder_dims)-1
            push!(decoder_layers, BatchNorm(decoder_dims[i+1]))
        end
        push!(decoder_layers, activation_fn)
        if i < length(decoder_dims)-2
            push!(decoder_layers, Dropout(drop_rate))
        end
    end

    return Chain(Flux.flatten, decoder_layers..., ReshapeLayer(out_shape))
end

# Encoder-Decoder Chain
function build_enc_dec(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    (enc_length, dec_length, mid_nodes, activation_fn) = params
    mid_shape = (mid_nodes,)

    encoder_params = (enc_length, activation_fn)
    decoder_params = (dec_length, activation_fn)

    encoder = build_encoder(in_shape, mid_shape, encoder_params; drop_rate = drop_rate)
    decoder = build_decoder(mid_shape, out_shape, decoder_params; drop_rate = drop_rate)

    return Chain(encoder, decoder)
end

function build_fixed(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    (layers, width, activation_fn) = params
    in_nodes = prod(in_shape)
    out_nodes = prod(out_shape)

    layers_list = []
    push!(layers_list, Dense(in_nodes, width))
    push!(layers_list, BatchNorm(width))
    push!(layers_list, activation_fn)
    push!(layers_list, Dropout(drop_rate))
    for i in 2:layers-1
        push!(layers_list, Dense(width, width))
        if i != layers-1
            push!(layers_list, BatchNorm(width))
        end
        push!(layers_list, activation_fn)
        if i < layers-2
            push!(layers_list, Dropout(drop_rate))
        end
    end
    push!(layers_list, Dense(width, out_nodes))

    return Chain(Flux.flatten, layers_list..., ReshapeLayer(out_shape))
end

# Single-Layer Chain
function build_single(in_shapes::ShapeVector, out_shapes::ShapeVector, params::Vector; drop_rate::Float32 = 0f0) :: Chain
    in_shape = in_shapes[1]; out_shape = out_shapes[1]
    (activation_fn,) = params

    in_nodes = prod(in_shape)
    out_nodes = prod(out_shape)

    layers_list = []

    push!(layers_list, Dense(in_nodes, out_nodes))
    push!(layers_list, BatchNorm(width))
    push!(layers_list, activation_fn)
    push!(layers_list, Dropout(drop_rate))

    return Chain(Flux.flatten, layers_list..., ReshapeLayer(output_sample_shape))
end

model_formats = Dict(
    :empty => build_empty,
    :single => build_single,
    :fixed => build_fixed,
    :encoder => build_encoder,
    :decoder => build_decoder,
    :encoder_decoder => build_enc_dec
)