import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm, colormaps


def get_mlp_min_max_width(mlp):
    """Gets the min and max width of an MLP in sklearn."""
    widths = [layer.shape[0] for layer in mlp.coefs_]
    widths.append(mlp.n_outputs_)
    min_width = min(widths)
    max_width = max(widths)
    return (min_width, max_width)


def print_ascii_mlp(mlp):
    """Prints an MLP as ASCII"""
    _, max_width = get_mlp_min_max_width(mlp)
    number_of_spaces = max_width * 3 + 5
    widths = [layer.shape[0] for layer in mlp.coefs_]
    widths.append(mlp.n_outputs_)
    for layer_number, width in enumerate(widths):
        layer_string = " . " * width
        print(
            f"Layer {layer_number:3}: {layer_string.center(number_of_spaces)}"
        )


def build_neurons_dataframe(mlp):
    """Builds a DataFrame holding information about the neurons in an MLP"""
    weights = mlp.coefs_
    _, max_width = get_mlp_min_max_width(mlp)

    widths = [layer.shape[0] for layer in weights]
    widths.append(mlp.n_outputs_)

    neurons_df = pd.DataFrame(
        columns=["layer_number", "neuron_number", "x", "y"]
    )

    for from_layer_number, from_layer_width in enumerate(widths):
        for neuron_number in range(from_layer_width):
            x_coordinate = from_layer_number
            y_coordinate = neuron_number + max_width / 2 - from_layer_width / 2
            neuron_information = {
                "layer_number": from_layer_number,
                "neuron_number": neuron_number,
                "x": x_coordinate,
                "y": y_coordinate,
            }
            neurons_df.loc[len(neurons_df)] = neuron_information
    return neurons_df


# Build the weights DataFrame
def build_weights_dataframe(mlp, neurons_df):
    """Builds a DataFrame holding the weights of an sklearn MLP."""
    weights = mlp.coefs_

    weights_df = pd.DataFrame(
        columns=["from_layer_number", "to_layer_number", "x", "y", "weight"]
    )

    widths = [layer.shape[0] for layer in weights]
    widths.append(mlp.n_outputs_)

    for from_layer_number, from_layer_width in enumerate(widths[:-1]):
        to_layer_number = from_layer_number + 1
        width_next_layer = widths[to_layer_number]
        layer_weights = weights[from_layer_number]
        for neuron_index_this_layer in range(from_layer_width):
            from_layer_mask = neurons_df["layer_number"] == from_layer_number
            to_layer_mask = neurons_df["layer_number"] == to_layer_number
            from_neuron_mask = (
                neurons_df["neuron_number"] == neuron_index_this_layer
            )
            from_neuron = neurons_df[from_layer_mask & from_neuron_mask]

            for neuron_index_next_layer in range(width_next_layer):
                to_neuron_mask = (
                    neurons_df["neuron_number"] == neuron_index_next_layer
                )
                to_neuron = neurons_df[to_layer_mask & to_neuron_mask]
                x_coordinates = [from_neuron.x, to_neuron.x]
                y_coordinates = [from_neuron.y, to_neuron.y]
                weight = layer_weights[
                    neuron_index_this_layer, neuron_index_next_layer
                ]
                weight_information = {
                    "from_layer_number": from_layer_number,
                    "to_layer_number": to_layer_number,
                    "x": x_coordinates,
                    "y": y_coordinates,
                    "weight": weight,
                }
                weights_df.loc[len(weights_df)] = weight_information

    return weights_df


def build_colormap(weights_df, colormap_name):
    """Build a colormap and scale the weights to match the colormap"""
    weight_values_flat_array = weights_df.weight.to_numpy()
    min_weight = min(weight_values_flat_array)
    max_weight = max(weight_values_flat_array)
    norm = plt.Normalize(min_weight, max_weight)
    cmap = colormaps[colormap_name]
    colors = cmap(norm(weight_values_flat_array))
    weights_df["colors"] = colors.tolist()
    return cmap, norm


def plot_weights(weights_df, linewidth, line_alpha):
    """Plots the weights from the weights_df."""
    for _, row in weights_df.iterrows():
        plt.plot(
            row.x,
            row.y,
            color=row.colors,
            linewidth=linewidth,
            alpha=line_alpha,
            zorder=0,
        )


def plot_neurons(neurons_df, neuron_color, neuron_size, neuron_alpha):
    """Plots the neurons from the neurons DataFrame."""
    plt.scatter(
        neurons_df.x,
        neurons_df.y,
        color=neuron_color,
        s=neuron_size,
        alpha=neuron_alpha,
        zorder=1,
    )


def build_dataframes(mlp):
    """Builds DataFrames holding the neurons and weights of an MLP."""
    neurons_df = build_neurons_dataframe(mlp)
    weights_df = build_weights_dataframe(mlp, neurons_df)
    return neurons_df, weights_df


def generate_plots(
    neurons_df,
    weights_df,
    figure_width,
    figure_height,
    linewidth,
    neuron_alpha,
    neuron_color,
    line_alpha,
    neuron_size,
    colormap_name,
):
    """Generates the plots to visualize an MLP."""
    fig = plt.figure(figsize=(figure_width, figure_height))
    cmap, norm = build_colormap(weights_df, colormap_name)
    plot_weights(weights_df, linewidth, line_alpha)
    plot_neurons(neurons_df, neuron_color, neuron_size, neuron_alpha)
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=plt.gca())
    plt.axis("off")
    plt.show()
    return fig


def display_mlp(
    mlp,
    linewidth=1.5,
    neuron_alpha=1,
    neuron_color="black",
    line_alpha=0.8,
    figure_width=40,
    figure_height=6,
    neuron_size=30,
    colormap_name="Greys",
):
    """Displays the weights of an MLP. Does not plot intercepts (biases)."""
    neurons_df, weights_df = build_dataframes(mlp)
    fig = generate_plots(
        neurons_df,
        weights_df,
        figure_width,
        figure_height,
        linewidth,
        neuron_alpha,
        neuron_color,
        line_alpha,
        neuron_size,
        colormap_name,
    )
    return fig
