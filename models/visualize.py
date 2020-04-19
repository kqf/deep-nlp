
import bokeh.models as bm
import bokeh.plotting as pl
from bokeh.io import output_notebook

from sklearn.manifold import TSNE
from sklearn.preprocessing import scale


def draw_vectors(x, y, radius=10, alpha=0.25, color="blue",
                 width=600, height=400, show=True, **kwargs):
    output_notebook()

    if isinstance(color, str):
        color = [color] * len(x)
    data_source = bm.ColumnDataSource(
        {"x": x, "y": y, "color": color, **kwargs})

    fig = pl.figure(active_scroll="wheel_zoom", width=width, height=height)
    fig.scatter("x", "y", size=radius, color="color",
                alpha=alpha, source=data_source)

    fig.add_tools(bm.HoverTool(
        tooltips=[(key, "@" + key) for key in kwargs.keys()]))
    if show:
        pl.show(fig)
    return fig


def get_tsne_projection(word_vectors):
    tsne = TSNE(n_components=2, verbose=100)
    return scale(tsne.fit_transform(word_vectors))


def visualize_embeddings(embeddings, token, colors):
    tsne = get_tsne_projection(embeddings)
    draw_vectors(tsne[:, 0], tsne[:, 1], color=colors, token=token)


from IPython.core.display import HTML
import matplotlib.pyplot as plt


def get_color_hex(weight):
    cmap = plt.get_cmap("RdYlGn")
    rgba = cmap(weight, bytes=True)
    return "#%02X%02X%02X" % rgba[:3]


def visualize(word, weights):
    symb_template = '<span style="background-color: {color_hex}">{symb}</span>'
    res = ""
    for symb, weight in zip(word, weights):
        res += symb_template.format(color_hex=get_color_hex(weight), symb=symb)
    res = "<p>" + res + "</p>"
    return HTML(res)
