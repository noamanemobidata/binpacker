from shiny import App, ui, render, reactive
import shinyswatch
import plotly.graph_objects as go
import random
import pandas as pd
from decimal import Decimal
import faicons as fa


# Define rotation types as constants
class RotationType:
    RT_WHD = 0
    RT_HWD = 1
    RT_HDW = 2
    RT_DHW = 3
    RT_DWH = 4
    RT_WDH = 5

    ALL = [RT_WHD, RT_HWD, RT_HDW, RT_DHW, RT_DWH, RT_WDH]


# Define axis constants
class Axis:
    WIDTH = 0
    HEIGHT = 1
    DEPTH = 2

    ALL = [WIDTH, HEIGHT, DEPTH]


def rect_intersect(item1, item2, x, y):
    d1 = item1.get_dimension()
    d2 = item2.get_dimension()

    cx1 = item1.position[x] + d1[x] / 2
    cy1 = item1.position[y] + d1[y] / 2
    cx2 = item2.position[x] + d2[x] / 2
    cy2 = item2.position[y] + d2[y] / 2

    ix = max(cx1, cx2) - min(cx1, cx2)
    iy = max(cy1, cy2) - min(cy1, cy2)

    return ix < (d1[x] + d2[x]) / 2 and iy < (d1[y] + d2[y]) / 2


def intersect(item1, item2):
    return (
        rect_intersect(item1, item2, Axis.WIDTH, Axis.HEIGHT)
        and rect_intersect(item1, item2, Axis.HEIGHT, Axis.DEPTH)
        and rect_intersect(item1, item2, Axis.WIDTH, Axis.DEPTH)
    )


def get_limit_number_of_decimals(number_of_decimals):
    return Decimal("1.{}".format("0" * number_of_decimals))


def set_to_decimal(value, number_of_decimals):
    number_of_decimals = get_limit_number_of_decimals(number_of_decimals)

    return Decimal(value).quantize(number_of_decimals)


DEFAULT_NUMBER_OF_DECIMALS = 3
START_POSITION = [0, 0, 0]


class Item:
    def __init__(self, name, width, height, depth, weight):
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.weight = weight
        self.rotation_type = 0
        self.position = START_POSITION
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS

    def format_numbers(self, number_of_decimals):
        self.width = set_to_decimal(self.width, number_of_decimals)
        self.height = set_to_decimal(self.height, number_of_decimals)
        self.depth = set_to_decimal(self.depth, number_of_decimals)
        self.weight = set_to_decimal(self.weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        return "%s(%sx%sx%s, weight: %s) pos(%s) rt(%s) vol(%s)" % (
            self.name,
            self.width,
            self.height,
            self.depth,
            self.weight,
            self.position,
            self.rotation_type,
            self.get_volume(),
        )

    def get_volume(self):
        return set_to_decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )

    def get_dimension(self):
        if self.rotation_type == RotationType.RT_WHD:
            dimension = [self.width, self.height, self.depth]
        elif self.rotation_type == RotationType.RT_HWD:
            dimension = [self.height, self.width, self.depth]
        elif self.rotation_type == RotationType.RT_HDW:
            dimension = [self.height, self.depth, self.width]
        elif self.rotation_type == RotationType.RT_DHW:
            dimension = [self.depth, self.height, self.width]
        elif self.rotation_type == RotationType.RT_DWH:
            dimension = [self.depth, self.width, self.height]
        elif self.rotation_type == RotationType.RT_WDH:
            dimension = [self.width, self.depth, self.height]
        else:
            dimension = []

        return dimension


class Bin:
    def __init__(self, name, width, height, depth, max_weight):
        self.name = name
        self.width = width
        self.height = height
        self.depth = depth
        self.max_weight = max_weight
        self.items = []
        self.unfitted_items = []
        self.number_of_decimals = DEFAULT_NUMBER_OF_DECIMALS

    def format_numbers(self, number_of_decimals):
        self.width = set_to_decimal(self.width, number_of_decimals)
        self.height = set_to_decimal(self.height, number_of_decimals)
        self.depth = set_to_decimal(self.depth, number_of_decimals)
        self.max_weight = set_to_decimal(self.max_weight, number_of_decimals)
        self.number_of_decimals = number_of_decimals

    def string(self):
        return "%s(%sx%sx%s, max_weight:%s) vol(%s)" % (
            self.name,
            self.width,
            self.height,
            self.depth,
            self.max_weight,
            self.get_volume(),
        )

    def get_volume(self):
        return set_to_decimal(
            self.width * self.height * self.depth, self.number_of_decimals
        )

    def get_total_weight(self):
        total_weight = 0

        for item in self.items:
            total_weight += item.weight

        return set_to_decimal(total_weight, self.number_of_decimals)

    def put_item(self, item, pivot):
        fit = False
        valid_item_position = item.position
        item.position = pivot

        for i in range(0, len(RotationType.ALL)):
            item.rotation_type = i
            dimension = item.get_dimension()
            if (
                self.width < pivot[0] + dimension[0]
                or self.height < pivot[1] + dimension[1]
                or self.depth < pivot[2] + dimension[2]
            ):
                continue

            fit = True

            for current_item_in_bin in self.items:
                if intersect(current_item_in_bin, item):
                    fit = False
                    break

            if fit:
                if self.get_total_weight() + item.weight > self.max_weight:
                    fit = False
                    return fit

                self.items.append(item)

            if not fit:
                item.position = valid_item_position

            return fit

        if not fit:
            item.position = valid_item_position

        return fit


class Packer:
    def __init__(self):
        self.bins = []
        self.items = []
        self.unfit_items = []
        self.total_items = 0

    def add_bin(self, bin):
        return self.bins.append(bin)

    def add_item(self, item):
        self.total_items = len(self.items) + 1

        return self.items.append(item)

    def pack_to_bin(self, bin, item):
        fitted = False

        if not bin.items:
            response = bin.put_item(item, START_POSITION)

            if not response:
                bin.unfitted_items.append(item)

            return

        for axis in range(0, 3):
            items_in_bin = bin.items

            for ib in items_in_bin:
                pivot = [0, 0, 0]
                w, h, d = ib.get_dimension()
                if axis == Axis.WIDTH:
                    pivot = [ib.position[0] + w, ib.position[1], ib.position[2]]
                elif axis == Axis.HEIGHT:
                    pivot = [ib.position[0], ib.position[1] + h, ib.position[2]]
                elif axis == Axis.DEPTH:
                    pivot = [ib.position[0], ib.position[1], ib.position[2] + d]

                if bin.put_item(item, pivot):
                    fitted = True
                    break
            if fitted:
                break

        if not fitted:
            bin.unfitted_items.append(item)

    def pack(
        self,
        bigger_first=False,
        distribute_items=False,
        number_of_decimals=DEFAULT_NUMBER_OF_DECIMALS,
    ):
        for bin in self.bins:
            bin.format_numbers(number_of_decimals)

        for item in self.items:
            item.format_numbers(number_of_decimals)

        self.bins.sort(key=lambda bin: bin.get_volume(), reverse=bigger_first)
        self.items.sort(key=lambda item: item.get_volume(), reverse=bigger_first)

        for bin in self.bins:
            for item in self.items:
                self.pack_to_bin(bin, item)

            if distribute_items:
                for item in bin.items:
                    self.items.remove(item)


def create_3d_box(x, y, z, dx, dy, dz, color, name=None):
    """Creates 3D box data for Plotly visualization with colored faces."""
    faces = []
    # Bottom face
    faces.append(
        go.Mesh3d(
            x=[x, x + dx, x + dx, x],
            y=[y, y, y + dy, y + dy],
            z=[z, z, z, z],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    # Top face
    faces.append(
        go.Mesh3d(
            x=[x, x + dx, x + dx, x],
            y=[y, y, y + dy, y + dy],
            z=[z + dz, z + dz, z + dz, z + dz],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    # Front face
    faces.append(
        go.Mesh3d(
            x=[x, x + dx, x + dx, x],
            y=[y, y, y, y],
            z=[z, z, z + dz, z + dz],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    # Back face
    faces.append(
        go.Mesh3d(
            x=[x, x + dx, x + dx, x],
            y=[y + dy, y + dy, y + dy, y + dy],
            z=[z, z, z + dz, z + dz],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    # Left face
    faces.append(
        go.Mesh3d(
            x=[x, x, x, x],
            y=[y, y + dy, y + dy, y],
            z=[z, z, z + dz, z + dz],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    # Right face
    faces.append(
        go.Mesh3d(
            x=[x + dx, x + dx, x + dx, x + dx],
            y=[y, y + dy, y + dy, y],
            z=[z, z, z + dz, z + dz],
            color=color,
            opacity=0.5,
            i=[0, 1, 2, 3],
            j=[1, 2, 3, 0],
            k=[2, 3, 0, 1],
            name=name,
        )
    )
    return faces


def random_color():
    """Generates a random color in RGB format."""
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    return f"rgb({r},{g},{b})"


# Définir l'interface utilisateur
app_ui = ui.page_fluid(
    ui.tags.style(
        """
        .top-right-button {
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 10; /* Ensure the button is on top of other elements */
        }
        .card {
            position: relative; /* Required for the absolute positioning of the button */
            padding-top: 40px; /* Add padding to avoid overlap with content */
        }
    """
    ),
    ui.h1("📦 3D Bin Packing Tool"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.card(
                ui.card_header("Instructions"),
                ui.p("1. Configure bins and items."),
                ui.p("2. Click 'Add Bin' and 'Add Item' to add them to the list."),
                ui.p("3. Click 'Pack & Plot' to visualize the packing."),
                ui.p("4. Select a bin to see the packing details."),
                ui.input_action_button(
                    "clear_all", "Clear All", icon=fa.icon_svg("broom")
                ),
                ui.input_action_button(
                    "pack_and_plot",
                    "Pack & Plot",
                    class_="btn-primary",
                    icon=fa.icon_svg("play"),
                ),
                ui.card_footer("miskowski85@hotmail.fr"),
            ),
        ),
        ui.row(
            ui.column(12, ui.h2("Current Bins and Items")),
            ui.layout_column_wrap(
                ui.card(
                    ui.h3("Bins"),
                    ui.input_action_button(
                        "add_bin",
                        "Add",
                        class_="top-right-button",
                        icon=fa.icon_svg("plus"),
                    ),
                    ui.layout_column_wrap(
                        ui.input_numeric("bin_width", "Width", value=25.0, step=0.1),
                        ui.input_numeric("bin_height", "Height", value=22.0, step=0.1),
                        ui.input_numeric("bin_depth", "Depth", value=5.75, step=0.1),
                        ui.input_numeric("bin_max_weight", "MaxWeight", value=25),
                        width=4,
                    ),
                    ui.output_data_frame("bins_table"),
                ),
                ui.card(
                    ui.h3("Items"),
                    ui.input_action_button(
                        "add_item",
                        "Add",
                        class_="top-right-button",
                        icon=fa.icon_svg("plus"),
                    ),
                    ui.layout_column_wrap(
                        ui.input_numeric("item_width", "Width", value=7.8740, step=0.1),
                        ui.input_numeric(
                            "item_height", "Height", value=3.9370, step=0.1
                        ),
                        ui.input_numeric("item_depth", "Depth", value=1.9685, step=0.1),
                        ui.input_numeric("item_weight", "Weight", value=4),
                        width=4,
                    ),
                    ui.output_data_frame("items_table"),
                ),
                width=2,
            ),
            ui.input_select("bin_selector", "Select Bin to Display", choices=[]),
            ui.layout_column_wrap(
                ui.card(ui.output_ui("packing_plot")),
                ui.card(ui.output_ui("packing_summary")),
                width=2,
            ),
        ),
    ),
)


# Définir la logique du serveur
def server(input, output, session):
    bins = reactive.Value([])
    items = reactive.Value([])
    packed_bins = reactive.Value(None)

    @reactive.Effect
    @reactive.event(input.add_bin)
    def _():
        new_bin = {
            "name": f"Bin {len(bins.get()) + 1}",
            "width": input.bin_width(),
            "height": input.bin_height(),
            "depth": input.bin_depth(),
            "max_weight": input.bin_max_weight(),
            "color": f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})",
        }
        bins.set(bins.get() + [new_bin])

    @reactive.Effect
    @reactive.event(input.add_item)
    def _():
        new_item = {
            "name": f"Item {len(items.get()) + 1}",
            "width": input.item_width(),
            "height": input.item_height(),
            "depth": input.item_depth(),
            "weight": input.item_weight(),
            "color": f"rgb({random.randint(0, 255)}, {random.randint(0, 255)}, {random.randint(0, 255)})",
        }
        items.set(items.get() + [new_item])

    @reactive.Effect
    @reactive.event(input.clear_all)
    def _():
        bins.set([])
        items.set([])
        packed_bins.set(None)

    @output
    @render.data_frame
    def bins_table():
        df = pd.DataFrame(bins.get())
        if "color" in df.columns:
            df = df.drop(columns=["color"])
        return df

    @output
    @render.data_frame
    def items_table():
        df = pd.DataFrame(items.get())
        if "color" in df.columns:
            df = df.drop(columns=["color"])
        return df

    @reactive.Effect
    @reactive.event(input.pack_and_plot)
    def _():
        packer = Packer()

        for bin_data in bins.get():
            bin = Bin(
                bin_data["name"],
                bin_data["width"],
                bin_data["height"],
                bin_data["depth"],
                bin_data["max_weight"],
            )
            packer.add_bin(bin)

        for item_data in items.get():
            item = Item(
                item_data["name"],
                item_data["width"],
                item_data["height"],
                item_data["depth"],
                item_data["weight"],
            )
            packer.add_item(item)

        packer.pack(distribute_items=True)
        packed_bins.set(packer.bins)
        # Mettre à jour les choix du sélecteur
        bin_choices = [bin.name for bin in packer.bins]
        ui.update_select(
            "bin_selector",
            choices=bin_choices,
            selected=bin_choices[0] if bin_choices else None,
        )

    @output
    @render.ui
    def packing_plot():
        if packed_bins.get() is None:
            return ui.tags.div("No bins packed yet.")

        selected_bin_name = input.bin_selector()

        data = []
        for bin in packed_bins.get():
            if bin.name == selected_bin_name:
                bin_data = next(b for b in bins.get() if b["name"] == bin.name)
                data.extend(
                    create_3d_box(
                        0,
                        0,
                        0,
                        bin.width,
                        bin.height,
                        bin.depth,
                        bin_data["color"],
                        bin.name,
                    )
                )
                for item in bin.items:
                    item_data = next(i for i in items.get() if i["name"] == item.name)
                    pos_x, pos_y, pos_z = item.position
                    width, height, depth = item.get_dimension()
                    data.extend(
                        create_3d_box(
                            pos_x,
                            pos_y,
                            pos_z,
                            width,
                            height,
                            depth,
                            item_data["color"],
                            item.name,
                        )
                    )

        layout = go.Layout(
            scene=dict(
                xaxis=dict(title="X"),
                yaxis=dict(title="Y"),
                zaxis=dict(title="Z"),
                aspectmode="data",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )

        fig = go.Figure(data=data, layout=layout)
        html = fig.to_html(full_html=False)
        styled_html = f"""
        <div style="width: 100%; height: 80vh; overflow: hidden;">
            {html}
        </div>
        """

        return ui.HTML(styled_html)

    @output
    @render.ui
    def packing_summary():
        if packed_bins.get() is None:
            return ui.p("No bins packed yet.")

        selected_bin_name = input.bin_selector()

        for bin in packed_bins.get():
            if bin.name == selected_bin_name:
                bin_volume = bin.width * bin.height * bin.depth
                total_volume = sum(
                    item.width * item.height * item.depth for item in bin.items
                )
                volume_percentage = (total_volume / bin_volume) * 100
                total_weight = sum(item.weight for item in bin.items)
                weight_percentage = (total_weight / bin.max_weight) * 100

                gauge_fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=volume_percentage,
                        title={"text": "Volume Used (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "darkblue"},
                            "steps": [
                                {"range": [0, 50], "color": "lightgray"},
                                {"range": [50, 75], "color": "gray"},
                                {"range": [75, 100], "color": "green"},
                            ],
                        },
                    )
                )

                gauge_html = gauge_fig.to_html(full_html=False, include_plotlyjs="cdn")

                return ui.div(
                    ui.tags.div(
                        ui.layout_columns(
                            ui.value_box(
                                "Total Volume Used",
                                ui.p(f"{total_volume:.2f} cubic units"),
                                showcase=fa.icon_svg("calculator"),
                            ),
                            ui.value_box(
                                "Total Items Packed",
                                ui.p(f"{len(bin.items)} items"),
                                showcase=fa.icon_svg("cubes"),
                            ),
                            ui.value_box(
                                "Weight",
                                ui.p(f"{weight_percentage:.2f} %"),
                                showcase=fa.icon_svg("weight-hanging"),
                            ),
                            fill=False,
                        ),
                        # style="display: flex; justify-content: space-around;"
                    ),
                    ui.HTML(
                        f"""
                    <div style="width: 100%; height: 300px;">
                        {gauge_html}
                    </div>
                """
                    ),
                )

        return ui.p("No bin selected.")


# Créer l'application Shiny
app = App(app_ui, server)
