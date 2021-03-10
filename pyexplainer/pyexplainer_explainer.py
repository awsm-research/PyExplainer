import json
import string
import os
from sklearn.utils import check_random_state
import ipywidgets as widgets
from IPython.core.display import display, HTML
from IPython.display import clear_output

"""
remember to remove the line under "# debugging purpose - to be removed
"""


def data_validation(data):
    """Validate the if the given data format is a list of dictionary.

    Parameters
    ----------
    param1 : :obj:`Any`
        Data to be validated.

    Returns
    -------
    :obj:`bool`
        True: The data is a list of dictionary.\n
        False: The data is not a list of dictionary.

    Examples
    --------
    >>> from pypkgs import pypkgs
    >>> a = pd.Categorical(["character", "hits", "your", "eyeballs"])
    >>> b = pd.Categorical(["but", "integer", "where it", "counts"])
    >>> pypkgs.catbind(a, b)
    [character, hits, your, eyeballs, but, integer, where it, counts]
    Categories (8, object): [but, character, counts,
    eyeballs, hits, integer, where it, your]
    """
    valid = True
    if str(type(data)) == "<class 'list'>":
        for i in range(len(data)):
            if str(type(data[i])) != "<class 'dict'>":
                print(
                    "Data Format Error - the input data should be a list of dictionary")
                valid = False
                break
    else:
        valid = False
    return valid


def id_generator(size=15, random_state=None):
    """Generate unique ids for div tag which will contain the visualisation stuff from d3.

    Parameters
    ----------
    param1 : :obj:`int`
        An integer that specifies the length of the returned id, default = 15.
    param2 : :obj:`np.random.RandomState`, default is None.
        A RandomState instance.

    Returns
    -------
    :obj:`str`
        A random identifier.

    Examples
    --------
    >>> from pypkgs import pypkgs
    >>> a = pd.Categorical(["character", "hits", "your", "eyeballs"])
    >>> b = pd.Categorical(["but", "integer", "where it", "counts"])
    >>> pypkgs.catbind(a, b)
    [character, hits, your, eyeballs, but, integer, where it, counts]
    Categories (8, object): [but, character, counts,
    eyeballs, hits, integer, where it, your]
    """

    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(random_state.choice(chars, size, replace=True))


class Explainer():
    """ __init__ method of Explainer class.

    Parameters
    ----------
    param1 : :obj:`list` of :obj:`dict`
        A list of dictionary that contains the specification of the bullet chart to be created.
    param2 : :obj:`list` of :obj:`dict`
        A list of dictionary that contains the information about risk scores.
    param3 : :obj:`np.random.RandomState`, default is None
        A RandomState instance.

    """

    def __init__(self, bullet_data, risk_data, random_state=None):
        valid_bullet_data = data_validation(bullet_data)
        if valid_bullet_data:
            self.set_bullet_data(bullet_data)
        else:
            self.set_bullet_data([{}])
            print(
                "Bullet Data Format Error - the input data should be a list of dictionary")

        valid_risk_data = data_validation(risk_data)
        if valid_risk_data:
            self.set_risk_data(risk_data)
        else:
            self.set_risk_data([{}])
            print(
                "Risk Data Format Error - the input data should be a list of dictionary")

        self.random_state = random_state
        # add setter later
        self.bullet_output = widgets.Output(
            layout={'border': '3px solid black'})
        self.hbox_items = []

    def generate_sliders(self):
        slider_widgets = []
        data = self.get_bullet_data()
        style = {'description_width': '40%', 'widget_width': '60%'}
        layout = widgets.Layout(width='99%', height='20px')

        for d in data:
            # decide to use either IntSlider or FloatSlider
            if isinstance(d['step'], int):
                # create IntSlider obj and store it into a list
                slider = widgets.IntSlider(
                    value=d['markers'][0],
                    min=d['ticks'][0],
                    max=d['ticks'][-1],
                    step=d['step'][0],
                    description=d['title'],
                    layout=layout,
                    style=style,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='d'
                    )
                slider_widgets.append(slider)
            else:
                # create FloatSlider obj and store it into a list
                slider = widgets.FloatSlider(
                    value=d['markers'][0],
                    min=d['ticks'][0],
                    max=d['ticks'][-1],
                    step=d['step'][0],
                    description=d['title'],
                    layout=layout,
                    style=style,
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f'
                    )
                slider_widgets.append(slider)
        return slider_widgets

    def generate_html(self):
        """Generate html and return it as a String.

        Returns
        ----------
        :obj:`str`
            html String

        Examples
        --------
        >>> from pypkgs import pypkgs
        >>> a = pd.Categorical(["character", "hits", "your", "eyeballs"])
        >>> b = pd.Categorical(["but", "integer", "where it", "counts"])
        >>> pypkgs.catbind(a, b)
        [character, hits, your, eyeballs, but, integer, where it, counts]
        Categories (8, object): [but, character, counts,
        eyeballs, hits, integer, where it, your]
        """

        css_filepath = "css/styles.css"
        css_stylesheet = """
            <link rel="stylesheet" href="%s" />
        """ % (css_filepath)

        d3_filepath = "js/d3.min.js"
        bulletjs_filepath = "js/bullet.js"
        d3_script = """
        <script src="%s"></script>
        <script src="%s"></script>
        """ % (d3_filepath, bulletjs_filepath)

        main_title = "What to do to decrease the risk of having defects?"
        title = """
        <div style="position: relative; top: 0; width: 100vw; text-align: center">
            <b>%s</b>
        </div>
        """ % main_title

        unique_id = id_generator(
            random_state=check_random_state(self.get_random_state()))
        bullet_data = self.to_js_data(self.get_bullet_data())
        risk_data = self.to_js_data(self.get_risk_data())

        d3_operation_script = """
        <script>

        var margin = { top: 5, right: 40, bottom: 20, left: 500 },
          width = 1300 - margin.left - margin.right,
          height = 50 - margin.top - margin.bottom;

        var chart = d3.bullet().width(width).height(height);

        var bulletData = %s

        var riskData = %s

        // define the color of the box
        var boxColor = "box green";
        var riskPred = riskData[0].riskPred[0];
        if (riskPred.localeCompare("Yes")==0) {
            boxColor = "box orange";
        }

        // append risk prediction and risk score
        d3.select("#d3-target-bullet-%s")
          .append("div")
          .attr("class", "riskPred")
          .data(riskData)
          .text((d) => d.riskPred)
          .append("div")
          .attr("class", boxColor);

        d3.select("#d3-target-bullet-%s")
          .append("div")
          .attr("class", "riskScore")
          .data(riskData)
          .text((d) => "Risk Score: " + d.riskScore);

        var svg = d3
          .select("#d3-target-bullet-%s")
          .selectAll("svg")
          .data(bulletData)
          .enter()
          .append("svg")
          .attr("class", "bullet")
          .attr("width", width + margin.left + margin.right)
          .attr("height", height + margin.top + margin.bottom)
          .append("g")
          .attr(
            "transform",
            "translate(" + margin.left + "," + margin.top + ")"
          )
          .call(chart);

        var title = svg
          .append("g")
          .style("text-anchor", "end")
          .attr("transform", "translate(-6," + height / 2 + ")");

        title
          .append("text")
          .attr("class", "title")
          .text((d) => d.title);

        title
          .append("text")
          .attr("class", "subtitle")
          .attr("dy", "1em")
          .text((d) => d.subtitle);

        </script>
        """ % (bullet_data, risk_data, unique_id, unique_id, unique_id)

        html = """
        <!DOCTYPE html>
        <html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head>
            %s
            %s
        </head>
        <body>
            <div class="bullet-chart">
                %s
                <div class="d3-target-bullet" id="d3-target-bullet-%s" />
            </div>
            %s
        </body>
        </html>
        """ % (css_stylesheet, d3_script, title, unique_id, d3_operation_script)

        return html

    def generate_progress_bar_items(self):
        progress_bar = widgets.FloatProgress(
            value=0,
            min=0,
            max=100,
            bar_style='info',
            layout=widgets.Layout(width='40%'),
            orientation='horizontal',
        )

        left_text = widgets.Label("Risk Score: ")
        right_text = widgets.Label("0")
        self.set_hbox_items(
            [left_text, progress_bar, right_text, widgets.Label("%")])

    def get_bullet_data(self):
        return self.bullet_data

    def get_hbox_items(self):
        return self.hbox_items

    def get_random_state(self):
        return self.random_state

    def get_risk_data(self):
        return self.risk_data

    def get_risk_pred(self):
        return self.get_risk_data()[0]['riskPred'][0]

    def get_risk_score(self):
        risk_score = self.get_risk_data()[0]['riskScore'][0].strip("%")
        return float(risk_score)

    def on_value_change(self, change):
        # get the id to refer to the specific variable and calculation
        description = change['owner'].description.split(" ")[0].strip()
        # update risk score progress bar
        # update risk score
        self.set_risk_score(self.get_risk_score()+10)
        # update risk score text
        self.run_bar_animation()
        # update d3 bullet chart
        out = self.bullet_output
        out.clear_output()
        # debugging purpose - to be removed
        from time import sleep
        sleep(1)
        with out:
            # display d3 bullet chart
            html = self.generate_html()
            display(HTML(html))

    def run_bar_animation(self):
        import time
        items_in_hbox = self.get_hbox_items()
        progress_bar = items_in_hbox[1]

        risk_score = self.get_risk_score()
        risk_prediction = True
        if self.get_risk_pred().upper() == "NO":
            risk_prediction = False
        if risk_prediction:
            progress_bar.style = {'bar_color': '#FA8128'}
        else:
            progress_bar.style = {'bar_color': '#00FF00'}

        # play speed of the animation
        play_speed = 1
        # progress bar animation
        # count start from the current val of the progress bar
        count = progress_bar.value
        left_text = items_in_hbox[0]
        right_text = items_in_hbox[2]
        while count < risk_score:
            progress_bar.value += play_speed  # signal to increment the progress bar
            new_progress_value = float(right_text.value) + play_speed

            if new_progress_value > risk_score:
                right_text.value = str(risk_score)
            else:
                right_text.value = str(new_progress_value)
            time.sleep(.01)
            count += play_speed
        # update the right text
        self.set_right_text(right_text)

    def set_bullet_data(self, bullet_data):
        self.bullet_data = bullet_data

    def set_hbox_items(self, hbox_items):
        self.hbox_items = hbox_items

    def set_random_state(self, random_state):
        self.random_state = random_state

    def set_risk_data(self, risk_data):
        self.risk_data = risk_data

    def set_risk_score(self, risk_score):
        risk_score = str(risk_score) + '%'
        self.get_risk_data()[0]['riskScore'][0] = risk_score

    def set_right_text(self, right_text):
        self.get_hbox_items()[2] = right_text

    def show_visualisation(self):
        """Display the html string in a cell of Jupyter Notebook.

        Examples
        --------
        >>> from pypkgs import pypkgs
        >>> a = pd.Categorical(["character", "hits", "your", "eyeballs"])
        >>> b = pd.Categorical(["but", "integer", "where it", "counts"])
        >>> pypkgs.catbind(a, b)
        [character, hits, your, eyeballs, but, integer, where it, counts]
        Categories (8, object): [but, character, counts,
        eyeballs, hits, integer, where it, your]
        """
        # display risk score progress bar
        self.generate_progress_bar_items()
        items = self.get_hbox_items()
        display(widgets.HBox(items))
        self.run_bar_animation()

        # display sliders
        sliders = self.generate_sliders()
        for slider in sliders:
            slider.observe(self.on_value_change, names='value')
            display(slider)

        out = self.bullet_output
        out.clear_output()
        display(out)
        with out:
            # display d3 bullet chart
            html = self.generate_html()
            display(HTML(html))

    def to_js_data(self, list_of_dict):
        return (str(list_of_dict) + ";")
