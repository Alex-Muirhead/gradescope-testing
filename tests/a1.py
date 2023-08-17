__author__ = "Alex Muirhead"

import math
import inspect
import types
from a1_support import HELP, INVALID, MAIN_PROMPT, load_data

CURVE_COEFFS = (-2.579E-03, 2.311E-02, -2.155E-03, 3.703E-05, -1.367E-06)
W_TO_KW = 1E-03
DOLLAR_PER_KWH = 0.2


def demo_format(*output_names):

    if output_names == []:
        output_names = ["output"]

    def demo(self, *args, **kwargs):
        str_kwargs = list(map(lambda x: f"{x[0]}={x[1]}", kwargs.items()))
        str_args   = list(map(str, args)) + str_kwargs
        call       = f"{self.__name__}({', '.join(str_args)})"
        assignment = ", ".join(output_names)
        if assignment:
            assignment += " = "
        # Print output in-case the eval fails!
        print(">>>", assignment+call)

        frame = inspect.currentframe().f_back
        try:
            values = eval(call, frame.f_globals, frame.f_locals)
        except NameError as error:
            raise error.with_traceback(None)
        finally:
            del frame

        if not isinstance(values, tuple):
            values = (values,)
        for var, val in zip(output_names, values):
            print(">>>", var)
            print(val)

        return values

    def wrapper(func):
        func.demo = types.MethodType(demo, func)
        return func

    return wrapper


class Table:

    def __init__(self, alignment='^', widths=-1, sep='#'):
        from itertools import repeat

        # Allow for a single value to be copied across table
        if isinstance(alignment, str):
            alignment = repeat(alignment)
        if isinstance(widths, int):
            widths = repeat(widths)

        self.alignment = alignment
        self.widths = widths
        self.sep = sep
        self._pad = 1

    def format(self, content):
        from itertools import chain, zip_longest
        assert isinstance(content, dict), "Content must be a dictionary"

        header_values = content.keys()
        table_values = zip(*content.values())  # Transpose the table values

        # Allow for width of -1 to mean self-padding based on header
        widths = [
            w-2*self._pad if (w != -1) else len(h) for w, h in
            zip(self.widths, header_values)
        ]

        header = self._format_row(header_values, widths, header=True)
        rule = self.sep * len(header)
        rows = [self._format_row(items, widths) for items in table_values]

        return '\n'.join(chain([rule, header, rule], rows, [rule]))

    def _format_row(self, items, widths, header=False):
        from itertools import chain, repeat
        alignment = [] if header else self.alignment
        rounding = {float: '.2f'}
        cells = [
            f"{i:{a}{w}{rounding.get(type(i), '')}}" for (i, a, w)
            in zip(items, chain(alignment, repeat('^')), widths)
        ]
        sep = ' '*self._pad + self.sep + ' '*self._pad
        row = sep + sep.join(cells) + sep
        return row.strip()


@demo_format("v_hub")
def determine_hub_speed(v_meas, h_meas, h_hub, alpha):
    """Determine the windspeed at the hub, from ground data.

    Parameters
    ----------
    v_meas: float
        Windspeed measured at the ground
    h_meas: float
        Height of at which the windspeed is measured
    h_hub: float
        Height of the hub
    alpha: float
        Correlation coefficient

    Returns
    -------
    float:
        The speed at the hub
    """
    return v_meas * pow(h_hub / h_meas, alpha)


@demo_format("P_wind")
def determine_windpower(v_hub, r, rho):
    """Determine the power in given wind conditions

    Parameters
    ----------
    v_hub: float
        The windspeed at the turbine hub
    r: float
        The radius of the turbine blades
    rho: float
        The air density

    Returns
    -------
    float:
        The power contained in the wind
    """
    area = math.pi * (r**2)
    dyn_pres = 0.5 * rho * (v_hub**2)
    return area * dyn_pres * v_hub * W_TO_KW


@demo_format("coeff_p")
def determine_mech_coeff(v_hub, radius, omega, coefs):

    result = 0
    ratio = radius * omega / v_hub
    for i, a in enumerate(coefs, 1):
        result += a * ratio**i

    return result


@demo_format("P_mech")
def determine_mech_power(v_hub, speeds, radius, omega, rho, coefs):
    v_cutin, v_cutout, v_rated = speeds

    # Handle cases where the turbine is shut off
    if v_hub < v_cutin or v_hub >= v_cutout:
        return 0

    # Cap the velocity at the rated speed
    v_hub = min(v_hub, v_rated)

    coeff = determine_mech_coeff(v_hub, radius, omega, coefs)
    power = coeff * determine_windpower(v_hub, radius, rho)  # [kW]

    return power


@demo_format("daily_revenue", "average_power")
def determine_revenue(speeds, h_meas, h_hub, alpha, r, omega, rho, coeffs):
    # Extract value for this case
    v_rated = speeds[2]
    total_energy = 0

    for hour in range(0, 24):
        # Step 1. Predict v_meas
        v_meas = (1.0 + 0.2 * math.cos(math.tau * (hour - 12) / 24)) * v_rated
        # Step 2. Calculate v_hub
        v_hub = determine_hub_speed(v_meas, h_meas, h_hub, alpha)
        # Step 3. Calculate mechanical power
        power = determine_mech_power(v_hub, speeds, r, omega, rho, coeffs)

        # Step 4. Accumulate power
        total_energy += 0.9 * power  # [kWh]

    average_power = total_energy / 24
    daily_revenue = total_energy * DOLLAR_PER_KWH

    return daily_revenue, average_power


@demo_format()
def print_table(parameters, separator):
    print_table_general(parameters, 2, separator)


@demo_format()
def print_table_general(parameters, number_columns, separator):
    table_content = {"Case number": [], "Daily revenue ($)": [], "Ave power (kW)": []}
    for case_num, case_params in enumerate(parameters, 1):
        revenue, avg_power = determine_revenue(*case_params)

        table_content["Case number"].append(case_num)
        table_content["Daily revenue ($)"].append(revenue)
        table_content["Ave power (kW)"].append(avg_power)

    if number_columns == 2:
        table_content.pop("Ave power (kW)")

    # Can't be bothered re-implementing this
    table = Table(widths=25, sep=separator)
    print(table.format(table_content), end='\n\n')


def pack(raw_parameters):
    packed_params = ()
    for param_group in raw_parameters:
        speeds, param_group = param_group[:3], param_group[3:]
        param_group, coeffs = param_group[:6], param_group[6:]
        packed_params += ((tuple(speeds), *param_group, tuple(coeffs)),)
    return packed_params


def main():

    parameters = ()

    while True:

        command = input(MAIN_PROMPT)
        match command.strip().split(' '):
            case ('h',):
                print(HELP)
            case ('q',):
                confirmation = input("Are you sure (y/n): ")
                if confirmation.strip().lower() not in ['y', 'n']:
                    print(INVALID)
                    continue
                if confirmation == 'y':
                    break
            case ('r',):
                directory  = input("Please specify the directory: ")
                file_name  = input("Please specify the filename: ")
                parameters = pack(load_data(directory, file_name))
            case ('p', num_cols, sep):
                if not parameters:
                    print("Can't print a table without parameters!")
                    continue

                num_cols = int(num_cols)
                print_table_general(parameters, num_cols, sep)
            case _:
                print(INVALID)


if __name__ == "__main__":

    height  = 60  # [m]
    radius  = 40  # [m]
    density = 1.225   # [kg/m3]
    speeds  = (2, 18, 12)  # [m/s]
    windspeed = 9
    omega = 2.31

    windspeed, = determine_hub_speed.demo(windspeed, 10, height, 0.14)
    windspeed = round(windspeed, 3)

    determine_windpower.demo(windspeed, radius, density)

    determine_mech_coeff.demo(windspeed, radius, omega, "CURVE_COEFFS")

    determine_mech_power.demo(windspeed, "speeds", radius, omega, density, "CURVE_COEFFS")

    determine_revenue.demo("speeds", 10, height, 0.14, radius, omega, density, "CURVE_COEFFS")

    # FIXME: These will need to be tweaked, so they give reasonable variation
    long_coeff_list = (
        (-2.579E-03, +2.311E-02, -2.155E-03, +3.703E-05, -1.367E-06),
        (-6.798E-03, +3.552E-02, -4.583E-03, +1.395E-04),
        (+1.338E-03, +1.604E-02,    0.0E-00,    0.0E-00, -6.220E-06)
    )
    long_parameter_list = (
        ((2, 18,   12), 10, 60, 0.14, 40, 2.3, density, long_coeff_list[0]),
        ((2, 17,   11), 10, 70, 0.14, 45, 1.8, density, long_coeff_list[1]),
        ((2, 16, 10.5), 10, 50, 0.14, 35, 2.2, density, long_coeff_list[2]),
    )
    print_table.demo("long_parameter_list", "'*'")
    print_table_general.demo("long_parameter_list", 3, "'@'")
