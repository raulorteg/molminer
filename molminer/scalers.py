import json


class PropertyScaler:
    """
    Scales and unscales molecular property values using precomputed statistics.

    Parameters
    ----------
    stats_path : str
        Path to a JSON file with 'mean' and 'std' values for each property.
    """

    def __init__(self, stats_path: str) -> None:
        with open(stats_path, "r") as f:
            self.stats = json.load(f)

    def scale(self, value: float, property_name: str) -> float:
        """
        Scale a value to zero mean and unit variance.

        Parameters
        ----------
        value : float
            Original property value.
        property_name : str
            Name of the property.

        Returns
        -------
        float
            Scaled value.
        """
        assert property_name in self.stats.keys(), (
            f"Property name <{property_name}> not in stats file."
        )

        return (value - self.stats[property_name]["mean"]) / self.stats[property_name][
            "std"
        ]

    def unscale(self, scaled_value: float, property_name: str) -> float:
        """
        Convert a scaled value back to its original scale.

        Parameters
        ----------
        scaled_value : float
            Scaled value.
        property_name : str
            Name of the property.

        Returns
        -------
        float
            Original property value.
        """
        assert property_name in self.stats.keys(), (
            f"Property name <{property_name}> not in stats file."
        )

        return (
            scaled_value * self.stats[property_name]["std"]
            + self.stats[property_name]["mean"]
        )

    def get(self, statistic: str, property_name: str) -> float:
        """
        Get a statistic (e.g. 'mean', 'std') for a property.

        Parameters
        ----------
        statistic : str
            Name of the statistic.
        property_name : str
            Name of the property.

        Returns
        -------
        float
            Requested statistic value.
        """
        return self.stats[property_name][statistic]
