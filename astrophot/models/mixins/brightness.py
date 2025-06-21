from ...param import forward


class RadialMixin:

    @forward
    def brightness(self, x, y):
        """
        Calculate the brightness at a given point (x, y) based on radial distance from the center.
        """
        x, y = self.transform_coordinates(x, y)
        return self.radial_model(self.radius_metric(x, y))
